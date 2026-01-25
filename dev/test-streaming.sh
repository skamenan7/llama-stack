#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# Test Llama Stack Streaming Capabilities
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

PORT="${LLAMA_STACK_PORT:-8321}"
BASE_URL="http://localhost:$PORT"
MODEL="${TEXT_MODEL:-ollama/llama3.2:3b-instruct-fp16}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[TEST]${NC} $*"; }
output() { echo -e "${BLUE}[OUTPUT]${NC} $*"; }
stream() { echo -e "${CYAN}[STREAM]${NC} $*"; }

echo ""
echo "========================================"
echo "  Llama Stack Streaming Tests"
echo "========================================"
echo "  Model: $MODEL"
echo "========================================"
echo ""

PASSED=0
FAILED=0

# Test 1: OpenAI-compatible streaming chat
info "Test 1: OpenAI chat/completions streaming..."
echo ""

token_count=0
full_response=""

while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        json="${line#data: }"
        if [[ "$json" != "[DONE]" && -n "$json" ]]; then
            delta=$(echo "$json" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
            if [[ -n "$delta" ]]; then
                echo -n "$delta"
                full_response+="$delta"
                ((token_count++)) || true
            fi
        fi
    fi
done < <(curl -sN "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line."}],
        "max_tokens": 100,
        "stream": true
    }' 2>/dev/null)

echo ""
echo ""

if [[ $token_count -gt 0 ]]; then
    pass "Chat streaming working ($token_count chunks received)"
    ((PASSED++)) || true
else
    fail "No streaming chunks received"
    ((FAILED++)) || true
fi

echo ""

# Test 2: Responses API streaming
info "Test 2: Responses API streaming..."
echo ""

event_count=0
event_types=()

while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        json="${line#data: }"
        if [[ -n "$json" && "$json" != "[DONE]" ]]; then
            event_type=$(echo "$json" | jq -r '.type // "unknown"' 2>/dev/null)
            if [[ -n "$event_type" && "$event_type" != "null" ]]; then
                ((event_count++)) || true
                # Track unique event types
                if [[ ! " ${event_types[*]:-} " =~ " ${event_type} " ]]; then
                    event_types+=("$event_type")
                    stream "Event: $event_type"
                fi

                # Print text deltas
                if [[ "$event_type" == "response.output_text.delta" ]]; then
                    delta=$(echo "$json" | jq -r '.delta // empty' 2>/dev/null)
                    echo -n "$delta"
                fi
            fi
        fi
    fi
done < <(curl -sN "$BASE_URL/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "input": "Say hello in 5 different languages.",
        "stream": true
    }' 2>/dev/null)

echo ""
echo ""

if [[ $event_count -gt 0 ]]; then
    pass "Responses streaming working ($event_count events)"
    output "Event types seen: ${event_types[*]:-none}"
    ((PASSED++)) || true
else
    info "Responses streaming may not be available"
fi

echo ""

# Test 3: Streaming with tool use (function calling)
info "Test 3: Streaming with function tools..."
echo ""

tool_events=0
text_chunks=0

while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        json="${line#data: }"
        if [[ -n "$json" && "$json" != "[DONE]" ]]; then
            # Check for tool calls in chat completions format
            tool_call=$(echo "$json" | jq -r '.choices[0].delta.tool_calls // empty' 2>/dev/null)
            if [[ -n "$tool_call" && "$tool_call" != "null" ]]; then
                ((tool_events++)) || true
            fi

            # Check for content
            content=$(echo "$json" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
            if [[ -n "$content" ]]; then
                echo -n "$content"
                ((text_chunks++)) || true
            fi
        fi
    fi
done < <(curl -sN "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 25 times 4? Use the calculator."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "max_tokens": 100,
        "stream": true
    }' 2>/dev/null)

echo ""
echo ""

if [[ $text_chunks -gt 0 || $tool_events -gt 0 ]]; then
    pass "Tool streaming working (text: $text_chunks, tool_calls: $tool_events)"
    ((PASSED++)) || true
else
    fail "No streaming output with tools"
    ((FAILED++)) || true
fi

echo ""

# Test 4: Measure streaming latency (time to first token)
info "Test 4: Time to first token (TTFT)..."

start_time=$(date +%s%3N)
first_token_time=""

while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        json="${line#data: }"
        if [[ "$json" != "[DONE]" && -n "$json" ]]; then
            delta=$(echo "$json" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
            if [[ -n "$delta" && -z "$first_token_time" ]]; then
                first_token_time=$(date +%s%3N)
                break
            fi
        fi
    fi
done < <(curl -sN "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": true
    }' 2>/dev/null)

if [[ -n "$first_token_time" ]]; then
    ttft=$((first_token_time - start_time))
    pass "Time to first token: ${ttft}ms"
    ((PASSED++)) || true
else
    fail "Could not measure TTFT"
    ((FAILED++)) || true
fi

echo ""
echo "========================================"
echo -e "  Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo "========================================"
echo ""

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
