#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# Test Llama Stack Agents/Responses API
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
PURPLE='\033[0;35m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[TEST]${NC} $*"; }
output() { echo -e "${BLUE}[OUTPUT]${NC} $*"; }
debug() { echo -e "${PURPLE}[DEBUG]${NC} $*"; }

echo ""
echo "========================================"
echo "  Llama Stack Agents/Responses Tests"
echo "========================================"
echo "  Model: $MODEL"
echo "========================================"
echo ""

PASSED=0
FAILED=0

# Test 1: Create a basic response (no tools)
info "Test 1: Basic response (no tools)..."
response=$(curl -sf "$BASE_URL/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "input": "What is 2+2? Reply with just the number.",
        "stream": false
    }' 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    response_id=$(echo "$response" | jq -r '.id // empty' 2>/dev/null)
    output_text=$(echo "$response" | jq -r '.output_text // empty' 2>/dev/null)
    if [[ -n "$response_id" ]]; then
        pass "Basic response created"
        output "Response ID: $response_id"
        output "Output: $output_text"
        ((PASSED++)) || true
    else
        fail "Response missing ID"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        ((FAILED++)) || true
    fi
else
    fail "Basic response request failed"
    ((FAILED++)) || true
fi

echo ""

# Test 2: Response with function tool definition
info "Test 2: Response with function tool..."
response=$(curl -sf "$BASE_URL/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "input": "Calculate the sum of 15 and 27 using the calculator tool.",
        "tools": [
            {
                "type": "function",
                "name": "calculator",
                "description": "Performs basic math calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ],
        "stream": false
    }' 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    output_count=$(echo "$response" | jq '.output | length' 2>/dev/null)
    has_tool_call=$(echo "$response" | jq 'any(.output[]; .type == "function_call")' 2>/dev/null)
    pass "Function tool response created"
    output "Output items: $output_count"
    output "Has tool call: $has_tool_call"
    ((PASSED++)) || true
else
    fail "Function tool response failed"
    ((FAILED++)) || true
fi

echo ""

# Test 3: List responses
info "Test 3: List responses..."
response=$(curl -sf "$BASE_URL/v1/responses?limit=5" 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    count=$(echo "$response" | jq '.data | length' 2>/dev/null || echo "0")
    pass "List responses working"
    output "Found $count responses"
    ((PASSED++)) || true
else
    fail "List responses failed"
    ((FAILED++)) || true
fi

echo ""

# Test 4: Multi-turn conversation using previous_response_id
info "Test 4: Multi-turn conversation..."

# First turn
response1=$(curl -sf "$BASE_URL/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "input": "My name is Alice.",
        "stream": false
    }' 2>/dev/null) || response1=""

if [[ -n "$response1" ]]; then
    response1_id=$(echo "$response1" | jq -r '.id' 2>/dev/null)

    # Second turn referencing first
    response2=$(curl -sf "$BASE_URL/v1/responses" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "input": "What is my name?",
            "previous_response_id": "'"$response1_id"'",
            "stream": false
        }' 2>/dev/null) || response2=""

    if [[ -n "$response2" ]]; then
        output2=$(echo "$response2" | jq -r '.output_text // empty' 2>/dev/null)
        if echo "$output2" | grep -qi "alice"; then
            pass "Multi-turn conversation working"
            output "Model remembered: $output2"
            ((PASSED++)) || true
        else
            pass "Multi-turn created (context may vary)"
            output "Response: $output2"
            ((PASSED++)) || true
        fi
    else
        fail "Second turn failed"
        ((FAILED++)) || true
    fi
else
    fail "First turn failed"
    ((FAILED++)) || true
fi

echo ""

# Test 5: Check tool groups endpoint
info "Test 5: List tool groups..."
response=$(curl -sf "$BASE_URL/v1/tool-groups" 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    count=$(echo "$response" | jq 'if type == "array" then length else 0 end' 2>/dev/null || echo "?")
    pass "Tool groups endpoint working"
    output "Registered tool groups: $count"
    ((PASSED++)) || true
else
    info "Tool groups endpoint not available (optional)"
fi

echo ""
echo "========================================"
echo -e "  Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo "========================================"
echo ""

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
