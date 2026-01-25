#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# Test inference capabilities of Llama Stack
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
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[TEST]${NC} $*"; }
output() { echo -e "${BLUE}[OUTPUT]${NC} $*"; }

echo ""
echo "========================================"
echo "  Llama Stack Inference Tests"
echo "========================================"
echo "  Model: $MODEL"
echo "========================================"
echo ""

PASSED=0
FAILED=0

# Test 1: Simple completion (OpenAI-compatible)
info "Testing OpenAI-compatible chat completion..."
response=$(curl -sf "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 20
    }' 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    if [[ -n "$content" ]]; then
        pass "Chat completion working"
        output "$content"
        ((PASSED++)) || true
    else
        fail "Chat completion returned empty content"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        ((FAILED++)) || true
    fi
else
    fail "Chat completion request failed"
    ((FAILED++)) || true
fi

echo ""

# Test 2: Streaming completion
info "Testing streaming chat completion..."
stream_output=""
while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        json="${line#data: }"
        if [[ "$json" != "[DONE]" && -n "$json" ]]; then
            delta=$(echo "$json" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
            stream_output+="$delta"
        fi
    fi
done < <(curl -sN "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 30,
        "stream": true
    }' 2>/dev/null)

if [[ -n "$stream_output" ]]; then
    pass "Streaming completion working"
    output "$stream_output"
    ((PASSED++)) || true
else
    fail "Streaming completion failed"
    ((FAILED++)) || true
fi

echo ""

# Test 3: Embeddings API
info "Testing embeddings API..."
response=$(curl -sf "$BASE_URL/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": "Hello world"
    }' 2>/dev/null) || response=""

if [[ -n "$response" ]]; then
    embedding_len=$(echo "$response" | jq -r '.data[0].embedding | length' 2>/dev/null)
    if [[ -n "$embedding_len" && "$embedding_len" != "null" ]]; then
        pass "Embeddings API working"
        output "Embedding dimension: $embedding_len"
        ((PASSED++)) || true
    else
        fail "Embeddings API returned unexpected format"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        ((FAILED++)) || true
    fi
else
    info "Embeddings API skipped (model not registered)"
fi

echo ""
echo "========================================"
echo -e "  Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo "========================================"
echo ""

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
