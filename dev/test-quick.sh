#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# Quick health check for Llama Stack services
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

PORT="${LLAMA_STACK_PORT:-8321}"
BASE_URL="http://localhost:$PORT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[TEST]${NC} $*"; }

echo ""
echo "========================================"
echo "  Llama Stack Quick Health Check"
echo "========================================"
echo ""

PASSED=0
FAILED=0

# Test 1: Health endpoint
info "Checking /v1/health endpoint..."
if curl -sf "$BASE_URL/v1/health" >/dev/null 2>&1; then
    pass "Health endpoint responding"
    ((PASSED++)) || true
else
    fail "Health endpoint not responding"
    ((FAILED++)) || true
fi

# Test 2: Providers endpoint
info "Checking /v1/providers endpoint..."
if response=$(curl -sf "$BASE_URL/v1/providers" 2>/dev/null); then
    provider_count=$(echo "$response" | jq 'length' 2>/dev/null || echo "?")
    pass "Providers endpoint responding ($provider_count providers)"
    ((PASSED++)) || true
else
    fail "Providers endpoint not responding"
    ((FAILED++)) || true
fi

# Test 3: List models
info "Checking /v1/models endpoint..."
if response=$(curl -sf "$BASE_URL/v1/models" 2>/dev/null); then
    model_count=$(echo "$response" | jq '.data | length' 2>/dev/null || echo "?")
    pass "Models endpoint responding ($model_count models registered)"
    ((PASSED++)) || true
else
    fail "Models endpoint not responding"
    ((FAILED++)) || true
fi

# Test 4: Ollama connectivity
info "Checking Ollama connectivity..."
if curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    model_count=$(curl -s "http://localhost:11434/api/tags" | jq '.models | length' 2>/dev/null || echo "?")
    pass "Ollama responding ($model_count models available)"
    ((PASSED++)) || true
else
    fail "Ollama not responding"
    ((FAILED++)) || true
fi

# Test 5: PostgreSQL connectivity
info "Checking PostgreSQL connectivity..."
if docker exec llama-stack-postgres pg_isready -U llamastack >/dev/null 2>&1; then
    pass "PostgreSQL responding"
    ((PASSED++)) || true
else
    fail "PostgreSQL not responding"
    ((FAILED++)) || true
fi

echo ""
echo "========================================"
echo -e "  Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo "========================================"
echo ""

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
