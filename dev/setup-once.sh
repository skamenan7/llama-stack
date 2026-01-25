#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# One-time setup for Llama Stack local development
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo ""
echo "========================================"
echo "  Llama Stack Local Dev - First Time Setup"
echo "========================================"
echo ""

cd "$WORKTREE_ROOT"

# 1. Create Python virtual environment
log_info "Setting up Python virtual environment..."
if [[ ! -d ".venv" ]]; then
    uv venv --python 3.12
    log_ok "Virtual environment created"
else
    log_ok "Virtual environment already exists"
fi

# 2. Install dependencies
log_info "Installing dependencies..."
uv sync --group dev
log_ok "Dependencies installed"

# 3. Install pre-commit hooks
log_info "Setting up pre-commit hooks..."
uv run pre-commit install
log_ok "Pre-commit hooks installed"

# 4. Pull Ollama models
log_info "Pulling required Ollama models..."
echo "  This may take a while on first run..."

models=(
    "llama3.2:3b-instruct-fp16"
    "llama-guard3:1b"
)

for model in "${models[@]}"; do
    log_info "Pulling $model..."
    ollama pull "$model"
done
log_ok "Models pulled"

# 5. Start PostgreSQL
log_info "Starting PostgreSQL container..."
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q llama-stack-postgres; then
    log_ok "PostgreSQL already running"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q llama-stack-postgres; then
    docker start llama-stack-postgres
    log_ok "PostgreSQL started"
else
    docker run -d \
        --name llama-stack-postgres \
        -e POSTGRES_USER=llamastack \
        -e POSTGRES_PASSWORD=llamastack \
        -e POSTGRES_DB=llamastack \
        -p 5432:5432 \
        --restart unless-stopped \
        postgres:17
    log_ok "PostgreSQL started"
fi

# 6. Generate initial report
log_info "Generating initial status report..."
mkdir -p "$WORKTREE_ROOT/reports"
"$SCRIPT_DIR/generate-report.sh" > "$WORKTREE_ROOT/reports/initial-setup.html"
log_ok "Report generated at reports/initial-setup.html"

echo ""
echo "========================================"
echo -e "  ${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start all services:"
echo "     ./dev/llama-dev start"
echo ""
echo "  2. Check status:"
echo "     ./dev/llama-dev status"
echo ""
echo "  3. Run quick tests:"
echo "     ./dev/llama-dev test quick"
echo ""
echo "  4. View the setup report:"
echo "     xdg-open reports/initial-setup.html"
echo ""
