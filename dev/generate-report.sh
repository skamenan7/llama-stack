#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# Generate HTML status report for Llama Stack local dev environment
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/.env" 2>/dev/null || true

PORT="${LLAMA_STACK_PORT:-8321}"
BASE_URL="http://localhost:$PORT"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Collect service status
ollama_status="stopped"
ollama_pid=""
if pgrep -x ollama >/dev/null 2>&1; then
    ollama_status="running"
    ollama_pid=$(pgrep -x ollama)
fi

postgres_status="stopped"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q llama-stack-postgres; then
    postgres_status="running"
fi

stack_status="stopped"
stack_version=""
if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    stack_status="running"
    stack_version=$(curl -sf "$BASE_URL/version" 2>/dev/null | jq -r '.version // "unknown"' 2>/dev/null || echo "unknown")
fi

# Get model list
models_json="[]"
if [[ "$stack_status" == "running" ]]; then
    models_json=$(curl -sf "$BASE_URL/v1/models" 2>/dev/null | jq '.data // []' 2>/dev/null || echo "[]")
fi

ollama_models_json="[]"
if [[ "$ollama_status" == "running" ]]; then
    ollama_models_json=$(curl -sf "http://localhost:11434/api/tags" 2>/dev/null | jq '.models // []' 2>/dev/null || echo "[]")
fi

# Generate HTML
cat << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Llama Stack Status Report</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --border: #30363d;
            --accent-orange: #f0883e;
            --accent-purple: #a371f7;
            --accent-pink: #db61a2;
            --accent-red: #f85149;
            --success: #3fb950;
            --warning: #d29922;
            --error: #f85149;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        header h1 {
            color: var(--accent-orange);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        header .timestamp {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
        }

        .card h2 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card h2 .icon { font-size: 1.3rem; }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .status-badge.running {
            background-color: rgba(63, 185, 80, 0.15);
            color: var(--success);
        }

        .status-badge.stopped {
            background-color: rgba(248, 81, 73, 0.15);
            color: var(--error);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: currentColor;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
        }

        .info-row:last-child { border-bottom: none; }

        .info-label {
            color: var(--text-secondary);
        }

        .info-value {
            font-family: 'SF Mono', Consolas, monospace;
            color: var(--accent-purple);
        }

        .model-list {
            list-style: none;
            margin-top: 0.5rem;
        }

        .model-list li {
            padding: 0.5rem;
            background-color: var(--bg-tertiary);
            margin-bottom: 0.5rem;
            border-radius: 4px;
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.85rem;
        }

        .section-title {
            color: var(--accent-pink);
            font-size: 1.3rem;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .command-block {
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre;
            word-wrap: normal;
            word-break: normal;
            text-shadow: none;
            box-shadow: none;
        }

        .command-block code {
            background-color: transparent;
            text-shadow: none;
        }

        .command-block .comment {
            color: var(--text-secondary);
        }

        .command-block .command {
            color: var(--success);
        }

        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .action-btn {
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }

        .action-btn:hover {
            background-color: var(--accent-purple);
            color: white;
        }

        .env-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .env-table td {
            padding: 0.5rem;
            border-bottom: 1px solid var(--border);
            font-family: 'SF Mono', Consolas, monospace;
            font-size: 0.8rem;
        }

        .env-table td:first-child {
            color: var(--accent-orange);
            white-space: nowrap;
        }

        .env-table td:last-child {
            color: var(--text-secondary);
            word-break: break-all;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Llama Stack Local Development</h1>
            <div class="timestamp">Generated: $TIMESTAMP</div>
        </header>

        <div class="grid">
            <!-- Service Status Cards -->
            <div class="card">
                <h2><span class="icon">🦙</span> Ollama</h2>
                <span class="status-badge $ollama_status">
                    <span class="status-dot"></span>
                    $ollama_status
                </span>
                <div style="margin-top: 1rem;">
                    <div class="info-row">
                        <span class="info-label">PID</span>
                        <span class="info-value">${ollama_pid:-N/A}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">URL</span>
                        <span class="info-value">http://localhost:11434</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🐘</span> PostgreSQL</h2>
                <span class="status-badge $postgres_status">
                    <span class="status-dot"></span>
                    $postgres_status
                </span>
                <div style="margin-top: 1rem;">
                    <div class="info-row">
                        <span class="info-label">Container</span>
                        <span class="info-value">llama-stack-postgres</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Port</span>
                        <span class="info-value">5432</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Database</span>
                        <span class="info-value">llamastack</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">📚</span> Llama Stack</h2>
                <span class="status-badge $stack_status">
                    <span class="status-dot"></span>
                    $stack_status
                </span>
                <div style="margin-top: 1rem;">
                    <div class="info-row">
                        <span class="info-label">Version</span>
                        <span class="info-value">${stack_version:-N/A}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Port</span>
                        <span class="info-value">$PORT</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Health</span>
                        <span class="info-value">$BASE_URL/health</span>
                    </div>
                </div>
            </div>
        </div>

        <h2 class="section-title">Registered Models</h2>
        <div class="grid">
            <div class="card">
                <h2><span class="icon">🤖</span> Llama Stack Models</h2>
                <ul class="model-list">
EOF

# Add Llama Stack models
echo "$models_json" | jq -r '.[] | .id // .model_id // "unknown"' 2>/dev/null | while read -r model; do
    echo "                    <li>$model</li>"
done

cat << EOF
                </ul>
            </div>
            <div class="card">
                <h2><span class="icon">🦙</span> Ollama Models</h2>
                <ul class="model-list">
EOF

# Add Ollama models
echo "$ollama_models_json" | jq -r '.[].name // empty' 2>/dev/null | while read -r model; do
    echo "                    <li>$model</li>"
done

cat << EOF
                </ul>
            </div>
        </div>

        <h2 class="section-title">Quick Commands</h2>
        <div class="command-block"><span class="comment"># Start all services</span>
<span class="command">./dev/llama-dev start</span>

<span class="comment"># Stop all services</span>
<span class="command">./dev/llama-dev stop</span>

<span class="comment"># Restart all services</span>
<span class="command">./dev/llama-dev restart</span>

<span class="comment"># Check status</span>
<span class="command">./dev/llama-dev status</span>

<span class="comment"># View logs</span>
<span class="command">./dev/llama-dev logs stack</span>

<span class="comment"># Run quick tests</span>
<span class="command">./dev/llama-dev test quick</span>

<span class="comment"># Run inference tests</span>
<span class="command">./dev/llama-dev test inference</span></div>

        <h2 class="section-title">Environment Configuration</h2>
        <div class="card">
            <table class="env-table">
                <tr><td>LLAMA_STACK_PORT</td><td>${LLAMA_STACK_PORT:-8321}</td></tr>
                <tr><td>LLAMA_STACK_CONFIG</td><td>${LLAMA_STACK_CONFIG:-ollama}</td></tr>
                <tr><td>OLLAMA_URL</td><td>${OLLAMA_URL:-http://localhost:11434/v1}</td></tr>
                <tr><td>POSTGRES_HOST</td><td>${POSTGRES_HOST:-127.0.0.1}</td></tr>
                <tr><td>POSTGRES_PORT</td><td>${POSTGRES_PORT:-5432}</td></tr>
                <tr><td>POSTGRES_DB</td><td>${POSTGRES_DB:-llamastack}</td></tr>
                <tr><td>TEXT_MODEL</td><td>${TEXT_MODEL:-ollama/llama3.2:3b-instruct-fp16}</td></tr>
                <tr><td>EMBEDDING_MODEL</td><td>${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}</td></tr>
            </table>
        </div>

        <h2 class="section-title">API Endpoints</h2>
        <div class="command-block"><span class="comment"># Health check</span>
<span class="command">curl $BASE_URL/health</span>

<span class="comment"># List models</span>
<span class="command">curl $BASE_URL/v1/models | jq</span>

<span class="comment"># Chat completion (OpenAI-compatible)</span>
<span class="command">curl $BASE_URL/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${TEXT_MODEL:-ollama/llama3.2:3b-instruct-fp16}",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'</span></div>

        <footer>
            Llama Stack Local Development Environment<br>
            Worktree: $WORKTREE_ROOT
        </footer>
    </div>
</body>
</html>
EOF
