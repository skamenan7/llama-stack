# AWS Bedrock Provider - Examples and Testing Guide

This document provides comprehensive examples for using the AWS Bedrock provider with Llama Stack, following the QE onboarding guide format.

## Table of Contents

1. [Overview](#overview)
2. [Use Cases and Examples](#use-cases-and-examples)
3. [Python SDK Examples](#python-sdk-examples)
4. [cURL Examples](#curl-examples)
5. [Stack Configuration](#stack-configuration)
6. [Testing Guide](#testing-guide)
7. [Provider Testing Status](#provider-testing-status)
8. [Version Support Matrix](#version-support-matrix)
9. [API Feature Support Matrix](#api-feature-support-matrix)

---

## Overview

AWS Bedrock provides an OpenAI-compatible endpoint that allows you to use models like OpenAI GPT-OSS through a familiar API. The Llama Stack Bedrock provider (`remote::bedrock`) enables seamless integration with this endpoint.

### Key Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming (SSE) support
- System messages for role-based prompting
- Multi-turn conversation with context retention
- Configurable sampling parameters (temperature, top_p, max_tokens)

### Limitations

- No `/v1/embeddings` support (use native Bedrock embedding models)
- No `/v1/completions` support (only chat completions)
- No tool calling support (times out on Bedrock)
- No `/v1/models` endpoint (models must be configured statically)

---

## Use Cases and Examples

| Use Case | Description | Format | Status |
|----------|-------------|--------|--------|
| **Simple Inference** | Basic chat completion with Bedrock OpenAI-compatible endpoint | Jupyter Notebook / Python script | Available |
| **Streaming Inference** | SSE streaming chat completion | Python script | Available |
| **Multi-turn Conversation** | Context retention across messages | Python script | Available |
| **System Messages** | Role-based prompting | Python script | Available |
| **RAG Workflow** | N/A - Bedrock endpoint doesn't support `/v1/embeddings` | - | Not supported |
| **Agentic Workflow** | N/A - Tool calling times out on Bedrock OpenAI endpoint | - | Not supported |

---

## Python SDK Examples

### Setup

```python
import os
from llama_stack_client import LlamaStackClient

# Configuration
BASE_URL = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")
MODEL_ID = "bedrock/openai.gpt-oss-20b-1:0"

# Initialize client (no auth needed - server handles Bedrock auth)
client = LlamaStackClient(base_url=BASE_URL)
```

### Example 1: Simple Chat Completion (Non-Streaming)

```python
# Uses OpenAI-compatible API: client.chat.completions.create()
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Which planet do humans live on?"}],
    stream=False,
)

print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")
```

**Expected Output:**
```
Response: Humans live on **Earth**—the third planet from the Sun...
Tokens used: 214
```

### Example 2: Streaming Response

```python
stream = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "What's the name of the Sun in Latin?"}],
    stream=True,
)

print("Streaming response: ", end="")
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

**Expected Output:**
```
Streaming response: The Latin name for the Sun is **Sol**.
```

### Example 3: Multi-turn Conversation (Context Retention)

```python
conversation = [
    {"role": "user", "content": "My name is Alice"},
    {
        "role": "assistant",
        "content": "Nice to meet you, Alice! How can I help you today?",
    },
    {"role": "user", "content": "What is my name?"},
]

response = client.chat.completions.create(
    model=MODEL_ID, messages=conversation, stream=False
)

print(f"Response: {response.choices[0].message.content}")
# Expected: Response mentions "Alice" (context retained)
```

### Example 4: System Messages (Role-Based Prompting)

```python
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "system",
            "content": "You are Shakespeare. Respond only in Shakespearean English.",
        },
        {"role": "user", "content": "Tell me about the weather today"},
    ],
    stream=False,
)

print(f"Shakespeare: {response.choices[0].message.content}")
```

### Example 5: Sampling Parameters

```python
# Low temperature = deterministic
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    temperature=0.1,
    max_tokens=50,
    stream=False,
)

# High temperature = creative
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    temperature=0.9,
    top_p=0.95,
    max_tokens=50,
    stream=False,
)
```

### Example 6: Error Handling

```python
def safe_chat_completion(messages, model_id=MODEL_ID):
    """Wrapper with error handling for chat completions."""
    try:
        response = client.chat.completions.create(
            model=model_id, messages=messages, stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("Auth Error: Check AWS_BEARER_TOKEN_BEDROCK on server")
        elif "404" in error_msg:
            print(f"Model not found: {model_id}")
        else:
            print(f"Error: {e}")
        return None


# Usage
result = safe_chat_completion([{"role": "user", "content": "Hello!"}])
```

---

## cURL Examples

### Example 1: Non-Streaming Chat Completion

```bash
curl -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/openai.gpt-oss-20b-1:0",
    "messages": [{"role": "user", "content": "Which planet do humans live on?"}],
    "stream": false
  }' | jq
```

**Response:**
```json
{
  "id": "rec-0eca8ac96560",
  "choices": [{
    "finish_reason": "stop",
    "index": 0,
    "message": {
      "content": "Humans live on **Earth**—the third planet from the Sun...",
      "role": "assistant"
    }
  }],
  "model": "openai.gpt-oss-20b-1:0",
  "usage": {"completion_tokens": 138, "prompt_tokens": 76, "total_tokens": 214}
}
```

### Example 2: Streaming Chat Completion

```bash
curl -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/openai.gpt-oss-20b-1:0",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "stream": true
  }'
```

**Response (SSE chunks):**
```
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"role": "assistant"}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"content": "1, "}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"content": "2, "}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"content": "3, "}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"content": "4, "}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"delta": {"content": "5"}}]}
data: {"id": "rec-624b41ef0112", "choices": [{"finish_reason": "stop"}]}
data: [DONE]
```

### Example 3: Multi-turn Conversation

```bash
curl -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/openai.gpt-oss-20b-1:0",
    "messages": [
      {"role": "user", "content": "My name is Alice"},
      {"role": "assistant", "content": "Nice to meet you, Alice!"},
      {"role": "user", "content": "What is my name?"}
    ],
    "stream": false
  }' | jq -r '.choices[0].message.content'
```

### Example 4: System Messages

```bash
curl -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/openai.gpt-oss-20b-1:0",
    "messages": [
      {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }' | jq
```

### Example 5: Sampling Parameters

```bash
curl -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock/openai.gpt-oss-20b-1:0",
    "messages": [{"role": "user", "content": "Write a haiku about coding"}],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 50,
    "stream": false
  }' | jq
```

### Example 6: Direct AWS Bedrock API Call (Without Llama Stack)

```bash
export AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-<your-presigned-url>"

curl -X POST "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai.gpt-oss-20b-1:0",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }'
```

### Quick Smoke Test Script

```bash
#!/bin/bash
export BASE_URL="${BASE_URL:-http://localhost:8321}"

# Auto-discover Bedrock model
MODEL_ID=$(curl -s "${BASE_URL}/v1/models" | jq -r '.data[] | select(.identifier | contains("bedrock")) | .identifier' | head -1)
echo "Using model: $MODEL_ID"

# Test non-streaming
echo "Testing non-streaming..."
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "messages": [{"role": "user", "content": "Say hello in 3 words"}],
    "stream": false
  }' | jq -r '.choices[0].message.content'

echo "Test complete!"
```

---

## Stack Configuration

### Minimal Configuration (`bedrock-run.yaml`)

```yaml
version: 2
image_name: bedrock-test
apis:
  - inference
  - models
providers:
  inference:
    - provider_id: bedrock
      provider_type: remote::bedrock
      config:
        api_key: ${env.AWS_BEARER_TOKEN_BEDROCK:=replay-mode-dummy-key}
        region_name: us-west-2
storage:
  backends:
    kv_default:
      type: kv_sqlite
      db_path: ${env.SQLITE_STORE_DIR:=./.llama}/kvstore.db
    sql_default:
      type: sql_sqlite
      db_path: ${env.SQLITE_STORE_DIR:=./.llama}/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
registered_resources:
  models:
    - model_id: openai.gpt-oss-20b-1:0
      provider_id: bedrock
      provider_resource_id: openai.gpt-oss-20b-1:0
      model_type: llm
      metadata:
        description: "OpenAI GPT-OSS 20B on Bedrock (us-west-2)"
server:
  port: 8321
```

### Environment Variables

```bash
# Required for live API calls
export AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-<your-presigned-url>"
export AWS_DEFAULT_REGION=us-west-2

# Optional
export SQLITE_STORE_DIR=$(mktemp -d)
export LLAMA_STACK_BASE_URL="http://localhost:8321"
```

### Running the Server

```bash
# Start Llama Stack with Bedrock provider
llama stack run bedrock-run.yaml
```

---

## Testing Guide

### Running Integration Tests

**Record Mode (creates recordings with live API):**
```bash
export AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-..."
export AWS_DEFAULT_REGION=us-west-2
export SQLITE_STORE_DIR=$(mktemp -d)

cd tests/integration/inference
uv run pytest -v -s \
  test_openai_completion.py::test_openai_chat_completion_non_streaming \
  test_openai_completion.py::test_openai_chat_completion_streaming \
  --setup=bedrock \
  --stack-config=ci-tests::run.yaml \
  --inference-mode=record \
  -k "client_with_models"
```

**Replay Mode (no credentials needed):**
```bash
cd tests/integration/inference
uv run pytest -v -s \
  test_openai_completion.py::test_openai_chat_completion_non_streaming \
  test_openai_completion.py::test_openai_chat_completion_streaming \
  --setup=bedrock \
  --stack-config=ci-tests::run.yaml \
  --inference-mode=replay \
  -k "client_with_models"
```

### Test Results

| Mode | Tests | Duration | API Calls |
|------|-------|----------|-----------|
| **Record** | 4/4 PASSED | ~7.87s | Live AWS Bedrock |
| **Replay** | 4/4 PASSED | ~0.81s | Pre-recorded JSON |

### Recorded Test Cases

| Test | Description | Recording File |
|------|-------------|----------------|
| `non_streaming_01` | "Which planet do humans live on?" | `0eca8ac9...json` |
| `streaming_01` | "What's the name of the Sun in latin?" | `624b41ef...json` |
| `non_streaming_02` | "Which planet has rings starting with S?" | `b992b97c...json` |
| `streaming_02` | "What is the name of the US capital?" | `b82395f9...json` |

---

## Provider Testing Status

| Provider | Testing Status | API | References |
|----------|---------------|-----|------------|
| AWS Bedrock (OpenAI-compat) | In CI | Inference (chat completions) | PR #4095 |

### Automated Test Coverage

| Test Type | Location | Description | Status |
|-----------|----------|-------------|--------|
| **Unit Tests** | `tests/unit/providers/test_bedrock.py` | Bedrock provider logic | Available |
| **Integration Tests** | `tests/integration/inference/test_openai_completion.py` | OpenAI chat completion | 4 tests |
| **CI Replay Mode** | `tests/integration/inference/recordings/*.json` | Pre-recorded responses | 4 recordings |

---

## Version Support Matrix

| Component | Minimum Supported Version | Latest Validated Version | Notes |
|-----------|--------------------------|-------------------------|-------|
| **AWS Bedrock SDK** | boto3 1.34+ | 1.35.x | Uses OpenAI-compatible endpoint |
| **OpenAI GPT-OSS Model** | openai.gpt-oss-20b-1:0 | openai.gpt-oss-20b-1:0 | us-west-2 region only |
| **Llama Stack** | 0.2.x | 0.2.x | `remote::bedrock` provider |
| **Python** | 3.10 | 3.12 | Required for type hints |
| **OpenShift Container Platform** | 4.14 | 4.16 | For RHOAI deployment |

---

## API Feature Support Matrix

| Feature | Supported | Notes |
|---------|-----------|-------|
| `/v1/chat/completions` | Yes | Full support |
| Streaming (SSE) | Yes | Full support |
| System messages | Yes | Full support |
| Multi-turn context | Yes | Full support |
| Sampling params | Yes | temperature, top_p, max_tokens |
| `/v1/completions` | No | Not supported by Bedrock |
| `/v1/models` | No | Static config required |
| `/v1/embeddings` | No | Not supported by Bedrock |
| Tool calling | No | Times out on Bedrock |

---

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Bedrock OpenAI Compatibility](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html)
- [Llama Stack Repository](https://github.com/meta-llama/llama-stack)
- [Llama Stack Client SDK](https://pypi.org/project/llama-stack-client/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Jupyter Notebook

A companion Jupyter notebook (`bedrock_inference_example.ipynb`) is available in this directory with interactive examples.
