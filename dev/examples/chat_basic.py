#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Basic chat completion example using Llama Stack.
"""

import httpx

BASE_URL = "http://localhost:8321"
MODEL = "ollama/llama3.2:3b-instruct-fp16"


def chat_completion(message: str) -> str:
    """Simple non-streaming chat completion."""
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 200,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def chat_streaming(message: str):
    """Streaming chat completion - yields tokens as they arrive."""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 200,
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                import json

                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content


if __name__ == "__main__":
    print("=== Non-streaming ===")
    result = chat_completion("What is the capital of France?")
    print(f"Response: {result}\n")

    print("=== Streaming ===")
    print("Response: ", end="", flush=True)
    for token in chat_streaming("Count from 1 to 5."):
        print(token, end="", flush=True)
    print("\n")
