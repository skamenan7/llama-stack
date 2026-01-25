#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Agents and Tool Calling example using Llama Stack Responses API.
"""

import json

import httpx

BASE_URL = "http://localhost:8321"
MODEL = "ollama/llama3.2:3b-instruct-fp16"


def create_response_with_tools(user_input: str, tools: list | None = None) -> dict:
    """Create a response using the Responses API with optional tools."""
    payload = {
        "model": MODEL,
        "input": user_input,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    response = httpx.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def multi_turn_conversation():
    """Demonstrate multi-turn conversation with previous_response_id."""
    print("=== Multi-turn Conversation ===\n")

    # First turn
    response1 = create_response_with_tools("My favorite color is blue.")
    print(f"Turn 1 Response ID: {response1['id']}")
    print(f"Turn 1 Output: {response1.get('output_text', 'N/A')}\n")

    # Second turn - references first
    payload = {
        "model": MODEL,
        "input": "What is my favorite color?",
        "previous_response_id": response1["id"],
        "stream": False,
    }
    response2 = httpx.post(f"{BASE_URL}/v1/responses", json=payload, timeout=60.0)
    response2.raise_for_status()
    data2 = response2.json()

    print(f"Turn 2 Response ID: {data2['id']}")
    print(f"Turn 2 Output: {data2.get('output_text', 'N/A')}\n")


def function_tool_example():
    """Demonstrate function tool calling."""
    print("=== Function Tool Calling ===\n")

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    response = create_response_with_tools("What's the weather like in Tokyo?", tools=tools)

    print(f"Response ID: {response['id']}")
    print(f"Output items: {len(response.get('output', []))}")

    for i, output in enumerate(response.get("output", [])):
        print(f"\nOutput {i + 1}:")
        print(f"  Type: {output.get('type')}")
        if output.get("type") == "function_call":
            print(f"  Function: {output.get('name')}")
            print(f"  Arguments: {output.get('arguments')}")
        elif output.get("type") == "message":
            content = output.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        print(f"  Content: {c.get('text', c)}")
                    else:
                        print(f"  Content: {c}")
            else:
                print(f"  Content: {content}")


def streaming_response():
    """Stream a response and show events."""
    print("=== Streaming Response ===\n")

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/responses",
        json={
            "model": MODEL,
            "input": "Write a haiku about programming.",
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        print("Events received:")
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data and data != "[DONE]":
                    event = json.loads(data)
                    event_type = event.get("type", "unknown")

                    if event_type == "response.output_text.delta":
                        # Print text as it arrives
                        print(event.get("delta", ""), end="", flush=True)
                    elif event_type in ["response.created", "response.completed"]:
                        print(f"\n[{event_type}]")

    print()


if __name__ == "__main__":
    print("Llama Stack Agents & Tools Examples")
    print("=" * 40)
    print()

    try:
        multi_turn_conversation()
    except httpx.HTTPStatusError as e:
        print(f"Multi-turn failed: {e}")
        print("(Responses API may need specific config)")

    print()

    try:
        function_tool_example()
    except httpx.HTTPStatusError as e:
        print(f"Function tools failed: {e}")

    print()

    try:
        streaming_response()
    except httpx.HTTPStatusError as e:
        print(f"Streaming failed: {e}")
