#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
MCP (Model Context Protocol) Tool Calling example.

This demonstrates how to use MCP tools with Llama Stack.
You'll need an MCP server running to use these examples.

Quick MCP server setup (optional):
  npx @anthropics/create-mcp-server
  # or use a pre-built one like filesystem, git, etc.
"""

import json

import httpx

BASE_URL = "http://localhost:8321"
MODEL = "ollama/llama3.2:3b-instruct-fp16"

# Example MCP server URL (you'd run your own)
MCP_SERVER_URL = "http://localhost:3000/sse"


def list_tool_groups():
    """List registered tool groups."""
    print("=== Registered Tool Groups ===\n")

    response = httpx.get(f"{BASE_URL}/v1/tool-groups", timeout=10.0)
    if response.status_code == 200:
        groups = response.json()
        if isinstance(groups, list):
            for group in groups:
                print(f"  - {group.get('toolgroup_id', group)}")
        else:
            print(f"  Response: {groups}")
    else:
        print(f"  Not available (status: {response.status_code})")


def register_mcp_toolgroup(toolgroup_id: str, mcp_url: str):
    """Register an MCP server as a tool group."""
    print(f"=== Registering MCP Tool Group: {toolgroup_id} ===\n")

    payload = {
        "toolgroup_id": toolgroup_id,
        "provider_id": "model-context-protocol",
        "mcp_endpoint": {"uri": mcp_url},
    }

    response = httpx.post(
        f"{BASE_URL}/v1/tool-groups",
        json=payload,
        timeout=30.0,
    )

    if response.status_code in [200, 201]:
        print(f"  Registered: {toolgroup_id}")
        return True
    else:
        print(f"  Failed: {response.status_code}")
        print(f"  {response.text}")
        return False


def use_mcp_tools(user_input: str, mcp_url: str, server_label: str = "mcp-server"):
    """Use MCP tools in a response."""
    print("=== Using MCP Tools ===\n")
    print(f"Input: {user_input}")
    print(f"MCP Server: {mcp_url}\n")

    payload = {
        "model": MODEL,
        "input": user_input,
        "tools": [
            {
                "type": "mcp",
                "server_label": server_label,
                "server_url": mcp_url,
                "require_approval": "never",
            }
        ],
        "stream": False,
    }

    response = httpx.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        timeout=120.0,
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Response ID: {data.get('id')}")
        print(f"Output: {data.get('output_text', 'N/A')}")

        # Show any MCP calls made
        for output in data.get("output", []):
            if output.get("type") == "mcp_call":
                print("\nMCP Call:")
                print(f"  Server: {output.get('server_label')}")
                print(f"  Tool: {output.get('name')}")
                print(f"  Output: {output.get('output')}")
                if output.get("error"):
                    print(f"  Error: {output.get('error')}")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)


def mcp_with_auth(user_input: str, mcp_url: str, oauth_token: str):
    """Use MCP tools with OAuth authentication."""
    print("=== MCP with Authentication ===\n")

    payload = {
        "model": MODEL,
        "input": user_input,
        "tools": [
            {
                "type": "mcp",
                "server_label": "authenticated-mcp",
                "server_url": mcp_url,
                # Pass token directly (not as "Bearer token")
                "authorization": oauth_token,
                "require_approval": "never",
            }
        ],
        "stream": False,
    }

    response = httpx.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        timeout=120.0,
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Output: {data.get('output_text', 'N/A')}")


def stream_mcp_response(user_input: str, mcp_url: str):
    """Stream a response with MCP tools to see events."""
    print("=== Streaming MCP Response ===\n")

    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/responses",
        json={
            "model": MODEL,
            "input": user_input,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "mcp",
                    "server_url": mcp_url,
                }
            ],
            "stream": True,
        },
        timeout=120.0,
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data and data != "[DONE]":
                    event = json.loads(data)
                    event_type = event.get("type", "")

                    # Show MCP-related events
                    if "mcp" in event_type.lower():
                        print(f"[{event_type}]")
                        if event_type == "response.mcp_call.completed":
                            print(f"  Tool: {event.get('name')}")

                    # Show text output
                    if event_type == "response.output_text.delta":
                        print(event.get("delta", ""), end="", flush=True)

    print()


if __name__ == "__main__":
    print("Llama Stack MCP Tools Examples")
    print("=" * 40)
    print()
    print("NOTE: These examples require an MCP server running.")
    print(f"      Expected at: {MCP_SERVER_URL}")
    print()

    # List existing tool groups
    try:
        list_tool_groups()
    except Exception as e:
        print(f"Could not list tool groups: {e}")

    print()
    print("To run MCP examples, start an MCP server first:")
    print("  npx -y @anthropics/mcp-server-memory")
    print("  # or")
    print("  npx -y @anthropics/mcp-server-filesystem /tmp")
    print()

    # Uncomment to test with a real MCP server:
    # register_mcp_toolgroup("my-mcp", MCP_SERVER_URL)
    # use_mcp_tools("List available tools", MCP_SERVER_URL)
