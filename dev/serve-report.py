#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Simple proxy server for the API Playground.

Serves static files and proxies /api/* requests to Llama Stack.
This avoids CORS issues when testing from the browser.
"""

import http.server
import json
import urllib.error
import urllib.request
from pathlib import Path

LLAMA_STACK_URL = "http://localhost:8321"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PORT = 8080


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPORTS_DIR), **kwargs)

    def do_OPTIONS(self):  # noqa: N802 - method name required by base class
        """Handle CORS preflight."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):  # noqa: N802 - method name required by base class
        if self.path.startswith("/api/"):
            self._proxy_request("GET")
        else:
            super().do_GET()

    def do_POST(self):  # noqa: N802 - method name required by base class
        if self.path.startswith("/api/"):
            self._proxy_request("POST")
        else:
            self.send_error(405, "POST only allowed for /api/*")

    def do_PUT(self):  # noqa: N802 - method name required by base class
        if self.path.startswith("/api/"):
            self._proxy_request("PUT")
        else:
            self.send_error(405, "PUT only allowed for /api/*")

    def do_DELETE(self):  # noqa: N802 - method name required by base class
        if self.path.startswith("/api/"):
            self._proxy_request("DELETE")
        else:
            self.send_error(405, "DELETE only allowed for /api/*")

    def _proxy_request(self, method: str):
        # Convert /api/v1/models -> http://localhost:8321/v1/models
        target_path = self.path[4:]  # Remove /api prefix
        target_url = f"{LLAMA_STACK_URL}{target_path}"

        # Read request body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Get content type from request (preserve multipart/form-data for file uploads)
        content_type = self.headers.get("Content-Type", "application/json")

        # Check if streaming (only for JSON requests)
        is_streaming = False
        if body and "application/json" in content_type:
            try:
                data = json.loads(body)
                is_streaming = data.get("stream", False)
            except json.JSONDecodeError:
                pass

        try:
            # Build headers - preserve Content-Type for multipart uploads
            headers = {"Content-Type": content_type}

            req = urllib.request.Request(
                target_url,
                data=body,
                method=method,
                headers=headers,
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                if is_streaming:
                    # Stream the response
                    self.send_response(200)
                    self._send_cors_headers()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()

                    for chunk in iter(lambda: response.read(1024), b""):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    # Regular response
                    response_body = response.read()
                    self.send_response(response.status)
                    self._send_cors_headers()
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", len(response_body))
                    self.end_headers()
                    self.wfile.write(response_body)

        except urllib.error.HTTPError as e:
            error_body = e.read()
            self.send_response(e.code)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(error_body))
            self.end_headers()
            self.wfile.write(error_body)

        except urllib.error.URLError as e:
            error_msg = json.dumps({"error": f"Failed to connect to Llama Stack: {e.reason}"})
            self.send_response(503)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(error_msg))
            self.end_headers()
            self.wfile.write(error_msg.encode())

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def log_message(self, format, *args):
        # Color-coded logging
        method = args[0].split()[0] if args else ""
        path = args[0].split()[1] if args and len(args[0].split()) > 1 else ""
        status = args[1] if len(args) > 1 else ""

        if path.startswith("/api/"):
            print(f"\033[36m[PROXY]\033[0m {method} {path} -> {status}")
        else:
            print(f"\033[33m[STATIC]\033[0m {method} {path} -> {status}")


def main():
    print("\033[32m=== API Playground Server ===\033[0m")
    print(f"Reports dir: {REPORTS_DIR}")
    print(f"Proxying /api/* -> {LLAMA_STACK_URL}")
    print(f"\n\033[1mOpen: http://localhost:{PORT}/local-dev-guide.html\033[0m\n")

    with http.server.HTTPServer(("", PORT), ProxyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
