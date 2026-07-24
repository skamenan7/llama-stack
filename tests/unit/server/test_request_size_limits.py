# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from ogx.core.datatypes import ServerConfig
from ogx.core.server.server import RequestBodyTooLargeError, RequestSizeLimitMiddleware


def _scope(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {"type": "http", "path": path, "headers": headers or [], "client": ("127.0.0.1", 1234)}


def _receive_messages(messages: list[Message]) -> Receive:
    async def receive() -> Message:
        return messages.pop(0)

    return receive


def _send_messages(sent: list[Message]) -> Send:
    async def send(message: Message) -> None:
        sent.append(message)

    return send


async def _consume_request_body(scope: Scope, receive: Receive, send: Send) -> None:
    while True:
        message = await receive()
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


class TestRequestSizeLimitMiddleware:
    async def test_allows_body_at_exact_limit(self) -> None:
        sent: list[Message] = []
        middleware = RequestSizeLimitMiddleware(
            _consume_request_body, max_request_body_size=10, max_file_upload_size=20
        )

        await middleware(
            _scope("/v1/responses", [(b"content-length", b"10")]),
            _receive_messages([{"type": "http.request", "body": b"x" * 10, "more_body": False}]),
            _send_messages(sent),
        )

        assert sent[0]["status"] == 200

    async def test_rejects_declared_oversized_body_before_calling_app(self) -> None:
        called = False

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal called
            called = True

        middleware = RequestSizeLimitMiddleware(app, max_request_body_size=10, max_file_upload_size=20)
        sent: list[Message] = []

        await middleware(
            _scope("/v1/responses", [(b"content-length", b"11")]),
            _receive_messages([]),
            _send_messages(sent),
        )

        assert called is False
        assert sent[0]["status"] == 413

    async def test_rejects_chunked_body_without_content_length(self) -> None:
        middleware = RequestSizeLimitMiddleware(
            _consume_request_body, max_request_body_size=10, max_file_upload_size=20
        )

        try:
            await middleware(
                _scope("/v1/responses"),
                _receive_messages(
                    [
                        {"type": "http.request", "body": b"12345", "more_body": True},
                        {"type": "http.request", "body": b"678901", "more_body": False},
                    ]
                ),
                _send_messages([]),
            )
        except RequestBodyTooLargeError:
            pass
        else:
            raise AssertionError("Expected the chunked request to exceed the configured limit")

    async def test_counts_body_when_content_length_is_forged(self) -> None:
        middleware = RequestSizeLimitMiddleware(
            _consume_request_body, max_request_body_size=10, max_file_upload_size=20
        )

        try:
            await middleware(
                _scope("/v1/responses", [(b"content-length", b"1")]),
                _receive_messages([{"type": "http.request", "body": b"12345678901", "more_body": False}]),
                _send_messages([]),
            )
        except RequestBodyTooLargeError:
            pass
        else:
            raise AssertionError("Expected the forged Content-Length request to exceed the configured limit")

    async def test_uses_upload_limit_for_upload_paths(self) -> None:
        sent: list[Message] = []
        middleware = RequestSizeLimitMiddleware(
            _consume_request_body, max_request_body_size=10, max_file_upload_size=20
        )

        await middleware(
            _scope("/v1/files", [(b"content-length", b"20")]),
            _receive_messages([{"type": "http.request", "body": b"x" * 20, "more_body": False}]),
            _send_messages(sent),
        )

        assert sent[0]["status"] == 200


def test_server_resource_limit_config_is_parsed() -> None:
    config = ServerConfig(
        insecure=True,
        max_request_body_size=4096,
        max_file_upload_size=8192,
        limit_concurrency=8,
        limit_max_requests=1000,
        timeout_keep_alive=9,
    )

    assert config.max_request_body_size == 4096
    assert config.max_file_upload_size == 8192
    assert config.limit_concurrency == 8
    assert config.limit_max_requests == 1000
    assert config.timeout_keep_alive == 9


def test_request_size_limit_returns_413_from_fastapi() -> None:
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.exception_handler(RequestBodyTooLargeError)
    async def request_body_too_large_handler(request: Request, exc: RequestBodyTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": "Request body exceeds the allowed size"})

    app.add_middleware(RequestSizeLimitMiddleware, max_request_body_size=10, max_file_upload_size=20)
    response = TestClient(app).post("/v1/responses", content=b"x" * 11)

    assert response.status_code == 413
