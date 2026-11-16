# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwcrypto.common import base64url_encode
from jwcrypto.jwk import JWK
from jwcrypto.jwt import JWT

from ogx.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    CustomAuthConfig,
    OAuth2IntrospectionConfig,
    OAuth2JWKSConfig,
    OAuth2TokenAuthConfig,
)
from ogx.core.server.auth import AuthenticationMiddleware

# 32-byte secret so the oct keys satisfy HS256's 256-bit key length requirement.
_HS256_SECRET = b"0123456789abcdef0123456789abcdef"


@pytest.fixture
def suppress_auth_errors(caplog):
    """Suppress expected ERROR/WARNING logs for tests that deliberately trigger authentication errors"""
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth")
    caplog.set_level(logging.CRITICAL, logger="ogx.core.server.auth_providers")


@pytest.fixture
def jwt_token_valid():
    key = JWK(kty="oct", kid="1234567890", k=base64url_encode(_HS256_SECRET))
    token = JWT(
        header={"alg": "HS256", "kid": "1234567890"},
        claims={
            "sub": "my-user",
            "groups": ["group1", "group2"],
            "scope": "foo bar",
            "aud": "ogx",
        },
    )
    token.make_signed_token(key)
    return token.serialize()


# --- Custom auth provider: 503 tests ---


@pytest.fixture
def custom_auth_client():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint="http://mock-auth-service/validate",
        ),
        access_policy=[],
    )
    app.add_middleware(
        AuthenticationMiddleware,
        auth_config=auth_config,
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "ok"}

    return TestClient(app)


def test_custom_auth_connection_error_returns_503(custom_auth_client, suppress_auth_errors):
    async def mock_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    with patch("httpx.AsyncClient.post", new=mock_connect_error):
        response = custom_auth_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_custom_auth_timeout_returns_503(custom_auth_client, suppress_auth_errors):
    async def mock_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("Read timed out")

    with patch("httpx.AsyncClient.post", new=mock_timeout):
        response = custom_auth_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service" in response.json()["error"]["message"]


# --- OAuth2 JWKS provider: 503 tests ---


@pytest.fixture
def oauth2_client():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            jwks=OAuth2JWKSConfig(uri="http://mock-authz-service/.well-known/jwks.json"),
            audience="ogx",
        ),
        access_policy=[],
    )
    app.add_middleware(
        AuthenticationMiddleware,
        auth_config=auth_config,
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "ok"}

    return TestClient(app)


def test_jwks_connection_error_returns_503(oauth2_client, jwt_token_valid, suppress_auth_errors):
    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_jwks_network_error_returns_503(oauth2_client, jwt_token_valid, suppress_auth_errors):
    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        side_effect=OSError("Network is unreachable"),
    ):
        response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_jwks_timeout_returns_503(oauth2_client, jwt_token_valid, suppress_auth_errors):
    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        side_effect=httpx.ReadTimeout("Connection timed out"),
    ):
        response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_jwks_http_500_returns_503(oauth2_client, jwt_token_valid, suppress_auth_errors):
    request = httpx.Request("GET", "http://mock-authz-service/.well-known/jwks.json")
    response_500 = httpx.Response(500, request=request)
    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        side_effect=httpx.HTTPStatusError("Internal Server Error", request=request, response=response_500),
    ):
        response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {jwt_token_valid}"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_bad_token_still_returns_401(oauth2_client, suppress_auth_errors):
    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        return_value={
            "keys": [
                {
                    "kid": "1234567890",
                    "kty": "oct",
                    "alg": "HS256",
                    "use": "sig",
                    "k": base64url_encode(_HS256_SECRET),
                }
            ]
        },
    ):
        response = oauth2_client.get("/test", headers={"Authorization": "Bearer not.a.valid.jwt"})
        assert response.status_code == 401
        assert "Invalid JWT token" in response.json()["error"]["message"]


def test_jwks_missing_kid_still_returns_401(oauth2_client, suppress_auth_errors):
    key = JWK(kty="oct", kid="1234567890", k=base64url_encode(_HS256_SECRET))
    token_with_unknown_kid = JWT(
        header={"alg": "HS256", "kid": "unknown-kid"},
        claims={"sub": "my-user", "groups": ["group1", "group2"], "scope": "foo bar", "aud": "ogx"},
    )
    token_with_unknown_kid.make_signed_token(key)

    with patch(
        "ogx.core.server.auth_providers._JwksKeySource.fetch_jwks",
        return_value={
            "keys": [
                {
                    "kid": "1234567890",
                    "kty": "oct",
                    "alg": "HS256",
                    "use": "sig",
                    "k": base64url_encode(_HS256_SECRET),
                }
            ]
        },
    ):
        response = oauth2_client.get("/test", headers={"Authorization": f"Bearer {token_with_unknown_kid.serialize()}"})
        assert response.status_code == 401
        assert "Invalid JWT token" in response.json()["error"]["message"]


# --- OAuth2 introspection provider: 503 tests ---


@pytest.fixture
def introspection_client():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config=OAuth2TokenAuthConfig(
            type=AuthProviderType.OAUTH2_TOKEN,
            introspection=OAuth2IntrospectionConfig(
                url="http://mock-authz-service/token/introspect",
                client_id="myclient",
                client_secret="abcdefg",
            ),
        ),
        access_policy=[],
    )
    app.add_middleware(
        AuthenticationMiddleware,
        auth_config=auth_config,
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "ok"}

    return TestClient(app)


def test_introspection_connection_error_returns_503(introspection_client, suppress_auth_errors):
    async def mock_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    with patch("httpx.AsyncClient.post", new=mock_connect_error):
        response = introspection_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_introspection_timeout_returns_503(introspection_client, suppress_auth_errors):
    async def mock_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("Read timed out")

    with patch("httpx.AsyncClient.post", new=mock_timeout):
        response = introspection_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service" in response.json()["error"]["message"]


# --- Kubernetes auth provider: 503 tests ---


@pytest.fixture
def kubernetes_auth_client():
    app = FastAPI()
    auth_config = AuthenticationConfig(
        provider_config={
            "type": "kubernetes",
            "api_server_url": "https://api.cluster.example.com:6443",
            "verify_tls": False,
            "claims_mapping": {"username": "roles", "groups": "roles"},
        },
    )
    app.add_middleware(
        AuthenticationMiddleware,
        auth_config=auth_config,
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "ok"}

    return TestClient(app)


def test_kubernetes_auth_connection_error_returns_503(kubernetes_auth_client, suppress_auth_errors):
    async def mock_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    with patch("httpx.AsyncClient.post", new=mock_connect_error):
        response = kubernetes_auth_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service unavailable" in response.json()["error"]["message"]


def test_kubernetes_auth_timeout_returns_503(kubernetes_auth_client, suppress_auth_errors):
    async def mock_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("Read timed out")

    with patch("httpx.AsyncClient.post", new=mock_timeout):
        response = kubernetes_auth_client.get("/test", headers={"Authorization": "Bearer token"})
        assert response.status_code == 503
        assert "Authentication service" in response.json()["error"]["message"]
