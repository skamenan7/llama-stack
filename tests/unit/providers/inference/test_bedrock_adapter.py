# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AuthenticationError

from llama_stack.apis.inference import OpenAIChatCompletionRequestWithExtraBody
from llama_stack.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from llama_stack.providers.remote.inference.bedrock.config import BedrockConfig


def test_adapter_initialization():
    config = BedrockConfig(api_key="test-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    assert adapter.config.auth_credential.get_secret_value() == "test-key"
    assert adapter.config.region_name == "us-east-1"


def test_client_url_construction():
    config = BedrockConfig(api_key="test-key", region_name="us-west-2")
    adapter = BedrockInferenceAdapter(config=config)

    assert adapter.get_base_url() == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"


def test_api_key_from_config():
    config = BedrockConfig(api_key="config-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)
    assert adapter.config.auth_credential.get_secret_value() == "config-key"


def test_api_key_from_header_overrides_config():
    """Test API key from request header overrides config via client property"""
    config = BedrockConfig(api_key="config-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)
    adapter.provider_data_api_key_field = "aws_bedrock_api_key"
    adapter.get_request_provider_data = MagicMock(return_value=SimpleNamespace(aws_bedrock_api_key="header-key"))

    # The client property is where header override happens (in OpenAIMixin)
    assert adapter.client.api_key == "header-key"


async def test_authentication_error_handling():
    """Test that AuthenticationError from OpenAI client is converted to ValueError with helpful message"""
    config = BedrockConfig(api_key="invalid-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    # Mock the parent class method to raise AuthenticationError
    mock_response = MagicMock()
    mock_response.message = "Invalid authentication credentials"
    auth_error = AuthenticationError(message="Invalid authentication credentials", response=mock_response, body=None)

    # Create a mock that raises the error
    mock_super = AsyncMock(side_effect=auth_error)

    # Patch the parent class method
    original_method = BedrockInferenceAdapter.__bases__[0].openai_chat_completion
    BedrockInferenceAdapter.__bases__[0].openai_chat_completion = mock_super

    try:
        with pytest.raises(ValueError) as exc_info:
            params = OpenAIChatCompletionRequestWithExtraBody(
                model="test-model", messages=[{"role": "user", "content": "test"}]
            )
            await adapter.openai_chat_completion(params=params)

        assert "AWS Bedrock authentication failed" in str(exc_info.value)
        assert "Please verify your API key" in str(exc_info.value)
    finally:
        # Restore original method
        BedrockInferenceAdapter.__bases__[0].openai_chat_completion = original_method
