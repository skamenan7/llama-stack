# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from ogx_api import OpenAIResponseObject, OpenAIResponseObjectStreamResponseCreated


def _response(metadata: dict[str, str] | None = None) -> OpenAIResponseObject:
    return OpenAIResponseObject(
        id="resp_native",
        created_at=123,
        model="test-model",
        output=[],
        status="completed",
        metadata=metadata,
        store=False,
    )


async def test_native_passthrough_omitted_store_forces_provider_store_false(
    openai_responses_impl,
    mock_inference_api,
):
    openai_responses_impl.native_responses_passthrough = "auto"
    mock_inference_api.openai_response = AsyncMock(return_value=_response())

    result = await openai_responses_impl.create_openai_response(
        input="hello",
        model="vllm/test-model",
        explicit_request_fields={"input", "model"},
    )

    assert result.metadata["_ogx_execution"] == "provider_native"
    assert result.metadata["_ogx_native_fallback_reason"] == ""
    native_request = mock_inference_api.openai_response.call_args.args[0]
    assert native_request.store is False
    assert native_request.metadata is None


async def test_native_passthrough_overwrites_reserved_metadata_keys(
    openai_responses_impl,
    mock_inference_api,
):
    openai_responses_impl.native_responses_passthrough = "auto"
    mock_inference_api.openai_response = AsyncMock(return_value=_response({"_ogx_execution": "client", "keep": "yes"}))

    result = await openai_responses_impl.create_openai_response(
        input="hello",
        model="vllm/test-model",
        metadata={"_ogx_execution": "client", "keep": "yes"},
        explicit_request_fields={"input", "model", "metadata"},
    )

    assert result.metadata["keep"] == "yes"
    assert result.metadata["_ogx_execution"] == "provider_native"
    native_request = mock_inference_api.openai_response.call_args.args[0]
    assert native_request.metadata == {"keep": "yes"}


async def test_native_passthrough_stream_marks_response_events(
    openai_responses_impl,
    mock_inference_api,
):
    openai_responses_impl.native_responses_passthrough = "auto"

    async def native_stream():
        yield OpenAIResponseObjectStreamResponseCreated(response=_response(), sequence_number=0)

    mock_inference_api.openai_response = AsyncMock(return_value=native_stream())

    stream = await openai_responses_impl.create_openai_response(
        input="hello",
        model="vllm/test-model",
        stream=True,
        explicit_request_fields={"input", "model", "stream"},
    )

    events = [event async for event in stream]
    assert events[0].response.metadata["_ogx_execution"] == "provider_native"
    native_request = mock_inference_api.openai_response.call_args.args[0]
    assert native_request.stream is True


async def test_required_native_passthrough_rejects_explicit_store_true(openai_responses_impl):
    openai_responses_impl.native_responses_passthrough = "required"

    with pytest.raises(ValueError, match="requires_storage"):
        await openai_responses_impl.create_openai_response(
            input="hello",
            model="vllm/test-model",
            store=True,
            explicit_request_fields={"input", "model", "store"},
        )


async def test_required_native_passthrough_unsupported_provider_is_runtime_error(
    openai_responses_impl,
    mock_inference_api,
):
    openai_responses_impl.native_responses_passthrough = "required"
    mock_inference_api.openai_response = AsyncMock(side_effect=NotImplementedError())

    with pytest.raises(RuntimeError, match="does not support"):
        await openai_responses_impl.create_openai_response(
            input="hello",
            model="openai/test-model",
            explicit_request_fields={"input", "model"},
        )
