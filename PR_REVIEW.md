# PR #4376 Review Summary

**PR**: feat: Refactor Agents API to use FastAPI Router
**Reviewed**: 2026-01-05
**Scope**: Changes specific to this PR only (not sweeping codebase changes)

---

## Quick Stats

| Category | Count |
|----------|-------|
| Critical (must fix) | 2 |
| Important (should fix) | 4 |
| Suggestions (nice to have) | 3 |

---

## Critical Issues

### 1. SSE Generator Swallows Exceptions Without Logging

**File**: `src/llama_stack_api/agents/fastapi_routes.py:55-69`

```python
except Exception as e:
    yield create_sse_event({"error": {"message": str(e)}})
    # NO LOGGING - operators have zero visibility into streaming failures
```

**Problem**: When streaming fails, the error is sent to the client but never logged. This makes debugging production issues impossible.

**Fix**: Add logging to match the existing pattern in `server.py`:

```python
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="agents")

async def sse_generator(event_gen):
    try:
        async for item in event_gen:
            yield create_sse_event(item)
    except asyncio.CancelledError:
        logger.info("SSE generator cancelled")  # Add this
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        logger.exception("Error in sse_generator")  # Add this
        yield create_sse_event({"error": {"message": str(e)}})
```

---

### 2. Missing Tests for 4 of 5 Endpoints

**File**: `tests/unit/core/routers/test_agents_router.py`

| Endpoint | Tested |
|----------|--------|
| `POST /v1/responses` | Yes (partial) |
| `GET /v1/responses/{id}` | No |
| `GET /v1/responses` | No |
| `GET /v1/responses/{id}/input_items` | No |
| `DELETE /v1/responses/{id}` | No |

**Fix**: Add at minimum one happy-path test per endpoint:

```python
async def test_get_response_returns_response_object():
    app = FastAPI()
    impl = AsyncMock(spec=Agents)
    impl.get_openai_response.return_value = OpenAIResponseObject(
        id="resp_123", created_at=0, model="test",
        object="response", output=[], status="completed"
    )
    router = build_fastapi_router(Api.agents, impl)
    app.include_router(router)

    get_endpoint = next(
        r.endpoint for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}"
        and "GET" in getattr(r, "methods", set())
    )

    from llama_stack_api.agents.models import RetrieveResponseRequest
    response = await get_endpoint(RetrieveResponseRequest(response_id="resp_123"))
    assert response.id == "resp_123"
```

---

## Important Issues

### 3. Missing `extra='forbid'` on Request Models

**File**: `src/llama_stack_api/agents/models.py`

**Problem**: Without `extra='forbid'`, typos in field names are silently ignored:

```python
# This silently ignores "streem" typo - no error raised
CreateResponseRequest(input="hi", model="test", streem=True)
```

**Fix**: Add to each request model class:

```python
from pydantic import ConfigDict

@json_schema_type
class CreateResponseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')  # Add this line
    # ... rest of fields
```

**Apply to**: `CreateResponseRequest`, `RetrieveResponseRequest`, `ListResponsesRequest`, `ListResponseInputItemsRequest`, `DeleteResponseRequest`, `ResponseGuardrailSpec`

---

### 4. No Validation Constraints on Numeric Fields

**File**: `src/llama_stack_api/agents/models.py`

**Problem**: Fields like `temperature` and `limit` accept invalid values like `temperature=-5`.

**Fix**: Add `ge`/`le` constraints:

```python
temperature: float | None = Field(
    default=None,
    ge=0.0,
    le=2.0,  # Add bounds
    description="Sampling temperature.",
)

max_infer_iters: int | None = Field(
    default=10,
    ge=1,  # Add minimum
    description="Maximum number of inference iterations.",
)

# In ListResponsesRequest:
limit: int | None = Field(
    default=50,
    ge=1,
    le=100,  # Add bounds
    description="The number of responses to return.",
)
```

---

### 5. Exception Chain Discarded

**File**: `src/llama_stack_api/agents/fastapi_routes.py:150, 174, 198, 212, 226`

**Problem**: Using `from None` loses the original stack trace:

```python
raise _http_exception_from_value_error(exc) from None  # Stack trace lost
```

**Fix**: Preserve the chain for debugging:

```python
raise _http_exception_from_value_error(exc) from exc  # Preserves stack trace
```

---

### 6. `ResponseItemInclude` Enum Not Used

**File**: `src/llama_stack_api/agents/models.py:146`

**Problem**: You defined `ResponseItemInclude` enum but don't use it:

```python
# Current - accepts any string
include: list[str] | None = Field(...)

# Should use the enum you created
include: list[ResponseItemInclude] | None = Field(...)
```

---

## Suggestions (Nice to Have)

### 7. Add `min_length=1` to ID Fields

**File**: `src/llama_stack_api/agents/models.py:123, 158`

```python
response_id: str = Field(
    ...,
    min_length=1,  # Prevent empty strings
    description="The ID of the OpenAI response to retrieve."
)
```

---

### 8. Test SSE Content Format

**File**: `tests/unit/core/routers/test_agents_router.py`

Current test only checks `media_type`. Consider also verifying the SSE format:

```python
async def test_sse_format_is_correct():
    # ... setup ...
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert events[0].startswith("data: ")
    assert events[0].endswith("\n\n")
```

---

### 9. Add Non-Streaming Response Test

**File**: `tests/unit/core/routers/test_agents_router.py`

Add a test for `stream=False` to verify it returns the object directly:

```python
async def test_create_response_returns_json_for_non_streaming():
    impl.create_openai_response.return_value = expected_response  # Not an iterator
    request = CreateResponseRequest(input="hi", model="test", stream=False)
    response = await create(request)

    assert not hasattr(response, 'media_type')  # Not StreamingResponse
    assert response.id == "resp_123"
```

---

## What's Good

- Clean Protocol design in `api.py`
- Excellent Field descriptions for OpenAPI documentation
- Proper SSE cancellation handling (re-raises `CancelledError`)
- Good OpenAPI schema test for POST endpoint
- Well-structured module layout

---

## Checklist Before Merge

**Must Fix:**
- [ ] Add logging to `sse_generator` (lines 55-69)
- [ ] Add `model_config = ConfigDict(extra='forbid')` to all request models
- [ ] Add basic tests for GET, DELETE, and list endpoints

**Should Fix:**
- [ ] Change `from None` to `from exc` in exception handling
- [ ] Add `ge`/`le` constraints to numeric fields
- [ ] Use `ResponseItemInclude` enum instead of `list[str]`

**Nice to Have:**
- [ ] Add `min_length=1` to `response_id` fields
- [ ] Add SSE format validation test
- [ ] Add non-streaming response test
