# Agents in Llama Stack

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [State Management Deep Dive](#state-management-deep-dive)
5. [The Agentic Loop](#the-agentic-loop)
6. [Tool Execution](#tool-execution)
7. [Streaming Architecture](#streaming-architecture)
8. [API Reference](#api-reference)
9. [Implementation Details](#implementation-details)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Executive Summary

Agents in Llama Stack are **stateful AI systems** that orchestrate multi-turn interactions between LLMs, tools, and external services. Unlike simple chat completions, agents maintain conversation history, execute tools autonomously, and manage complex workflows through an iterative reasoning loop.

**Key Capabilities:**
- Multi-turn conversations with persistent state
- Automatic tool calling (web search, file search, code execution, MCP servers, custom functions)
- Streaming responses with fine-grained event reporting
- Safety guardrails for input/output validation
- Session management through multiple state persistence mechanisms

**When to Use Agents:**
- Building chatbots that need memory across conversations
- Creating autonomous workflows that require tool execution
- Implementing RAG (Retrieval Augmented Generation) systems
- Developing interactive applications with complex state management

---

## Architecture Overview

### System Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agents API Layer                          │
│  (Protocol definition in llama_stack_api/agents/api.py)         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              Meta Reference Implementation                       │
│  (src/llama_stack/providers/inline/agents/meta_reference/)      │
│                                                                  │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ OpenAIResponses│  │StreamingResponse │  │  ToolExecutor  │  │
│  │      Impl      │──│  Orchestrator    │──│                │  │
│  └────────┬───────┘  └──────────────────┘  └────────────────┘  │
│           │                                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────┐
│                    State Persistence Layer                        │
│                                                                   │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │  Responses   │  │  Conversations │  │ conversation_messages│ │
│  │    Store     │  │      API       │  │    (cache table)     │ │
│  │  (SQL table) │  │  (SQL tables)  │  │                      │ │
│  └──────────────┘  └────────────────┘  └──────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────┐
│                      External Dependencies                        │
│                                                                   │
│  Inference API │ Tool Runtime │ Safety API │ Vector IO │ Files   │
└───────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Agents API Protocol** (`llama_stack_api/agents/api.py`)
   - Defines the contract for agent operations
   - Based on OpenAI Responses API design
   - Supports create, retrieve, list, delete operations

2. **Meta Reference Implementation**
   - `OpenAIResponsesImpl`: Core orchestration logic
   - `StreamingResponseOrchestrator`: Manages streaming events and tool execution loops
   - `ToolExecutor`: Handles tool invocation and result processing

3. **State Persistence**
   - `ResponsesStore`: Persists response objects with full history
   - `Conversations API`: Manages conversation sessions
   - Message cache: Stores chat completion messages for efficient continuation

---

## Core Concepts

### What is a Response?

A **Response** is the fundamental unit of agent interaction. It represents:
- User input (text, images, files)
- Model output (text, tool calls, reasoning)
- Execution history (tool results, intermediate steps)
- Metadata (tokens used, creation time, status)

```python
# From llama_stack_api/openai_responses/models.py
class OpenAIResponseObject(BaseModel):
    id: str                           # resp_{uuid}
    created_at: int                   # Unix timestamp
    model: str                        # LLM identifier
    status: str                       # completed | incomplete | failed
    output: list[OpenAIResponseOutput]  # Messages, tool calls, etc.
    usage: OpenAIResponseUsage | None   # Token counts
    instructions: str | None          # System prompt
    tools: list[OpenAIResponseInputTool] | None
    metadata: dict[str, str] | None
```

### Input Types

Agents accept flexible input formats:

```python
# Simple string
input = "What's the weather in Tokyo?"

# Structured messages
input = [
    OpenAIResponseMessage(
        role="user",
        content=[
            OpenAIResponseInputMessageContentText(text="Analyze this image"),
            OpenAIResponseInputMessageContentImage(
                image_url={"url": "data:image/png;base64,..."}
            )
        ]
    )
]
```

### Tool Types

Agents support multiple tool categories:

1. **Function Tools** (client-side execution)
   ```python
   {
       "type": "function",
       "name": "get_stock_price",
       "description": "Get current stock price",
       "parameters": {...}
   }
   ```

2. **Built-in Tools** (server-side execution)
   - `web_search`: Internet search
   - `file_search`: RAG over vector stores
   - `code_interpreter`: Python execution

3. **MCP Tools** (Model Context Protocol servers)
   ```python
   {
       "type": "mcp",
       "server_url": "http://localhost:8000",
       "server_label": "my_mcp_server"
   }
   ```

---

## State Management Deep Dive

This is the most critical section for understanding how agents maintain state across interactions.

### 1. ResponsesStore: Persistent Response History

**File:** `/src/llama_stack/providers/utils/responses/responses_store.py`

**What it does:**
Persists complete response objects to SQL database, storing:
- Response metadata (id, created_at, model, status)
- Full input history (all messages sent to the model)
- Complete output history (all messages and tool calls generated)
- Chat completion messages (OpenAI-format messages for efficient continuation)

**Storage schema:**
```sql
CREATE TABLE openai_responses (
    id TEXT PRIMARY KEY,           -- resp_{uuid}
    created_at INTEGER,            -- Unix timestamp
    model TEXT,                    -- Model identifier
    response_object JSON           -- Full response + input + messages
);

CREATE TABLE conversation_messages (
    conversation_id TEXT PRIMARY KEY,
    messages JSON                  -- List of OpenAIMessageParam
);
```

**When to use:**
- Retrieving historical responses by ID
- Auditing conversation history
- Implementing undo/redo functionality
- Debugging agent behavior

**Pros:**
- Complete audit trail
- Supports response retrieval by ID
- Enables list_responses pagination
- Stores original input alongside output

**Cons:**
- More storage overhead
- Not designed for real-time message access
- Requires explicit cleanup for old responses

**Code example:**
```python
# Store a response (happens automatically when store=True)
await responses_store.store_response_object(
    response_object=response,
    input=all_input_items,
    messages=chat_completion_messages
)

# Retrieve later
response_with_input = await responses_store.get_response_object("resp_abc123")
# Returns: _OpenAIResponseObjectWithInputAndMessages
#   - response_with_input.id
#   - response_with_input.input (original user input)
#   - response_with_input.output (model output)
#   - response_with_input.messages (chat completion format)

# List all responses
responses = await responses_store.list_responses(
    after="resp_xyz",
    limit=50,
    model="llama3-70b",
    order=Order.desc
)
```

**Relevant code locations:**
- Storage: `/src/llama_stack/providers/utils/responses/responses_store.py:84-113`
- Retrieval: `/src/llama_stack/providers/utils/responses/responses_store.py:156-173`
- Usage: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:543-548`

---

### 2. Conversations API: Session Management

**Files:**
- `/src/llama_stack_api/conversations/api.py`
- `/src/llama_stack/core/conversations/conversations.py`

**What it does:**
Manages conversation sessions as first-class objects with:
- Unique conversation IDs (`conv_{uuid}`)
- Metadata (key-value pairs)
- Items (messages, tool calls, tool outputs)
- Lifecycle management (create, update, delete)

**Storage schema:**
```sql
CREATE TABLE openai_conversations (
    id TEXT PRIMARY KEY,              -- conv_{uuid}
    created_at INTEGER,               -- Unix timestamp
    items JSON,                       -- Unused (items stored separately)
    metadata JSON                     -- User-defined key-value pairs
);

CREATE TABLE conversation_items (
    id TEXT PRIMARY KEY,              -- msg_{uuid} or item_{uuid}
    conversation_id TEXT,             -- Foreign key to openai_conversations
    created_at INTEGER,               -- Unix timestamp
    item_data JSON                    -- Full item payload
);
```

**When to use:**
- Building multi-session chatbots
- Grouping related interactions
- Implementing conversation-level metadata (user ID, tags, etc.)
- Supporting conversation list/search UIs

**Pros:**
- Clean session boundaries
- Metadata support for filtering/search
- Independent of response storage
- Supports conversation-level operations (archive, share, etc.)

**Cons:**
- Requires explicit conversation creation
- Items are denormalized (duplicated if in multiple conversations)
- No built-in support for conversation branching

**Code example:**
```python
# Create a conversation
conversation = await conversations_api.create_conversation(
    CreateConversationRequest(
        metadata={"user_id": "user123", "topic": "weather"}
    )
)
# Returns: Conversation(id="conv_abc...", created_at=..., metadata={...})

# Use conversation in response
response = await agents_api.create_openai_response(
    input="What's the weather?",
    model="llama3-70b",
    conversation=conversation.id  # Link to conversation
)

# List conversation history
items = await conversations_api.list_items(
    ListItemsRequest(
        conversation_id=conversation.id,
        order="asc",
        limit=100
    )
)
# Returns: ConversationItemList with all messages and tool calls

# Add items manually (for pre-loading context)
await conversations_api.add_items(
    conversation_id=conversation.id,
    request=AddItemsRequest(items=[
        OpenAIResponseMessage(role="user", content="Hello"),
        OpenAIResponseMessage(role="assistant", content="Hi there!")
    ])
)
```

**Relevant code locations:**
- Implementation: `/src/llama_stack/core/conversations/conversations.py:94-229`
- API definition: `/src/llama_stack_api/conversations/api.py:27-44`
- Syncing: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:559-577`

---

### 3. previous_response_id: Chaining Responses

**Files:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py`

**What it does:**
Links responses in a chain by prepending all previous input/output to the current request.

**Processing flow:**
```python
# When previous_response_id is provided:
# 1. Fetch previous response from ResponsesStore
previous_response = await responses_store.get_response_object(previous_response_id)

# 2. Prepend all previous input + output to new input
all_input = previous_response.input + previous_response.output + new_input

# 3. Use stored messages for efficient chat completion
if previous_response.messages:
    messages = previous_response.messages + convert_new_input(new_input)
else:
    # Backward compatibility: reconstruct from inputs
    messages = convert_all_inputs(all_input)
```

**When to use:**
- Linear conversation flows (no branching)
- Continuing from a specific response
- Building undo/redo with explicit checkpoints

**Pros:**
- Simple mental model (explicit parent reference)
- Works with any response (even from different conversations)
- No separate session management needed

**Cons:**
- Only supports linear chains (no branching)
- Cannot be used with `conversation` parameter (mutually exclusive)
- Requires client to track response IDs

**Code example:**
```python
# First turn
response1 = await agents_api.create_openai_response(
    input="What's 2+2?",
    model="llama3-70b"
)
# Returns: OpenAIResponseObject(id="resp_abc123", output=[...])

# Second turn - continue from previous response
response2 = await agents_api.create_openai_response(
    input="Now multiply that by 3",
    model="llama3-70b",
    previous_response_id=response1.id  # Links to previous
)
# The model will see:
# [previous input] What's 2+2?
# [previous output] 2+2 equals 4
# [new input] Now multiply that by 3
```

**Relevant code locations:**
- Input processing: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:120-187`
- Validation: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:376-383`

---

### 4. conversation field: Session Grouping

**Files:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py`

**What it does:**
Groups multiple responses under a single conversation session, automatically syncing items to the Conversations API.

**Processing flow:**
```python
# When conversation is provided:
# 1. Fetch conversation items from Conversations API
conversation_items = await conversations_api.list_items(
    ListItemsRequest(conversation_id=conversation, order="asc")
)

# 2. Fetch cached messages (optimization)
stored_messages = await responses_store.get_conversation_messages(conversation)

# 3. Build message history
if stored_messages:
    # Use cached messages + convert new input
    messages = stored_messages + convert_new_input(new_input)
else:
    # First turn or pre-messages: reconstruct from items
    all_input = conversation_items.data + new_input
    messages = convert_all_inputs(all_input)

# 4. After response completion, sync back to conversation
await conversations_api.add_items(conversation_id, new_items)
await responses_store.store_conversation_messages(conversation_id, messages)
```

**When to use:**
- Multi-session applications (multiple concurrent conversations)
- When conversation metadata is needed (user ID, tags, etc.)
- Supporting conversation-level operations (search, archive, share)

**Pros:**
- Natural session boundaries
- Metadata support
- Automatic item syncing
- Message caching for performance

**Cons:**
- Requires Conversations API to be configured
- More complex setup than previous_response_id
- Cannot be used with previous_response_id (mutually exclusive)
- Conversation must exist before use

**Code example:**
```python
# Create conversation
conversation = await conversations_api.create_conversation(
    CreateConversationRequest(metadata={"user_id": "alice"})
)

# First turn
response1 = await agents_api.create_openai_response(
    input="Hello",
    model="llama3-70b",
    conversation=conversation.id
)

# Second turn - automatically includes first turn context
response2 = await agents_api.create_openai_response(
    input="What did I just say?",
    model="llama3-70b",
    conversation=conversation.id  # Same conversation
)
# The model will see:
# [turn 1 input] Hello
# [turn 1 output] Hi there! How can I help you?
# [turn 2 input] What did I just say?

# View full conversation history
items = await conversations_api.list_items(
    ListItemsRequest(conversation_id=conversation.id)
)
# Returns all messages and tool calls in order
```

**Relevant code locations:**
- Input processing: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:152-182`
- Syncing: `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py:550-552`
- Message caching: `/src/llama_stack/providers/utils/responses/responses_store.py:240-282`

---

### State Management Comparison

| Feature | ResponsesStore | Conversations API | previous_response_id | conversation field |
|---------|---------------|-------------------|----------------------|-------------------|
| **Purpose** | Audit trail | Session management | Linear chaining | Session grouping |
| **Storage** | SQL (responses) | SQL (conversations + items) | Reference only | Reference only |
| **Granularity** | Per response | Per conversation | Per response | Per conversation |
| **Metadata** | Response-level | Conversation-level | N/A | Conversation-level |
| **Branching** | No | Yes (separate conversations) | No | Yes (separate conversations) |
| **Overhead** | Medium (stores everything) | Medium (stores items) | Low (reference only) | Low (reference only) |
| **Use Cases** | Debugging, auditing | Multi-session apps | Linear workflows | Complex apps |
| **Mutually Exclusive** | No | No | Yes (with conversation) | Yes (with previous_response_id) |

### How They Interact

```python
# Typical flow for conversation-based app:

# 1. Create conversation (Conversations API)
conversation = await conversations_api.create_conversation(...)

# 2. First response (uses conversation field)
response1 = await agents_api.create_openai_response(
    input="Hello",
    conversation=conversation.id,
    store=True  # Stores in ResponsesStore
)
# Behind the scenes:
# - ResponsesStore: Stores response + input + messages
# - Conversations API: Syncs input + output to conversation items
# - Message cache: Stores messages for next turn

# 3. Second response (uses conversation field)
response2 = await agents_api.create_openai_response(
    input="What did I say?",
    conversation=conversation.id,
    store=True
)
# Behind the scenes:
# - Fetches conversation items from Conversations API
# - Loads cached messages from ResponsesStore
# - Prepends to new input
# - Stores new response in ResponsesStore
# - Syncs new items to Conversations API
# - Updates message cache

# 4. Retrieve response later (ResponsesStore)
old_response = await agents_api.get_openai_response(
    RetrieveResponseRequest(response_id=response1.id)
)

# 5. View conversation history (Conversations API)
items = await conversations_api.list_items(
    ListItemsRequest(conversation_id=conversation.id)
)
```

### Design Decisions & Rationale

**Why separate ResponsesStore and Conversations API?**
- **Separation of concerns:** Responses track individual LLM interactions; conversations track sessions
- **Flexibility:** Responses can exist outside conversations (for one-off queries)
- **Performance:** Message caching optimizes multi-turn conversations
- **Compatibility:** Follows OpenAI's API design (separate `/responses` and `/conversations` endpoints)

**Why store messages separately from input/output?**
- **Efficiency:** Chat completion messages are in optimized format (no conversion needed)
- **Backward compatibility:** Old responses stored before messages feature still work
- **Accuracy:** Avoids format conversion errors when continuing conversations

**Why mutually exclusive previous_response_id and conversation?**
- **Clarity:** Two different mental models (linear chain vs. session)
- **Consistency:** Prevents ambiguity about which history to use
- **Simplicity:** Easier to reason about state

---

## The Agentic Loop

The agentic loop is the core iterative process that enables agents to autonomously reason, call tools, and refine their responses.

### Loop Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    User Input + Context                       │
│  (from previous_response_id or conversation or standalone)    │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                 Iteration 0 (Initial Turn)                    │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 1. Input Safety Check (if guardrails enabled)         │  │
│  │    - Run guardrails on combined message text          │  │
│  │    - If violation: emit refusal response and stop     │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ 2. Tool Processing                                     │ │
│  │    - Process MCP tool definitions (list_tools)         │ │
│  │    - Convert response tools to chat completion tools   │ │
│  │    - Build tool_choice from user preferences           │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ 3. Chat Completion                                     │ │
│  │    - Call inference_api.openai_chat_completion         │ │
│  │    - Stream chunks and emit delta events               │ │
│  │    - Accumulate: text, tool_calls, logprobs, usage     │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ 4. Output Safety Check (if guardrails enabled)        │ │
│  │    - Run guardrails on accumulated text                │ │
│  │    - If violation: emit refusal response and stop      │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ 5. Tool Call Decision                                  │ │
│  │    - Separate function vs. non-function tool calls     │ │
│  │    - Check MCP approval requirements                   │ │
│  │    - If function tools: STOP (client-side execution)   │ │
│  │    - If non-function tools: proceed to step 6          │ │
│  │    - If no tools: STOP (response completed)            │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ 6. Execute Non-Function Tools (server-side)            │ │
│  │    - For each tool call:                               │ │
│  │      * Check max_tool_calls limit                      │ │
│  │      * Emit progress events (in_progress, searching)   │ │
│  │      * Execute tool (web_search, file_search, MCP)     │ │
│  │      * Emit completion events (completed, failed)      │ │
│  │      * Append tool result to messages                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                    │
└──────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│            Iteration 1...N (Subsequent Turns)                 │
│                                                               │
│  - Prepend tool results to messages                          │
│  - Reset tool_choice to "auto" (avoid infinite loops)        │
│  - Repeat steps 3-6                                          │
│  - Stop conditions:                                          │
│    * Model returns no tool calls (completed)                 │
│    * Model returns function tool calls (client-side)         │
│    * max_infer_iters reached (incomplete)                    │
│    * Model finish_reason == "length" (incomplete)            │
│    * Error occurs (failed)                                   │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Store Response (if store=True)               │
│                                                               │
│  - Write to ResponsesStore (response + input + messages)     │
│  - If conversation: sync items to Conversations API          │
│  - If conversation: cache messages for next turn             │
└──────────────────────────────────────────────────────────────┘
```

### Loop Control Parameters

```python
# From CreateResponseRequest
max_infer_iters: int = 10          # Max iterations (default 10)
max_tool_calls: int | None = None  # Max total built-in tool calls
parallel_tool_calls: bool = True   # Allow multiple tool calls per turn
tool_choice: str | dict = "auto"   # Tool selection strategy
```

**max_infer_iters:**
Controls the maximum number of inference calls (chat completion rounds). Each iteration can generate multiple tool calls (if parallel_tool_calls=True).

**max_tool_calls:**
Limits the total number of built-in (server-side) tool calls across all iterations. When reached, subsequent tool calls are skipped with a message.

**parallel_tool_calls:**
- `True` (default): Model can generate multiple tool calls in a single turn
- `False`: Model generates at most one tool call per turn

**tool_choice:**
- `"auto"`: Model decides whether to call tools
- `"required"`: Model must call at least one tool
- `{"type": "function", "name": "tool_name"}`: Force specific tool
- `{"allowed_tools": [...], "mode": "required"}`: Restrict to specific tools

### Loop Termination

**Completed** (status="completed"):
- Model returns no tool calls
- All tool executions successful

**Incomplete** (status="incomplete"):
- max_infer_iters reached
- Model finish_reason == "length" (hit token limit)

**Failed** (status="failed"):
- Exception during processing
- Guardrail violation
- Model error

**Client-side function call:**
Model generates function tool call, which requires client to execute and submit results.

**Relevant code:**
- Main loop: `/src/llama_stack/providers/inline/agents/meta_reference/responses/streaming.py:234-439`
- Tool execution: `/src/llama_stack/providers/inline/agents/meta_reference/responses/streaming.py:992-1127`
- Termination logic: `/src/llama_stack/providers/inline/agents/meta_reference/responses/streaming.py:389-410`

---

## Tool Execution

### Tool Categories

**1. Function Tools (Client-Side)**

Definition:
```python
{
    "type": "function",
    "name": "get_stock_price",
    "description": "Get current stock price for a symbol",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"]
    }
}
```

Execution flow:
1. Model generates tool call in chat completion
2. Agent stops iteration and returns response with `function_tool_call` output item
3. Client executes function locally
4. Client submits result via new response with previous_response_id or conversation

**2. Built-in Tools (Server-Side)**

**Web Search:**
```python
{
    "type": "web_search",
    "max_results": 10,
    "search_engine": "brave"
}
```
- Backed by `web_search` tool in Tool Runtime
- Executes search and returns formatted results
- Emits streaming events: in_progress → searching → completed

**File Search (RAG):**
```python
{
    "type": "file_search",
    "vector_store_ids": ["vs_abc123"],
    "filters": {"file_name": {"eq": "report.pdf"}},
    "max_num_results": 5,
    "ranking_options": {"rank_fields": ["score"], "score_threshold": 0.7}
}
```
- Queries vector stores via VectorIO API
- Supports filters, ranking, multi-store search
- Injects results into context with citation annotations
- Emits streaming events: in_progress → searching → completed

**3. MCP Tools (Model Context Protocol)**

Definition:
```python
{
    "type": "mcp",
    "server_url": "http://localhost:8000",
    "server_label": "github_mcp",
    "authorization": {"type": "bearer", "token": "ghp_..."},
    "allowed_tools": ["create_issue", "list_repos"],
    "require_approval": "always"  # or "never" or {"always": [...], "never": [...]}
}
```

Execution flow:
1. Agent calls `list_mcp_tools` to discover available tools
2. Emits `mcp_list_tools.in_progress` → `mcp_list_tools.completed`
3. Filters tools by `allowed_tools` configuration
4. Adds filtered tools to chat completion tools list
5. When model calls MCP tool:
   - Check approval requirements
   - If approval needed: emit approval request and wait for client
   - Else: invoke tool via MCP protocol
   - Emit `mcp_call.in_progress` → `mcp_call.completed` | `mcp_call.failed`

### Tool Execution Architecture

**File:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/tool_executor.py`

```python
class ToolExecutor:
    async def execute_tool_call(
        self,
        tool_call: OpenAIChatCompletionToolCall,
        ctx: ChatCompletionContext,
        sequence_number: int,
        output_index: int,
        item_id: str,
        mcp_tool_to_server: dict | None = None,
    ) -> AsyncIterator[ToolExecutionResult]:
        """Execute a tool call and yield streaming events."""

        # 1. Emit progress events
        async for event in self._emit_progress_events(...):
            yield event

        # 2. Execute tool
        if tool_call.name in mcp_tool_to_server:
            result = await invoke_mcp_tool(...)
        elif tool_call.name == "knowledge_search":
            result = await self._execute_knowledge_search_via_vector_store(...)
        else:
            result = await tool_runtime_api.invoke_tool(...)

        # 3. Emit completion events
        async for event in self._emit_completion_events(...):
            yield event

        # 4. Build result messages
        output_message, input_message = await self._build_result_messages(...)

        # 5. Yield final result
        yield ToolExecutionResult(
            final_output_message=output_message,  # For response.output
            final_input_message=input_message,     # For next chat completion
            citation_files=...                     # For annotation
        )
```

### Tool Result Formatting

Tool results are formatted differently depending on type:

**Web Search:**
```python
OpenAIResponseOutputMessageWebSearchToolCall(
    id="ws_abc123",
    status="completed",
    # No results field - actual results injected via tool message
)
# Corresponding tool message:
OpenAIToolMessageParam(
    tool_call_id="call_abc123",
    content="Search results:\n1. Title: ...\nURL: ...\n\n2. ..."
)
```

**File Search:**
```python
OpenAIResponseOutputMessageFileSearchToolCall(
    id="fs_abc123",
    queries=["machine learning"],
    status="completed",
    results=[
        OpenAIResponseOutputMessageFileSearchToolCallResults(
            file_id="file_xyz",
            filename="ml_paper.pdf",
            text="Machine learning is...",
            score=0.92,
            attributes={"page": 3}
        )
    ]
)
# Corresponding tool message:
OpenAIToolMessageParam(
    tool_call_id="call_abc123",
    content="[Retrieved context with citations]\n[1] file_xyz: Machine learning is..."
)
```

**MCP Tools:**
```python
OpenAIResponseOutputMessageMCPCall(
    id="mcp_abc123",
    name="create_issue",
    arguments='{"title": "Bug report", "body": "..."}',
    server_label="github_mcp",
    output="Created issue #123: https://github.com/..."  # or None if failed
    error="Error (code 403): Unauthorized"  # or None if successful
)
# Corresponding tool message:
OpenAIToolMessageParam(
    tool_call_id="call_abc123",
    content="Created issue #123: https://github.com/..."
)
```

### Tool Approval Workflow (MCP)

```python
# Request configuration
{
    "type": "mcp",
    "server_label": "github_mcp",
    "require_approval": "always"
}

# Flow:
# 1. Model generates tool call
# 2. Agent checks approval requirement
if self._approval_required(tool_call.name):
    # 3. Check if client provided approval in context
    approval_response = ctx.approval_response(tool_call.name, tool_call.arguments)
    if approval_response:
        if approval_response.approve:
            # Execute tool
        else:
            # Skip tool (don't add to messages)
    else:
        # 4. Emit approval request
        yield OpenAIResponseMCPApprovalRequest(
            id="approval_abc123",
            name=tool_call.name,
            arguments=tool_call.arguments,
            server_label=server_label
        )
        # 5. Stop iteration and wait for client to submit approval
```

Client flow:
```python
# 1. Receive approval request in response.output
approval_request = response.output[0]  # type: OpenAIResponseMCPApprovalRequest

# 2. Prompt user for approval
user_approved = input(f"Approve {approval_request.name}? (y/n): ") == "y"

# 3. Submit approval via new response
await agents_api.create_openai_response(
    input=[
        OpenAIResponseMCPApprovalResponse(
            id=approval_request.id,
            approve=user_approved,
            reason="User declined" if not user_approved else None
        )
    ],
    previous_response_id=response.id,  # or conversation=...
    model="llama3-70b"
)
```

---

## Streaming Architecture

Streaming is the default mode for all agent interactions. It provides fine-grained progress updates and enables real-time UX.

### Event Types

**Response Lifecycle:**
```python
response.created          # Initial response object created
response.in_progress      # Processing started (emitted after guardrails)
response.completed        # Successfully finished
response.incomplete       # Stopped due to max_infer_iters or token limit
response.failed           # Error occurred
```

**Output Items:**
```python
response.output_item.added   # New item added to output (message, tool call, etc.)
response.output_item.done    # Item finalized
```

**Text Content:**
```python
response.content_part.added         # New content part added (text, reasoning, refusal)
response.output_text.delta          # Incremental text delta
response.content_part.done          # Content part finalized (with full text)

response.reasoning_text.delta       # Incremental reasoning delta (o1/o3 models)
response.reasoning_text.done        # Reasoning finalized

response.refusal.delta              # Incremental refusal delta (guardrail violation)
response.refusal.done               # Refusal finalized
```

**Tool Calls:**
```python
response.function_call.arguments.delta    # Function tool arguments delta
response.function_call.arguments.done     # Function tool arguments finalized

response.mcp_call.arguments.delta         # MCP tool arguments delta
response.mcp_call.arguments.done          # MCP tool arguments finalized

response.web_search_call.in_progress      # Web search started
response.web_search_call.searching        # Web search executing
response.web_search_call.completed        # Web search finished

response.file_search_call.in_progress     # File search started
response.file_search_call.searching       # File search executing
response.file_search_call.completed       # File search finished

response.mcp_call.in_progress             # MCP tool started
response.mcp_call.completed               # MCP tool finished
response.mcp_call.failed                  # MCP tool error

response.mcp_list_tools.in_progress       # MCP tool discovery started
response.mcp_list_tools.completed         # MCP tool discovery finished
```

### Streaming Example

```python
# Create streaming response
response_stream = await agents_api.create_openai_response(
    input="Search for weather in Tokyo and summarize",
    model="llama3-70b",
    tools=[{"type": "web_search"}],
    stream=True
)

# Process events
async for event in response_stream:
    match event.type:
        case "response.created":
            print(f"Response ID: {event.response.id}")

        case "response.output_item.added":
            if event.item.type == "message":
                print(f"Message started: {event.item.id}")
            elif event.item.type == "web_search_call":
                print(f"Web search started: {event.item.id}")

        case "response.output_text.delta":
            print(event.delta, end="", flush=True)

        case "response.web_search_call.searching":
            print("\n[Searching web...]")

        case "response.web_search_call.completed":
            print("\n[Search complete]")

        case "response.output_item.done":
            if event.item.type == "message":
                print(f"\nMessage complete: {event.item.content[0].text}")
            elif event.item.type == "web_search_call":
                print(f"\nSearch complete")

        case "response.completed":
            print(f"\n\nFinal response: {event.response.status}")
            print(f"Tokens used: {event.response.usage.total_tokens}")
```

### Non-Streaming Mode

For convenience, the API also supports non-streaming mode:

```python
response = await agents_api.create_openai_response(
    input="What's the weather?",
    model="llama3-70b",
    stream=False  # or omit (default is False)
)
# Returns: OpenAIResponseObject (final state)
# Internally:
# - Still uses streaming under the hood
# - Accumulates all events
# - Returns only the final "response.completed" event
```

### Sequence Numbers

Each streaming event includes a `sequence_number` field for ordering:

```python
class OpenAIResponseObjectStreamResponseOutputTextDelta(BaseModel):
    type: Literal["response.output_text.delta"]
    sequence_number: int  # Monotonically increasing
    delta: str
    item_id: str
    output_index: int
    content_index: int
```

Use sequence numbers to:
- Detect out-of-order delivery (if transport is unreliable)
- Resume streaming from a specific point
- Debug event ordering issues

---

## API Reference

### Create Response

**Endpoint:** `POST /agents/responses`

**Request:**
```python
CreateResponseRequest(
    # Required
    input: str | list[OpenAIResponseInput],
    model: str,

    # Optional - State management
    previous_response_id: str | None = None,
    conversation: str | None = None,
    store: bool = True,

    # Optional - Streaming
    stream: bool = False,

    # Optional - Model parameters
    temperature: float | None = None,
    text: OpenAIResponseText | None = None,  # Structured outputs

    # Optional - Tools
    tools: list[OpenAIResponseInputTool] | None = None,
    tool_choice: OpenAIResponseInputToolChoice | None = None,
    parallel_tool_calls: bool = True,
    max_tool_calls: int | None = None,

    # Optional - Prompts
    prompt: OpenAIResponsePrompt | None = None,
    instructions: str | None = None,

    # Optional - Safety
    guardrails: list[str | ResponseGuardrailSpec] | None = None,

    # Optional - Loop control
    max_infer_iters: int = 10,

    # Optional - Metadata
    metadata: dict[str, str] | None = None,
    include: list[ResponseItemInclude] | None = None,
)
```

**Response (non-streaming):**
```python
OpenAIResponseObject(
    id: str,                    # resp_{uuid}
    created_at: int,            # Unix timestamp
    model: str,
    status: str,                # completed | incomplete | failed
    output: list[OpenAIResponseOutput],
    usage: OpenAIResponseUsage | None,
    error: OpenAIResponseError | None,
    # Echo back request parameters
    instructions: str | None,
    tools: list[OpenAIResponseInputTool] | None,
    tool_choice: str | dict | None,
    text: OpenAIResponseText | None,
    prompt: OpenAIResponsePrompt | None,
    parallel_tool_calls: bool | None,
    max_tool_calls: int | None,
    metadata: dict[str, str] | None,
)
```

**Response (streaming):**
```python
AsyncIterator[OpenAIResponseObjectStream]
# Union of all event types (see Streaming Architecture)
```

**Example:**
```python
response = await agents_api.create_openai_response(
    input="What's the capital of France?",
    model="llama3-70b",
    temperature=0.7,
    instructions="Be concise and factual.",
    metadata={"user_id": "alice", "session_id": "sess_123"}
)
```

### Retrieve Response

**Endpoint:** `GET /agents/responses/{response_id}`

**Request:**
```python
RetrieveResponseRequest(
    response_id: str
)
```

**Response:**
```python
OpenAIResponseObject  # Same as create response
```

**Example:**
```python
response = await agents_api.get_openai_response(
    RetrieveResponseRequest(response_id="resp_abc123")
)
```

### List Responses

**Endpoint:** `GET /agents/responses`

**Request:**
```python
ListResponsesRequest(
    after: str | None = None,       # Pagination cursor
    limit: int = 50,                # 1-100
    model: str | None = None,       # Filter by model
    order: Order = Order.desc,      # asc | desc
)
```

**Response:**
```python
ListOpenAIResponseObject(
    object: "list",
    data: list[OpenAIResponseObjectWithInput],
    first_id: str,
    last_id: str,
    has_more: bool
)
```

**Example:**
```python
responses = await agents_api.list_openai_responses(
    ListResponsesRequest(
        limit=20,
        model="llama3-70b",
        order=Order.desc
    )
)
for response_with_input in responses.data:
    print(f"{response_with_input.id}: {response_with_input.input[0].content}")
```

### List Response Input Items

**Endpoint:** `GET /agents/responses/{response_id}/input_items`

**Request:**
```python
ListResponseInputItemsRequest(
    response_id: str,
    after: str | None = None,
    before: str | None = None,
    include: list[ResponseItemInclude] | None = None,
    limit: int = 20,
    order: Order = Order.desc,
)
```

**Response:**
```python
ListOpenAIResponseInputItem(
    object: "list",
    data: list[OpenAIResponseInput]  # Messages, tool outputs, etc.
)
```

**Example:**
```python
input_items = await agents_api.list_openai_response_input_items(
    ListResponseInputItemsRequest(
        response_id="resp_abc123",
        limit=10,
        order=Order.asc
    )
)
for item in input_items.data:
    if item.type == "message":
        print(f"{item.role}: {item.content[0].text}")
```

### Delete Response

**Endpoint:** `DELETE /agents/responses/{response_id}`

**Request:**
```python
DeleteResponseRequest(
    response_id: str
)
```

**Response:**
```python
OpenAIDeleteResponseObject(
    id: str,
    object: "response.deleted",
    deleted: bool
)
```

**Example:**
```python
result = await agents_api.delete_openai_response(
    DeleteResponseRequest(response_id="resp_abc123")
)
```

---

## Implementation Details

### Response Construction Flow

**File:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py`

```python
async def create_openai_response(...):
    # 1. Validate inputs
    if conversation and previous_response_id:
        raise ValueError("Mutually exclusive parameters")

    # 2. Process input with previous context
    all_input, messages, tool_context = await self._process_input_with_previous_response(
        input, tools, previous_response_id, conversation
    )

    # 3. Prepend instructions
    if instructions:
        messages.insert(0, OpenAISystemMessageParam(content=instructions))

    # 4. Prepend reusable prompt
    await self._prepend_prompt(messages, prompt)

    # 5. Build context
    ctx = ChatCompletionContext(
        model=model,
        messages=messages,
        response_tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        response_format=response_format,
        tool_context=tool_context,
        inputs=all_input,
    )

    # 6. Create orchestrator
    orchestrator = StreamingResponseOrchestrator(...)

    # 7. Stream or accumulate
    if stream:
        return orchestrator.create_response()
    else:
        # Accumulate all events, return final
        final_response = None
        async for event in orchestrator.create_response():
            if event.type in {"response.completed", "response.incomplete"}:
                final_response = event.response

        if final_response is None:
            raise ValueError("Stream never reached terminal state")

        return final_response
```

### Tool Context Recovery

**File:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/types.py`

```python
class ToolContext:
    """Manages tool state across response continuations."""

    def __init__(self, current_tools: list[OpenAIResponseInputTool] | None):
        self.tools_to_process: list[OpenAIResponseInputTool] = []
        self.previous_tools: dict[str, OpenAIResponseInputToolMCP] = {}
        self.previous_tool_listings: list[OpenAIResponseOutputMessageMCPListTools] = []

        # Separate current tools from previous listings
        if current_tools:
            for tool in current_tools:
                if isinstance(tool, OpenAIResponseOutputMessageMCPListTools):
                    self.previous_tool_listings.append(tool)
                else:
                    self.tools_to_process.append(tool)

    def recover_tools_from_previous_response(
        self,
        previous_response: OpenAIResponseObjectWithInput
    ):
        """Extract MCP tool mappings from previous response output."""
        for output_item in previous_response.output:
            if isinstance(output_item, OpenAIResponseOutputMessageMCPListTools):
                self.previous_tool_listings.append(output_item)
            elif isinstance(output_item, OpenAIResponseOutputMessageMCPCall):
                # Reconstruct MCP server info from call
                if output_item.server_label not in self.previous_tools:
                    self.previous_tools[output_item.name] = OpenAIResponseInputToolMCP(
                        server_label=output_item.server_label,
                        server_url="",  # Not stored in output
                    )
```

This enables MCP tool calls to work across response continuations without re-listing tools.

### Message Conversion

**File:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/utils.py`

```python
async def convert_response_input_to_chat_messages(
    input: str | list[OpenAIResponseInput],
    previous_messages: list[OpenAIMessageParam] | None = None,
    files_api: Files | None = None,
) -> list[OpenAIMessageParam]:
    """Convert response input format to chat completion format.

    Response input can include:
    - OpenAIResponseMessage (user/assistant messages)
    - OpenAIResponseInputFunctionToolCallOutput (function tool results)
    - OpenAIResponseMCPApprovalRequest/Response (MCP approval flow)
    - OpenAIResponseOutputMessageMCPListTools (MCP tool listings)
    - Other tool call types

    Chat completion only accepts:
    - OpenAIUserMessageParam
    - OpenAIAssistantMessageParam
    - OpenAISystemMessageParam
    - OpenAIToolMessageParam
    """

    messages = list(previous_messages) if previous_messages else []

    if isinstance(input, str):
        messages.append(OpenAIUserMessageParam(content=input))
        return messages

    for item in input:
        if isinstance(item, OpenAIResponseMessage):
            # Convert response message to chat message
            # Handle role conversion, content format, etc.
            ...
        elif isinstance(item, OpenAIResponseInputFunctionToolCallOutput):
            # Convert to OpenAIToolMessageParam
            messages.append(OpenAIToolMessageParam(
                tool_call_id=item.call_id,
                content=item.output
            ))
        elif isinstance(item, OpenAIResponseMCPApprovalResponse):
            # Store in context for approval check
            ctx.store_approval_response(item)
        # ... other conversions

    return messages
```

### Guardrail Integration

**File:** `/src/llama_stack/providers/inline/agents/meta_reference/responses/utils.py`

```python
async def run_guardrails(
    safety_api: Safety | None,
    text: str,
    guardrail_ids: list[str]
) -> str | None:
    """Run safety guardrails on text.

    Returns:
        Violation message if any guardrail failed, None otherwise
    """
    if not safety_api or not guardrail_ids:
        return None

    # Run all guardrails
    guardrail_results = await safety_api.run_shields(
        shield_ids=guardrail_ids,
        messages=[OpenAIUserMessageParam(content=text)]
    )

    # Check for violations
    for result in guardrail_results:
        if result.violation:
            return result.violation_message or "Content policy violation"

    return None
```

Guardrails are checked at two points:
1. **Input guardrails** (before first inference call)
2. **Output guardrails** (after each streaming chunk)

When a violation is detected, the agent emits a refusal response and stops processing.

---

## Best Practices

### State Management

**Use `conversation` for multi-session applications:**
```python
# Create conversation per user session
conversation = await conversations_api.create_conversation(
    CreateConversationRequest(metadata={"user_id": user_id})
)

# All responses in session use same conversation
for user_message in session_messages:
    response = await agents_api.create_openai_response(
        input=user_message,
        model="llama3-70b",
        conversation=conversation.id
    )
```

**Use `previous_response_id` for linear workflows:**
```python
# Step 1: Initial query
response1 = await agents_api.create_openai_response(
    input="Analyze the sales data",
    model="llama3-70b"
)

# Step 2: Follow-up
response2 = await agents_api.create_openai_response(
    input="What were the top 3 products?",
    model="llama3-70b",
    previous_response_id=response1.id
)
```

**Disable storage for temporary queries:**
```python
# One-off query (don't store)
response = await agents_api.create_openai_response(
    input="Quick test",
    model="llama3-70b",
    store=False  # Won't be persisted
)
```

### Tool Configuration

**Limit tool calls to prevent runaway loops:**
```python
response = await agents_api.create_openai_response(
    input="Search for information and summarize",
    model="llama3-70b",
    tools=[{"type": "web_search"}],
    max_tool_calls=5,  # At most 5 web searches
    max_infer_iters=10  # At most 10 inference calls
)
```

**Use specific tool_choice for reliability:**
```python
# Force specific tool
response = await agents_api.create_openai_response(
    input="Find weather in Tokyo",
    model="llama3-70b",
    tools=[{"type": "web_search"}, {"type": "function", "name": "get_weather"}],
    tool_choice={"type": "web_search"}  # Must use web_search
)
```

**Restrict MCP tools with allowed_tools:**
```python
{
    "type": "mcp",
    "server_url": "http://localhost:8000",
    "server_label": "github",
    "allowed_tools": ["create_issue", "list_repos"],  # Only these tools
    "require_approval": {"always": ["create_issue"]}  # Require approval for creates
}
```

### Streaming

**Always handle all event types:**
```python
async for event in response_stream:
    match event.type:
        case "response.completed" | "response.incomplete":
            # Terminal events
            handle_completion(event.response)
        case "response.failed":
            # Error handling
            handle_error(event.response.error)
        case "response.output_text.delta":
            # Incremental text
            display_text(event.delta)
        case _:
            # Log unknown events for debugging
            logger.debug(f"Unhandled event: {event.type}")
```

**Use sequence numbers for debugging:**
```python
last_sequence = -1
async for event in response_stream:
    if hasattr(event, "sequence_number"):
        if event.sequence_number <= last_sequence:
            logger.warning(f"Out of order event: {event.sequence_number}")
        last_sequence = event.sequence_number
```

### Error Handling

**Check response status:**
```python
response = await agents_api.create_openai_response(
    input="...",
    model="llama3-70b",
    stream=False
)

match response.status:
    case "completed":
        # Success
        process_output(response.output)
    case "incomplete":
        # Hit limits (max_infer_iters or token limit)
        logger.warning("Response incomplete, consider increasing max_infer_iters")
        process_output(response.output)  # Partial results
    case "failed":
        # Error occurred
        logger.error(f"Response failed: {response.error.message}")
        raise RuntimeError(response.error.message)
```

**Handle streaming errors:**
```python
try:
    async for event in response_stream:
        if event.type == "response.failed":
            raise RuntimeError(event.response.error.message)
        # ... handle other events
except Exception as e:
    logger.error(f"Stream error: {e}")
    # Clean up resources
```

### Safety

**Always use guardrails for user-facing applications:**
```python
response = await agents_api.create_openai_response(
    input=user_input,
    model="llama3-70b",
    guardrails=["llama_guard"]  # Shield ID from safety_api
)
# If violation detected, response will contain refusal content
if response.output[0].type == "message":
    if response.output[0].content[0].type == "refusal":
        display_warning(response.output[0].content[0].refusal)
```

**Validate conversation IDs:**
```python
try:
    response = await agents_api.create_openai_response(
        input="Hello",
        conversation="invalid_id",  # Missing "conv_" prefix
        model="llama3-70b"
    )
except InvalidConversationIdError as e:
    logger.error(f"Invalid conversation ID: {e}")
```

### Performance

**Use message caching for long conversations:**
```python
# The conversation field automatically enables message caching
# No additional configuration needed - just use conversation consistently
response = await agents_api.create_openai_response(
    input="Continue discussion",
    conversation=conversation.id,  # Messages cached automatically
    model="llama3-70b"
)
```

**Paginate response lists:**
```python
all_responses = []
after = None
while True:
    page = await agents_api.list_openai_responses(
        ListResponsesRequest(after=after, limit=100)
    )
    all_responses.extend(page.data)
    if not page.has_more:
        break
    after = page.last_id
```

**Optimize vector search with filters:**
```python
{
    "type": "file_search",
    "vector_store_ids": ["vs_abc123"],
    "filters": {
        "file_name": {"eq": "report.pdf"},
        "created_at": {"gte": "2024-01-01"}
    },
    "max_num_results": 3,  # Limit results
    "ranking_options": {
        "score_threshold": 0.7  # Filter low-quality matches
    }
}
```

---

## Troubleshooting

### Common Issues

**1. "Response with id {id} not found"**

**Cause:** Response was deleted or access control denied.

**Solution:**
```python
# Check if response was deleted
responses = await agents_api.list_openai_responses(
    ListResponsesRequest(limit=100)
)
if response_id not in [r.id for r in responses.data]:
    logger.error(f"Response {response_id} does not exist")

# Check access control policies
# If using AuthorizedSqlStore, ensure policy grants access
```

**2. "Mutually exclusive parameters: 'previous_response_id' and 'conversation'"**

**Cause:** Cannot use both state management approaches simultaneously.

**Solution:**
```python
# Choose one approach:
# Option 1: Linear chain
response = await agents_api.create_openai_response(
    input="Follow up",
    previous_response_id="resp_abc123",
    model="llama3-70b"
)

# Option 2: Conversation session
response = await agents_api.create_openai_response(
    input="Follow up",
    conversation="conv_xyz789",
    model="llama3-70b"
)
```

**3. "Response stream never reached a terminal state"**

**Cause:** Stream was interrupted or no terminal event emitted.

**Solution:**
```python
# Add timeout
import asyncio

try:
    response = await asyncio.wait_for(
        agents_api.create_openai_response(..., stream=False),
        timeout=60.0  # 60 second timeout
    )
except asyncio.TimeoutError:
    logger.error("Response timed out")
```

**4. "Invalid {max_tool_calls}=0; should be >= 1"**

**Cause:** max_tool_calls must be at least 1 if specified.

**Solution:**
```python
# Option 1: Remove constraint (allow unlimited)
response = await agents_api.create_openai_response(
    input="...",
    max_tool_calls=None,  # No limit
    model="llama3-70b"
)

# Option 2: Set reasonable limit
response = await agents_api.create_openai_response(
    input="...",
    max_tool_calls=10,  # At most 10 tool calls
    model="llama3-70b"
)
```

**5. "Tool {tool_name} not found in chat tools"**

**Cause:** tool_choice references a tool not in the tools list.

**Solution:**
```python
# Ensure tool is in tools list
tools = [
    {"type": "web_search"},
    {"type": "function", "name": "get_weather"}
]
response = await agents_api.create_openai_response(
    input="...",
    tools=tools,
    tool_choice={"type": "function", "name": "get_weather"},  # Must be in tools
    model="llama3-70b"
)
```

**6. Response status is "incomplete"**

**Cause:** Hit max_infer_iters or token limit.

**Solution:**
```python
# Option 1: Increase max_infer_iters
response = await agents_api.create_openai_response(
    input="Complex task requiring many steps",
    max_infer_iters=20,  # Increase from default 10
    model="llama3-70b"
)

# Option 2: Continue from incomplete response
if response.status == "incomplete":
    # Continue the work
    response2 = await agents_api.create_openai_response(
        input="Continue",
        previous_response_id=response.id,
        max_infer_iters=20,
        model="llama3-70b"
    )
```

**7. "Conversation {id} not found"**

**Cause:** Conversation doesn't exist or was deleted.

**Solution:**
```python
# Always create conversation before use
try:
    conversation = await conversations_api.get_conversation(
        GetConversationRequest(conversation_id="conv_abc123")
    )
except ValueError:
    # Conversation doesn't exist - create it
    conversation = await conversations_api.create_conversation(
        CreateConversationRequest(metadata={})
    )

# Or handle missing conversation gracefully
try:
    response = await agents_api.create_openai_response(
        input="...",
        conversation="conv_abc123",
        model="llama3-70b"
    )
except ConversationNotFoundError:
    # Fall back to stateless mode
    response = await agents_api.create_openai_response(
        input="...",
        model="llama3-70b"
    )
```

### Debugging Tips

**Enable debug logging:**
```python
import logging
logging.getLogger("agents::meta_reference").setLevel(logging.DEBUG)
logging.getLogger("openai_responses").setLevel(logging.DEBUG)
```

**Inspect stored responses:**
```python
# Retrieve full response with input
response_with_input = await responses_store.get_response_object("resp_abc123")
print(f"Input: {response_with_input.input}")
print(f"Output: {response_with_input.output}")
print(f"Messages: {response_with_input.messages}")  # Chat completion format
```

**Trace streaming events:**
```python
event_log = []
async for event in response_stream:
    event_log.append({
        "type": event.type,
        "sequence": getattr(event, "sequence_number", None),
        "timestamp": time.time()
    })
    # ... handle event

# Analyze event log
print(json.dumps(event_log, indent=2))
```

**Check tool execution:**
```python
# Log all tool calls
async for event in response_stream:
    if event.type == "response.output_item.added":
        if event.item.type in ["web_search_call", "file_search_call", "mcp_call"]:
            logger.info(f"Tool call started: {event.item.type} - {event.item.id}")
    elif event.type == "response.output_item.done":
        if event.item.type == "mcp_call":
            if event.item.error:
                logger.error(f"MCP call failed: {event.item.error}")
            else:
                logger.info(f"MCP call succeeded: {event.item.output}")
```

---

## Appendix: File Reference

### Core API Files
- `/src/llama_stack_api/agents/api.py` - Protocol definition
- `/src/llama_stack_api/agents/models.py` - Request/response models
- `/src/llama_stack_api/openai_responses/models.py` - OpenAI-compatible models

### Implementation Files
- `/src/llama_stack/providers/inline/agents/meta_reference/agents.py` - Main implementation
- `/src/llama_stack/providers/inline/agents/meta_reference/responses/openai_responses.py` - Core logic
- `/src/llama_stack/providers/inline/agents/meta_reference/responses/streaming.py` - Streaming orchestrator
- `/src/llama_stack/providers/inline/agents/meta_reference/responses/tool_executor.py` - Tool execution
- `/src/llama_stack/providers/inline/agents/meta_reference/responses/types.py` - Context types
- `/src/llama_stack/providers/inline/agents/meta_reference/responses/utils.py` - Helper functions

### Storage Files
- `/src/llama_stack/providers/utils/responses/responses_store.py` - Response persistence
- `/src/llama_stack/core/conversations/conversations.py` - Conversation service
- `/src/llama_stack/core/storage/sqlstore/authorized_sqlstore.py` - Access control layer

### Test Files
- `/tests/integration/agents/test_openai_responses.py` - Integration tests
- `/tests/unit/providers/agents/meta_reference/test_openai_responses.py` - Unit tests
- `/tests/unit/providers/agents/meta_reference/test_openai_responses_conversations.py` - Conversation tests

---

## Glossary

**Agent:** A stateful AI system that orchestrates multi-turn interactions between LLMs, tools, and external services.

**Response:** The fundamental unit of agent interaction, containing user input, model output, and execution history.

**Agentic Loop:** The iterative process where the model reasons, calls tools, and refines its response.

**Tool Call:** A request from the model to execute a tool (function, web search, file search, MCP server).

**Function Tool:** A client-side tool that requires the client to execute and return results.

**Built-in Tool:** A server-side tool that the agent executes automatically (web search, file search).

**MCP Tool:** A tool provided by a Model Context Protocol server.

**Conversation:** A session grouping multiple related responses with shared metadata.

**previous_response_id:** A reference to a previous response for linear conversation chaining.

**ResponsesStore:** SQL storage for complete response objects with input/output history.

**Conversations API:** Service for managing conversation sessions and items.

**Message Cache:** Optimized storage of chat completion messages for efficient continuation.

**Streaming:** Real-time event emission during response generation.

**Guardrails:** Safety checks applied to input/output text to detect policy violations.

**max_infer_iters:** Maximum number of chat completion calls per response.

**max_tool_calls:** Maximum number of built-in tool calls per response.

**tool_choice:** Strategy for selecting which tool(s) the model should call.

**parallel_tool_calls:** Whether to allow multiple tool calls in a single turn.

---

This documentation is maintained by the Llama Stack team. For questions or issues, please visit:
- GitHub: https://github.com/meta-llama/llama-stack
- Discord: https://discord.gg/llama-stack
