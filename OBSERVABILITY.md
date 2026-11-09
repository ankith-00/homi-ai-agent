# Agent observability

The agent emits OpenTelemetry traces to Pydantic Logfire when `LOGFIRE_TOKEN` is set. Without a token, tracing is disabled and chat requests continue normally.

Set these environment variables in the agent deployment:

```text
LOGFIRE_TOKEN=<your Logfire write token>
ENVIRONMENT=production
LOGFIRE_CAPTURE_CHAT_CONTENT=false
```

Each chat turn has a `homi.chat.turn` span with hashed user/session identifiers, input and output lengths, preparation duration, time to first text, tool count, and status. Child spans measure MongoDB history/profile reads, Mem0 search, and MongoDB history saves. Pydantic AI adds model request and tool execution spans.

Prompt and final response text are omitted by default. Set `LOGFIRE_CAPTURE_CHAT_CONTENT=true` only when you need to inspect that content; captured text is capped at 4,000 characters and common credential values and auth URLs are redacted. Pydantic AI prompt/completion content, tool arguments, and tool return bodies remain excluded from its automatic spans.

The agent's private internal reasoning is not recorded. Use the model request and tool spans to inspect the observable trajectory: model requests, selected tools, tool execution, and final answer metadata.
