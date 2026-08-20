"""Generic, tool-agnostic runtime infrastructure -- kuhaku's core layer.

No submodule here knows anything about RAG or any other specific tool (see
``kuhaku.tools.rag`` for that); this package is the boundary a tool is allowed to
depend on, never the other way around.

Submodules:

- ``config``: typed, environment-driven ``Settings`` (:func:`get_settings`,
  :func:`load_settings`) and ``configure_logging``.
- ``llm``: the ``LLMProvider`` interface and its Ollama/Anthropic/OpenAI/Vertex
  implementations, selected via :func:`~kuhaku.core.llm.build_llm_provider`.
- ``auth``: authentication context and authorization policy primitives.
- ``security``: the prompt-injection guard pipeline, output-side checks, and audit
  logging.
- ``observability``: structured logging, OpenTelemetry tracing, and OpenTelemetry
  metrics (``instrumented_step`` ties all three together for one pipeline stage).
- ``sanitization``: deterministic, regex-based PII masking.
- ``policy``: startup-time fallback/strictness policy for optional components.
- ``retry``: retry-with-backoff and circuit-breaker helpers for transient failures.
- ``exceptions``: shared exception types.
- ``models``: generic domain primitives (``Message``, ``ToolCall``,
  ``ExecutionResult``) any tool can use.

Nothing is eagerly imported here -- importing ``kuhaku.core`` alone stays cheap;
import the specific submodule you need.
"""
