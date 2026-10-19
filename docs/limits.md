# Rate limits

Hosted LLM providers meter each account — requests per minute, tokens per minute, tokens
per day — and answer an over-quota request with HTTP 429. kuhaku's providers retry a 429,
but they only find the quota by hitting it, and concurrent callers keep hitting it
together. On a free tier that turns a few hundred calls into hours of retries.

`kuhaku.core.llm.limits.RateLimitedLLM` wraps any `LLMProvider` without changing it. It
paces calls to the account's quota *before* they are sent, and makes every caller back
off together when a 429 does get through. It admits a call on a pre-call estimate and
corrects it with the token usage the provider reports once the call returns. It is off
unless you turn it on.

## Rate limits from the environment

Setting any of these makes `build_llm_provider` — and therefore `RAG()` and
`Extractor()` when they build their own provider — wrap the provider in a rate limiter:

```bash
export KUHAKU_LLM_RATE_LIMIT_REQUESTS_PER_MINUTE=30
export KUHAKU_LLM_RATE_LIMIT_TOKENS_PER_MINUTE=8000
export KUHAKU_LLM_RATE_LIMIT_TOKENS_PER_DAY=200000
```

One limiter is shared per provider name (`groq:llama-3.3-70b-versatile`), so everything in
the process that talks to that model spends one quota. It is per model, not per provider,
because that is how Groq and OpenAI meter. The output side of each call is reserved at
`KUHAKU_LLM_MAX_TOKENS` and credited back once the provider reports what was really used.

A process does not know what an earlier process already spent: the daily bucket starts
full on every start.

## Rate limits in code

```python
from kuhaku.core.llm.limits import RateLimitedLLM, RateLimiter, RateLimits


class EchoLLM:
    """Stand-in for a real provider."""

    name = "groq:llama-3.3-70b-versatile"

    def generate(self, system: str, user: str) -> str:
        return f"echo: {user}"


limiter = RateLimiter(RateLimits(requests_per_minute=30, tokens_per_minute=8_000))
llm = RateLimitedLLM(EchoLLM(), limiter)
print(llm.generate("You are terse.", "hello"))
```

Share the `RateLimiter` object between every wrapper that spends the same account's quota
— threads included; it is thread-safe. `RateLimiter.for_provider("groq")` uses kuhaku's
built-in default (30 requests and 6,000 tokens per minute, the lowest Groq free-tier
values); kuhaku has no default for OpenAI, Anthropic or Vertex because their quotas depend
on your account tier. `max_wait_seconds=` makes a call that would wait longer raise
`RateLimitTimeoutError` instead.

**Estimates.** By default a call is estimated at one token per 3.5 characters of prompt
plus 256 output tokens (`CharTokenEstimator`). Any callable
`(system, user) -> TokenEstimate` can replace it, e.g. a real tokenizer:

```python
from kuhaku.core.llm.limits import CharTokenEstimator, TokenEstimate

estimator = CharTokenEstimator(chars_per_token=4, output_tokens=1024)
print(estimator("system prompt", "user prompt"))


def my_estimator(system: str, user: str) -> TokenEstimate:
    return TokenEstimate(input_tokens=len((system + user).split()) * 2, output_tokens=512)
```

**429s.** When the provider still rejects a call, the `Retry-After` header (or Groq's and
OpenAI's `x-ratelimit-reset-tokens`) pauses every caller of that limiter, and the call is
retried once (`rate_limit_retries=`). With no hint the pause is `default_backoff_seconds`
(5). These retries come on top of the provider's own retry attempts, which do not read
`Retry-After`; a 429 also counts toward the provider's circuit breaker.

## Stacking

`RateLimitedLLM` is an `LLMWrapper`: it has the provider's own `generate()`/`name` shape
and passes `last_usage` through, so it composes with `TokenTrackingLLM` in any order.
`chat`, `generate_structured` and `chat_with_tools` are paced like `generate`, estimated
from the messages' text plus the schema or tool definitions (images and documents are not
counted up front) and reconciled with the response's own usage.

## What gets recorded

| Metric | Type | Labels |
|---|---|---|
| `kuhaku_llm_rate_limit_wait_seconds` | histogram | `gen_ai.system` |
| `kuhaku_llm_rate_limited_total` | counter | `gen_ai.system` |

The ambient span (the RAG engine's "generate" span, when there is one) gets
`kuhaku.llm.rate_limit.wait_seconds` and `kuhaku.llm.rate_limit.retries`.

## Limits of the approach

- Usage is read from the provider's `last_usage` attribute, a single slot. A provider
  object shared by several threads can report one call's usage to another; the totals
  stay right, individual reconciliations may not.
- There are no async variants. `RateLimiter.try_acquire()` never blocks — it returns
  a permit or the seconds to wait — so an async caller can await `asyncio.sleep` on it.
