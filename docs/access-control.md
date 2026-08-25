# Access control

kuhaku can restrict a document to the callers entitled to see it. The restriction is
enforced during retrieval, before ranking, so a document the caller may not see is never
retrieved, never cited, and never reaches the model.

**kuhaku does not authenticate anyone.** Your application proves who the user is and
hands kuhaku the result. kuhaku's job is enforcement.

## Tagging a document

```python
from kuhaku import RAG, AuthContext

rag = RAG(vector_store="./kuhaku-data")

rag.ingest(
    "Band 4 engineers are paid between X and Y.",
    filename="salary_policy.md",
    access_tags=["people_ops"],
)

rag.ingest(
    "Deploys run from the main branch every Tuesday.",
    filename="deploy_runbook.md",
)  # no tags: visible to everyone

answer = rag.ask(
    "what are the salary bands",
    auth_context=AuthContext(identity="ada", roles=("engineering",)),
)
print(answer.abstained)   # True — salary_policy.md was never retrieved
```

`load_documents` takes the same argument, but applies **one tag set to the whole call**:

<!-- no-exec -->
```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
rag.load_documents("./hr-policies", access_tags=["people_ops"])
```

For per-file tags, call `ingest` once per file.

## How it decides

One rule, applied to every chunk:

- An **untagged** chunk is visible to everyone. Tagging is opt-in, so an existing corpus
  keeps working unchanged.
- A **tagged** chunk is visible only when at least one of its tags also appears in the
  caller's `roles`.
- A tagged chunk with **no `auth_context`** is not retrievable. Tagging a document is
  what turns protection on for it, which is why there is no enable/disable switch to
  forget.

| Chunk | `auth_context` | Result |
|---|---|---|
| untagged | absent | visible |
| untagged | any | visible |
| tagged | absent | not retrieved |
| tagged | roles do not intersect the tags | not retrieved |
| tagged | roles intersect the tags | visible |

The comparison is a flat set intersection. There is no hierarchy, no ordering, no
wildcard, and no tag that implies another.

## Two things that will catch you

**Tags are compared as exact strings, and case matters.** A chunk tagged `["Admin"]` is
invisible to a caller with `roles=("admin",)`. Nothing normalizes either side. Pick one
casing convention and hold to it.

**Surrounding whitespace is not stripped.** A tag of `" people_ops"` passes validation
and is stored with its leading space, after which no role can ever match it — the
document becomes invisible to everyone. Strip your tags before passing them:

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")

raw_tags = [" people_ops", "finance ", "people_ops"]
tags = [t.strip() for t in raw_tags if t.strip()]
print(tags)                 # ['people_ops', 'finance', 'people_ops']

rag.ingest("Band 4 pay ranges.", filename="policy.md", access_tags=tags)
```

Only empty and whitespace-only tags are rejected outright, with a `ValueError`.

## What a blocked result looks like

Nothing. A result withheld for lack of entitlement is indistinguishable from a genuine
no-match: the same abstention message, `abstained=True`, and an empty `citations` list.

This is deliberate. Saying "you may not see this" would confirm to the caller that
matching restricted material exists, which is itself a disclosure.

```python
from kuhaku import RAG, AuthContext

rag = RAG(vector_store="./kuhaku-data")

blocked = rag.ask("salary bands", auth_context=AuthContext(identity="ada"))
absent = rag.ask("what is the airspeed of a swallow",
                 auth_context=AuthContext(identity="ada"))

print(blocked.abstained, absent.abstained)     # True True
print(blocked.text == absent.text)             # True
```

The audit record distinguishes the two cases for you — see [Security](security.md).

## AuthContext

```python
from kuhaku import AuthContext

ctx = AuthContext(
    identity="ada",
    is_authenticated=True,
    roles=("engineering", "people_ops"),
    permissions=("read:handbook",),
    metadata={"team": "platform"},
)

anonymous = AuthContext.anonymous()   # identity "", no roles
```

Only `roles` participates in access filtering. `identity`, `is_authenticated` and `roles`
are written to the audit record; `permissions` and `metadata` are carried for your own
application's use and are not interpreted by kuhaku.

`AuthContext` is immutable. Build a new one rather than mutating.

## Choosing tag names

kuhaku assigns no meaning to a tag string. `["people_ops"]`, `["level-3"]` and
`["muhasebe"]` behave identically — the vocabulary is yours. Three constants ship as a
starting point and nothing more:

```python
from kuhaku import ACCESS_TAG_PUBLIC, ACCESS_TAG_INTERNAL, ACCESS_TAG_RESTRICTED
```

They are plain strings (`"public"`, `"internal"`, `"restricted"`) with no special
handling anywhere in the library.

## Where enforcement lives

Filtering happens inside the built-in retrievers, in all three strategies:

- **Dense** pushes the filter into the vector store query, so ineligible chunks are
  excluded before similarity ranking.
- **Sparse (BM25)** narrows the candidate set before sorting and before the `top_k` cut,
  so an ineligible high-scoring chunk can never crowd out an eligible lower-scoring one.
- **Hybrid** inherits both, and fusion never sees an ineligible chunk.

The consequence matters: an entitled caller still receives a full `top_k` result set.
Implementations that rank first and drop afterwards quietly return less to the people who
were allowed to see it.

**If you supply your own retriever, you own enforcement.** The rule lives in the
retrievers, not in the engine, so a custom retriever passed to `RAGEngine` bypasses it
entirely unless you apply it yourself:

<!-- no-exec -->
```python
from kuhaku.tools.rag.retriever import is_entitled

class MyRetriever:
    def retrieve(self, query, top_k, *, auth_context=None, doc_type=None):
        candidates = my_search(query)
        return [c for c in candidates if is_entitled(c.chunk, auth_context)][:top_k]
```

See [Extending](extending.md) for the full retriever contract.

## Not in 0.1.0

`kuhaku.core.auth` also contains `AuthorizationPolicy`, `ConfigAuthorizationPolicy`,
`APIKeyAuthProvider`, `JWTAuthProvider` and `AuthProviderRegistry`. They are not wired
into retrieval or into `RAGEngine` in this release. The access control that works today
is the tag intersection described on this page.
