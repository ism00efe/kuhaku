# Quickstart

## Install

```bash
pip install kuhaku
```

Point kuhaku at a hosted model — no local server needed:

```bash
export KUHAKU_LLM_PROVIDER=openai      # or: anthropic, vertex, ollama (the default)
export OPENAI_API_KEY=sk-...
```

Embeddings run locally whichever LLM provider you choose, so the first ingest downloads
a ~490 MB model. See the [README's installation section](https://github.com/ism00efe/kuhaku/blob/main/README.md#installation-in-detail)
before installing on a small disk.

## Ingest and ask

```python
from kuhaku import RAG

rag = RAG()
rag.ingest(
    "To reset your password, go to Settings > Security > Reset Password. "
    "A reset link is emailed to the address on file.",
    filename="handbook.md",
)

answer = rag.ask("How do I reset my password?")
print(answer.text)
```

The answer text carries its citations inline as `[S1]`-style tags. `answer.citations`
maps each tag back to its document; `answer.retrieved`, `answer.redactions` (what PII
sanitization masked), `answer.abstained` and `answer.trace_id` are on the same object.

An empty knowledge base gets its own explicit message rather than a guess:

```python
from kuhaku import RAG

rag = RAG()  # nothing ingested yet
answer = rag.ask("what is the meaning of life?")
print(answer.text)
assert answer.retrieved == []
```

Once something *is* ingested, a query that retrieval can't match to anything — or that
matches only chunks the caller isn't entitled to (see [access control](access-control.md))
— abstains instead: `answer.abstained` is `True` and `answer.retrieved` is empty. kuhaku
never lets the model improvise an answer from an empty prompt.

## Loading a directory

```python
import tempfile
from pathlib import Path

from kuhaku import RAG

with tempfile.TemporaryDirectory() as corpus_dir:
    Path(corpus_dir, "faq.md").write_text(
        "# Refunds\n\nRefunds take 1-5 business days.", encoding="utf-8"
    )
    Path(corpus_dir, "notes.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # ignored: unsupported type

    rag = RAG()
    indexed = rag.load_documents(corpus_dir)
    print(f"indexed {indexed} document(s)")
```

`load_documents` walks the top level of a directory (non-recursive) and ingests every
`.txt`, `.md` and `.pdf` file it finds, skipping anything else.

Next: [access control](access-control.md) to keep some of those documents scoped to the
right callers.
