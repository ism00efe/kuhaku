# Getting started

This page takes you from an empty environment to a working question-and-answer loop
over your own documents. Every code block on this page runs as written. For pulling
structured data out of text instead, see [Structured extraction](extraction.md).

## Install

```bash
pip install "kuhaku[rag]"
```

Requires Python 3.11 or newer. `pip install kuhaku` alone is light; the `rag` extra adds
the vector store and the embedding library, which is large — see
[Installation in detail](https://github.com/ism00efe/kuhaku#installation-in-detail) for
what it pulls in and how to avoid the CUDA packages on Linux.

## Pick where the language model runs

kuhaku talks to six LLM providers. Set the key of the one you want and kuhaku uses it;
with several set, it continues with the one you used last.

**A hosted model** — nothing to run locally:

```bash
export OPENROUTER_API_KEY=sk-or-...    # free models available
# or OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY -- or pin one:
export KUHAKU_LLM_PROVIDER=openrouter
```

**A local model** — no API key, no data leaving the machine:

```bash
ollama serve
ollama pull llama3.2                   # any Ollama chat model
```

With no API key set, a running Ollama server is picked up without any environment
variable, with the model you pulled; with several, kuhaku asks which one once.

Either way the *embedding* model runs locally. Dense retrieval needs it on disk (about
490 MB); kuhaku asks before downloading it and searches by keyword (BM25) until then. An
API key removes the language model download, not this one. See [Providers](providers.md)
for the fully hosted alternative.

## Your first answer

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")

rag.ingest(
    "To reset your password, open Settings, choose Security, and click "
    "Reset password. A confirmation email arrives within five minutes.",
    filename="handbook.md",
)

answer = rag.ask("How do I reset my password?")
print(answer.text)
```

`RAG(vector_store=...)` names a directory for the vector store. **Pass it.** Without it
kuhaku creates a fresh temporary directory for every `RAG()` instance, so anything you
ingest disappears the next time your program starts.

## Reading the answer

`ask()` returns an `Answer`. The generated text carries its citations inline as
`[S1]`-style tags, and the other fields tell you where those tags came from and what
happened along the way.

```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
answer = rag.ask("How do I reset my password?")

print(answer.text)          # the answer, with [S1] tags in the prose
print(answer.abstained)     # True when retrieval found nothing worth answering from
print(answer.trace_id)      # correlates this request's logs, metrics and audit record
print(answer.redactions)    # what PII sanitization masked, e.g. ["EMAIL×2"]

for c in answer.citations:
    print(c.tag, c.title, c.source_path, round(c.score, 3))
```

`answer.retrieved` holds the chunks the answer was grounded in, each as a
`RetrievedChunk` with a `.chunk` and a `.score`.

When retrieval runs against a non-empty store and finds nothing relevant, kuhaku sets
`abstained=True` and returns a fixed message rather than letting the model improvise an
answer. The same happens when every result falls below the confidence threshold.

An *empty* store is a different case with its own message, and it leaves
`abstained=False` — nothing was retrieved because there was nothing to retrieve, which is
a setup problem rather than a judgement about the question. Read `answer.abstained`
alongside the text when you need to tell the two apart.

## Loading a directory

<!-- no-exec -->
```python
from kuhaku import RAG

rag = RAG(vector_store="./kuhaku-data")
count = rag.load_documents("./handbook")
print(f"indexed {count} documents")
```

`load_documents` reads `.txt`, `.md` and `.pdf` files from one directory. It is not
recursive, and it applies one set of access tags to the whole call — see
[Access control](access-control.md).

## Where to go next

- [Configuration](configuration.md) — every setting that has an effect, its default, and
  the environment variable that changes it.
- [Access control](access-control.md) — restricting documents to the people entitled to
  see them.
- [Providers](providers.md) — choosing and configuring the language and embedding models.
- [Extending](extending.md) — replacing the retriever, the store, or the model with your
  own.
