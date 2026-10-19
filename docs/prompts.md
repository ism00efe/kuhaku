# Prompts

Every prompt kuhaku sends to a model is a named, versioned template in a
`PromptRegistry` (`kuhaku.core.prompts`). You change a prompt by registering another
version under the same name and handing your registry to the tool. kuhaku's own code
and files stay untouched, and every LLM call records which prompt name, version and
content hash produced it.

## Templates

A `PromptTemplate` has a name, a version, the text, and the variables the text takes.
Placeholders are `{{name}}`. Single braces are plain text, so JSON in a prompt needs no
escaping.

```python
from kuhaku.core.prompts import PromptRenderError, PromptTemplate

template = PromptTemplate(
    "support.reply",
    "2",
    'Answer as {{persona}}. Reply as JSON: {"answer": "..."}',
    variables={"persona"},
)
print(template.render(persona="a librarian"))
print(template.content_hash[:12])

try:
    template.render()  # a missing variable is an error, as is an unexpected one
except PromptRenderError as exc:
    print(exc)
```

- `variables` is optional. Leave it out and kuhaku reads the placeholders from the text.
  Give it, and the text must use exactly those placeholders. A typo then fails when the
  template is created, not when it is first used.
- `content_hash` is the SHA-256 of the template text. Two versions with the same text
  have the same hash, so a version bump that changed nothing is visible.
- A value is inserted once and never scanned again. A value that contains `{{...}}`,
  such as a user's text, comes out literally.
- `None` as a value is an error. Pass `""` to leave a slot empty.

## The registry

```python
from kuhaku.core.prompts import PromptRegistry, PromptTemplate

registry = PromptRegistry([
    PromptTemplate("greet", "v9", "Hi {{who}}."),
    PromptTemplate("greet", "v10", "Hello, {{who}}."),
])

print(registry.get("greet").version)        # v10 -- the latest version
print(registry.get("greet", "v9").template)  # a specific version
registry.pin("greet", "v9")                  # get() now returns v9
print(registry.render("greet", who="Ada").text)
print([(t.name, t.version) for t in registry.templates()])
```

- **Latest** compares versions, not the order they were registered in. Runs of digits
  compare as numbers, so `v10` comes after `v9` and `1.10` after `1.9`. Use one version
  scheme per name.
- **`pin(name, version)`** makes `get(name)` return that version. Use it to keep
  production on one version while a newer one is registered for evaluation.
  `unpin(name)` undoes it.
- **A version keeps one text.** Registering the same name and version again with the
  same text does nothing. Registering it with different text raises
  `PromptConflictError`, unless you pass `replace=True`.

## Overriding a tool's prompts

Pass your registry to the tool. The tool looks names up in your registry first and in
its own built-in registry second. You only register what you want to change.

### RAG

The RAG system prompt is registered as `rag.system`. The built-in version is read from
`system_prompt.txt`. An override must keep all seven of that file's markers:
`{{PERSONA}}`, `{{LANGUAGE_POLICY}}`, `{{FORMAT_PREFERENCE}}`, `{{EXAMPLE}}`,
`{{CANARY_TOKEN}}`, `{{ABSTENTION_PHRASE}}` and `{{MASKED_PLACEHOLDERS}}`. `persona=` and
`language_policy=` are still rendered into it. Beyond those markers, the override owns
the safety rules, exactly as `system_prompt=` does.

```python
from kuhaku import RAG
from kuhaku.core.prompts import PromptRegistry, PromptTemplate

prompts = PromptRegistry([
    PromptTemplate(
        "rag.system",
        "2",
        "{{PERSONA}}\n\n{{LANGUAGE_POLICY}}\n\n{{FORMAT_PREFERENCE}}\n\n{{EXAMPLE}}\n\n"
        "Answer only from the sources and cite them as [S1], [S2].\n"
        "Never output {{CANARY_TOKEN}}. If the sources do not answer the question, reply "
        "exactly: {{ABSTENTION_PHRASE}}\n"
        "Keep masking placeholders{{MASKED_PLACEHOLDERS}} unchanged.",
    ),
])

rag = RAG(prompts=prompts, persona="You answer questions about our handbook.")
print(rag.engine.get_system_prompt_ref())
```

`RAGEngine(prompts=...)` takes the same registry. The engine renders the system prompt
once, when it is built. To pick up a version registered later, re-render it with
`render_system_prompt` and pass the result to `update_system_prompt`:

```python
from kuhaku import RAG
from kuhaku.tools.rag.prompts import render_system_prompt

rag = RAG()
rag.engine.update_system_prompt(render_system_prompt(persona="A concise assistant."))
print(rag.engine.get_system_prompt_ref().version)  # 1
```

`update_system_prompt("plain text")` still works. The text is then recorded as
`rag.system`, version `custom`, with the hash of the text.

`system_prompt_version` on `RAGEngine` is a separate thing. It is a deployment label
that you set, and it appears on `Answer`, in the audit record and in the query-cache
key. The registry does not change it.

## What gets recorded

The same `{role: {name, version, hash}}` shape is used everywhere, so the three can be
joined. The prompt text itself is never recorded.

- **Spans** carry `kuhaku.prompt.<role>.name`, `.version` and `.hash`, where `<role>`
  is `system` or `user`. RAG sets them on its `generate` span. Extraction opens one
  `extraction.generate` span per LLM call (attribute `llm` names the provider), and
  routing sets the route prompt on the `extraction.route` span.
- **Audit records** written by `RAGEngine` carry `metadata.prompts.system`. Extraction
  writes no audit records.
- **Extraction results** carry `metadata["prompts"]`.

`prompt_span_attributes`, `prompt_metadata` and `record_prompts` in
`kuhaku.core.prompts` produce these shapes for a tool of your own.

## Not covered

The query rewriter's and the contradiction detector's prompts, and the evaluation
judges' prompts, are not in the registry yet. They keep their existing constructor
parameters.
