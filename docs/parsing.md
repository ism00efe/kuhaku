# Document parsing

`kuhaku.tools.parsing` turns a file — PDF, DOCX, HTML, Markdown, plain text or an image —
into one structured document: pages of headings, paragraphs, lists, tables, code blocks
and figure placeholders, each with its provenance (source, page, and a bounding box when
the format has one). No LLM is involved.

It is a standalone tool. It does not import the RAG or extraction tools, and they do not
import it: you parse, then hand the text to whichever tool you need.

## Install

HTML, Markdown and plain text need nothing beyond kuhaku itself. The heavier formats are
optional extras, imported only when a file of that type is parsed:

| Extra | Installs | Enables |
|---|---|---|
| `kuhaku[parsing-pdf]` | pdfplumber | PDF headings, tables and bounding boxes |
| `kuhaku[parsing-docx]` | python-docx | DOCX |
| `kuhaku[parsing-ocr]` | pytesseract, Pillow | images, and scanned PDF pages |
| `kuhaku[parsing]` | all of the above | |

Without `parsing-pdf`, PDFs are still read — as plain text per page through pypdf, which
kuhaku already depends on — and the result carries a `layout_unavailable` warning.
`parsing-ocr` also needs the [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
binary on `PATH`. A missing extra raises `MissingDependency`, whose message names the
`pip install` command.

## Parse a file

```python
from kuhaku.tools.parsing import DocumentParser

markdown = b"""# Price list

Valid from March.

| Product | Price |
|---|---|
| Tea | 4 |
"""

doc = DocumentParser().parse(markdown, filename="prices.md")

print(doc.title)              # "Price list"
print(doc.tables[0].rows)     # (("Product", "Price"), ("Tea", "4"))
print(doc.to_markdown())      # headings as #, tables as pipe tables
print(doc.to_text())          # the same content as plain text
print(doc.warnings)           # () -- nothing went wrong
```

`parse` takes a path (`str` or `Path`) or raw `bytes`. A `str` is always a path; to
parse text you already hold, pass `text.encode()` with a `filename` whose suffix names
the format. The parser is chosen by an explicit `media_type=`, then the file suffix,
then the content's magic bytes; anything else raises `UnsupportedFormat`.

## The document model

```python
from kuhaku.tools.parsing import DocumentParser, Heading, Table

html = b"<h1>Report</h1><p>Intro.</p><table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
doc = DocumentParser().parse(html, filename="report.html")

for page in doc.pages:
    for block in page.blocks:
        where = block.provenance            # .source, .page, .bbox, .ocr
        if isinstance(block, Heading):
            print("heading", block.level, block.text)
        elif isinstance(block, Table):
            print("table", block.header, block.rows[1:])
        else:
            print(block.kind.value, block.text)
```

- `ParsedDocument` — `pages`, `warnings`, `metadata` (title/author when the format has
  them), `media_type`, `trace_id`; `blocks`, `tables`, `title`, `ocr_pages`;
  `to_text()` and `to_markdown(page_breaks=False)`.
- Blocks — `Heading` (`level` 1–6), `Paragraph`, `ListBlock` (`items` of `ListItem`
  with a nesting `level`, `ordered`), `Table` (`rows` of cell strings, `has_header`),
  `CodeBlock`, `Figure` (`alt`, `src`; the pixels are not kept).
- `Provenance.bbox` is set for PDFs read with pdfplumber, in PDF points with a top-left
  origin. DOCX, HTML, Markdown and text have no fixed pages: everything is on page 1,
  with no bounding box.

`to_markdown(page_breaks=True)` puts `<!-- page N -->` before each page, so page numbers
survive into a consumer that only sees text.

## Scanned PDFs and images

A PDF page with no text layer is treated as scanned. When an OCR engine is available,
the images embedded in that page are recognized and the page's blocks are marked
`provenance.ocr=True`; `doc.ocr_pages` lists those pages. When no engine is available,
the page is left empty **and** a `no_text_layer` warning names it — a scanned page never
comes back as silently empty text. An image file without an engine parses to a single
`Figure` with an `ocr_unavailable` warning.

`DocumentParser(ocr="auto")` (the default) uses Tesseract only when the `parsing-ocr`
extra and the binary are both present. Pass `ocr=None` to never OCR, or any object with
a `name` and a `recognize(image: bytes) -> str` method to use another engine:

```python
from kuhaku.tools.parsing import DocumentParser


class MyOCR:
    name = "my-ocr"

    def recognize(self, image: bytes) -> str:
        return "text recognized by my service"


parser = DocumentParser(ocr=MyOCR())
```

OCR works on the images embedded in a page, not on a rendering of it, so text drawn as
vector outlines is not recovered. OCR'd blocks carry no bounding box.

## Warnings

`doc.warnings` is a tuple of `ParseWarning(code, message, page)`. The codes are:

| Code | Meaning |
|---|---|
| `no_text_layer` | a PDF page had no text and could not be OCR'd |
| `ocr_unavailable` | an image was parsed without an OCR engine |
| `ocr_failed` | the OCR engine raised on one image; the rest of the page was kept |
| `ocr_empty` | OCR ran but found no text |
| `layout_unavailable` | a PDF was read with pypdf (text only) because pdfplumber is not installed |
| `decode_errors` | text was not valid UTF-8; bad bytes became U+FFFD |

A parse that produced warnings also logs one `WARNING` line listing their codes.

## Feeding RAG

`RAG.ingest` takes text. Pass it the Markdown rendering: headings and pipe tables are
what `RAG(chunking="structural")` splits on, so tables become their own
chunks instead of being packed into prose. Give a `.md` filename so the stored source
reads as Markdown.

```python
from pathlib import Path

from kuhaku import RAG
from kuhaku.tools.parsing import DocumentParser

doc = DocumentParser().parse(
    b"<h1>Expenses</h1><p>Claims are paid monthly.</p>", filename="expenses.html"
)

rag = RAG(chunking="structural")
chunks_added, redactions = rag.ingest(
    doc.to_markdown(), filename=Path(doc.source).with_suffix(".md").name
)
```

`RAG.ingest` sanitizes and guards the text as it does for any other input; the parsing
tool itself does neither. `RAG.load_documents` still reads only `.txt`, `.md` and `.pdf`
(as plain text) — to index DOCX, HTML or scanned files, parse them first and call
`ingest` per document.

## Feeding structured extraction

`Extractor.extract` also takes text. The Markdown rendering keeps table rows on one line
each, which is what a model reads most reliably. This example needs a configured LLM
provider (see [providers](providers.md)), so it is not run as part of the docs test:

<!-- no-exec -->
```python
from pydantic import BaseModel

from kuhaku.tools.extraction import Extractor
from kuhaku.tools.parsing import DocumentParser


class Invoice(BaseModel):
    vendor: str
    total: float


doc = DocumentParser().parse("invoice.pdf")
if doc.warnings:
    print("parsed with warnings:", [w.code for w in doc.warnings])

result = Extractor(Invoice).extract(doc.to_markdown())
print(result.status, result.value)
```

## Custom parsers

A parser is any object with `name`, `media_types`, `suffixes` and
`parse(source: DocumentSource) -> ParsedDocument`. Register it on a registry; a type
that is already taken is replaced, which is also how a built-in parser is swapped out.

```python
from kuhaku.tools.parsing import (
    DocumentParser,
    DocumentSource,
    Page,
    Paragraph,
    ParsedDocument,
    Provenance,
    default_registry,
)


class CsvAsTextParser:
    name = "csv-text"
    media_types = ("text/csv",)
    suffixes = (".csv",)

    def parse(self, source: DocumentSource) -> ParsedDocument:
        text = source.data.decode("utf-8")
        block = Paragraph(provenance=Provenance(source.name, 1), content=text)
        return ParsedDocument(
            source=source.name, media_type="text/csv", pages=(Page(number=1, blocks=(block,)),)
        )


registry = default_registry(ocr=None)
registry.register(CsvAsTextParser())
doc = DocumentParser(registry=registry).parse(b"a,b\n1,2", filename="data.csv")
```

The PDF parser's backend can be pinned the same way:
`registry.register(PdfParser(backend="pypdf"))` (or `"pdfplumber"`, which then raises
`MissingDependency` when the extra is absent instead of falling back).

## Observability

Each `parse` call is one `parsing` stage: an OpenTelemetry span, a structured log record
and a `STAGE_DURATION` sample (see [observability](observability.md)). The record
carries `parser`, `media_type`, `size_bytes`, `page_count`, `block_count`,
`table_count`, `ocr_page_count` and `warning_count`. The file name is not recorded —
it is caller-supplied and may itself be sensitive. `parse(..., trace_id=...)` binds an
id the same way `Extractor.extract` does, and it is returned on `doc.trace_id`.

## Limits

- Headings in PDFs are recognized by font size relative to the body text; bold-only
  headings stay paragraphs. pdfplumber finds ruled tables most reliably.
- A table's first row is assumed to be its header for PDF and DOCX tables.
- DOCX numbered vs. bulleted lists are told apart by style name ("List Number").
- No size limit is applied; check the size of untrusted uploads before parsing.
