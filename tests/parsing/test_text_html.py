from __future__ import annotations

from kuhaku.tools.parsing import (
    CodeBlock,
    DocumentParser,
    Figure,
    Heading,
    ListBlock,
    Paragraph,
    Table,
)

MARKDOWN = """# Handbook

Intro paragraph
spanning two lines.

## Leave

- Annual leave
  continues here
- Sick leave
  - Certificate after 3 days

1. Apply
2. Wait

| Type | Days |
|------|-----:|
| Annual | 20 |
| Pipe \\| cell | 1 |

![Org chart](org.png)

```python
print("hi")
```
"""


def _parse(text: str, filename: str):
    return DocumentParser(ocr=None).parse(text.encode("utf-8"), filename=filename)


def test_markdown_blocks() -> None:
    doc = _parse(MARKDOWN, "handbook.md")
    blocks = doc.pages[0].blocks
    assert [type(b) for b in blocks] == [
        Heading, Paragraph, Heading, ListBlock, ListBlock, Table, Figure, CodeBlock,
    ]
    assert (blocks[0].text, blocks[0].level, blocks[2].level) == ("Handbook", 1, 2)
    assert blocks[1].text == "Intro paragraph\nspanning two lines."
    bullets = blocks[3]
    assert not bullets.ordered
    assert [(i.text, i.level) for i in bullets.items] == [
        ("Annual leave continues here", 0),
        ("Sick leave", 0),
        ("Certificate after 3 days", 1),
    ]
    assert blocks[4].ordered and [i.text for i in blocks[4].items] == ["Apply", "Wait"]
    assert blocks[5].rows == (("Type", "Days"), ("Annual", "20"), ("Pipe | cell", "1"))
    assert (blocks[6].alt, blocks[6].src) == ("Org chart", "org.png")
    assert (blocks[7].content, blocks[7].language) == ('print("hi")', "python")
    assert doc.title == "Handbook"
    assert all(b.provenance.source == "handbook.md" and b.provenance.page == 1 for b in blocks)


def test_markdown_round_trips_structure() -> None:
    markdown = _parse(MARKDOWN, "handbook.md").to_markdown()
    assert "# Handbook" in markdown and "## Leave" in markdown
    assert "   - Certificate after 3 days" in markdown
    assert "1. Apply\n2. Wait" in markdown
    assert "| Pipe \\| cell | 1 |" in markdown
    assert "```python\nprint(\"hi\")\n```" in markdown
    # Parsing the rendering again yields the same blocks.
    again = _parse(markdown, "again.md")
    assert [type(b) for b in again.blocks] == [type(b) for b in _parse(MARKDOWN, "x.md").blocks]


def test_plain_text_is_paragraphs_only() -> None:
    doc = _parse("# not a heading\n\nsecond\r\n\r\nthird", "notes.txt")
    assert doc.media_type == "text/plain"
    assert [type(b) for b in doc.blocks] == [Paragraph, Paragraph, Paragraph]
    assert doc.blocks.__next__().text == "# not a heading"


def test_non_utf8_text_is_detected_and_reported() -> None:
    # Was: UTF-8 only, so this byte became U+FFFD. The encoding is now detected, and
    # the guess is still reported rather than silent.
    doc = DocumentParser(ocr=None).parse(b"caf\xe9 au lait", filename="menu.txt")
    assert "café au lait" in doc.to_text()
    assert [w.code for w in doc.warnings] == ["encoding_detected"]


def test_undecodable_declared_text_is_reported() -> None:
    doc = DocumentParser(ocr=None).parse(
        b"caf\xe9 au lait", filename="menu.txt", media_type="text/plain; charset=utf-8"
    )
    assert "caf� au lait" in doc.to_text()
    assert [w.code for w in doc.warnings] == ["decode_errors"]


def test_bom_is_dropped() -> None:
    doc = DocumentParser(ocr=None).parse(b"\xef\xbb\xbf# Title", filename="a.md")
    assert doc.title == "Title"


HTML = """<!doctype html>
<html><head><title> Policy  Page </title><style>p{color:red}</style>
<script>var x = "<p>not content</p>";</script></head>
<body>
<nav>Home</nav>
<h1>Expense <em>policy</em></h1>
<p>Claims are paid
   monthly.<br>Late claims are not.</p>
<div>Loose div text</div>
<ul><li>Meals<ul><li>Lunch</li></ul></li><li>Travel</ol></li></ul>
<ol><li>Submit<li>Approve</ol>
<table>
  <tr><th>Item</th><th>Limit</th></tr>
  <tr><td>Hotel</td><td>120 <b>EUR</b></td></tr>
  <tr><td>Nested</td><td><table><tr><td>a</td><td>b</td></tr></table></td></tr>
</table>
<table><tr><td>x</td><td>y</td></tr></table>
<img src="chart.png" alt="Spend chart">
<pre><code class="language-sql">SELECT *
  FROM claims;</code></pre>
</body></html>
"""


def test_html_blocks() -> None:
    doc = _parse(HTML, "policy.html")
    assert doc.media_type == "text/html"
    assert doc.metadata["title"] == "Policy Page"
    texts = [(type(b).__name__, b.text) for b in doc.blocks]
    assert ("Heading", "Expense policy") in texts
    assert ("Paragraph", "Claims are paid monthly. Late claims are not.") in texts
    assert ("Paragraph", "Loose div text") in texts
    assert not any("not content" in t or "color" in t for _, t in texts)

    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert [(i.text, i.level) for i in lists[0].items] == [
        ("Meals", 0), ("Lunch", 1), ("Travel", 0),
    ]
    assert not lists[0].ordered
    assert lists[1].ordered and [i.text for i in lists[1].items] == ["Submit", "Approve"]

    tables = doc.tables
    assert tables[0].has_header
    assert tables[0].rows == (("Item", "Limit"), ("Hotel", "120 EUR"), ("Nested", "a b"))
    assert not tables[1].has_header and tables[1].rows == (("x", "y"),)
    assert "|  |  |\n| --- | --- |\n| x | y |" in tables[1].to_markdown()

    [figure] = [b for b in doc.blocks if isinstance(b, Figure)]
    assert (figure.alt, figure.src) == ("Spend chart", "chart.png")
    [code] = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code.content == "SELECT *\n  FROM claims;" and code.language == "sql"


def test_html_without_closing_tags() -> None:
    doc = _parse("<p>one<p>two<ul><li>a<li>b", "x.htm")
    assert [b.text for b in doc.blocks] == ["one", "two", "- a\n- b"]
