"""HTML structure the token walker used to get wrong: omitted end tags in tables, SVG
titles, captions, unclosed headings, tables inside list items."""

from __future__ import annotations

from kuhaku.tools.parsing import DocumentParser, Heading, ListBlock, Paragraph, Table


def _parse(html: str):
    return DocumentParser(ocr=None).parse(html.encode(), filename="page.html")


def test_omitted_cell_and_row_end_tags_do_not_drop_the_document() -> None:
    doc = _parse("<table><tr><th>A<th>B<tr><td>1<td>2</table><p>after</p>")

    [table] = doc.tables
    assert table.rows == (("A", "B"), ("1", "2"))
    assert table.has_header
    assert [type(b) for b in doc.blocks] == [Table, Paragraph]
    assert doc.to_text().endswith("after")
    assert doc.warnings == ()


def test_tbody_and_cells_without_tr_are_handled() -> None:
    doc = _parse("<table><thead><tr><th>h</thead><tbody><td>x<td>y</tbody></table>")
    assert doc.tables[0].rows == (("h",), ("x", "y"))


def test_nested_table_text_stays_in_its_cell() -> None:
    doc = _parse(
        "<table><tr><td>a<table><tr><td>in<td>ner</table><td>b</table><p>z</p>"
    )
    assert doc.tables[0].rows == (("a in ner", "b"),)
    assert doc.to_text().endswith("z")


def test_a_table_left_open_is_emitted_and_reported() -> None:
    doc = _parse("<table><tr><td>1<td>2")
    assert doc.tables[0].rows == (("1", "2"),)
    assert [w.code for w in doc.warnings] == ["unclosed_table"]


def test_svg_title_does_not_become_the_document_title() -> None:
    doc = _parse("<body><svg><title>icon</title></svg><h1>Real</h1></body>")
    assert doc.title == "Real"
    doc = _parse(
        "<head><title>Page</title></head><body><svg><title>x</title></svg></body>"
    )
    assert doc.title == "Page"


def test_caption_is_kept_before_its_table() -> None:
    doc = _parse("<table><caption>Prices 2026</caption><tr><td>tea<td>5</table>")
    assert [type(b) for b in doc.blocks] == [Paragraph, Table]
    assert next(doc.blocks).text == "Prices 2026"


def test_unclosed_heading_stays_a_heading() -> None:
    doc = _parse("<h2>Title<p>body text</p>")
    blocks = list(doc.blocks)
    assert isinstance(blocks[0], Heading) and blocks[0].content == "Title"
    assert blocks[0].level == 2
    assert isinstance(blocks[1], Paragraph) and blocks[1].content == "body text"


def test_a_table_inside_a_list_item_keeps_document_order() -> None:
    doc = _parse("<ul><li>first<li>second<table><tr><td>t</table><li>third</ul>")
    blocks = list(doc.blocks)
    assert [type(b) for b in blocks] == [ListBlock, Table, ListBlock]
    assert [i.text for i in blocks[0].items] == ["first", "second"]
    assert [i.text for i in blocks[2].items] == ["third"]
