# SPDX-License-Identifier: Apache-2.0
"""HTML/string utilities used during cross-page table merging."""

from tensorlake_docai.tables.table_merging import (
    extract_json_from_response,
    get_table_column_count,
    get_table_column_types,
    infer_cell_type,
    merge_table_htmls,
    remove_header_rows_regex,
    slice_table_rows,
)

# --- merge_table_htmls ----------------------------------------------------


def test_merge_table_htmls_appends_rows():
    base = "<table><tr><td>a</td></tr></table>"
    nxt = "<table><tr><td>b</td></tr></table>"
    merged = merge_table_htmls(base, nxt)
    assert merged.count("<tr>") == 2
    assert "<td>a</td>" in merged and "<td>b</td>" in merged
    assert merged.strip().endswith("</table>")


def test_merge_table_htmls_skips_header_rows():
    base = "<table><tr><td>keep</td></tr></table>"
    # Two rows in next; skip the first.
    nxt = "<table><tr><td>drop</td></tr><tr><td>add</td></tr></table>"
    merged = merge_table_htmls(base, nxt, skip_rows=1)
    assert "<td>drop</td>" not in merged
    assert "<td>add</td>" in merged
    assert "<td>keep</td>" in merged


def test_merge_table_htmls_empty_next_returns_base():
    base = "<table><tr><td>a</td></tr></table>"
    assert merge_table_htmls(base, "<table></table>") == base


# --- slice_table_rows -----------------------------------------------------


def test_slice_table_rows_first_two():
    html = "<table><tr><td>1</td></tr>" "<tr><td>2</td></tr>" "<tr><td>3</td></tr></table>"
    out = slice_table_rows(html, end=2)
    assert "<td>1</td>" in out and "<td>2</td>" in out
    assert "<td>3</td>" not in out


def test_slice_table_rows_preserves_open_tag_attrs():
    html = '<table border="1"><tr><td>x</td></tr></table>'
    out = slice_table_rows(html, end=1)
    assert '<table border="1">' in out


# --- remove_header_rows_regex ---------------------------------------------


def test_remove_header_rows_regex_strips_n_rows():
    body = "<tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr>"
    assert remove_header_rows_regex(body, 0) == body
    out = remove_header_rows_regex(body, 2)
    assert "<td>a</td>" not in out
    assert "<td>b</td>" not in out
    assert "<td>c</td>" in out


# --- extract_json_from_response -------------------------------------------


def test_extract_json_plain():
    assert extract_json_from_response('{"a": 1}') == {"a": 1}


def test_extract_json_code_fenced():
    text = '```json\n{"a": 1, "b": [2]}\n```'
    assert extract_json_from_response(text) == {"a": 1, "b": [2]}


def test_extract_json_embedded_in_prose():
    text = 'Here you go: {"a": 1} — done.'
    assert extract_json_from_response(text) == {"a": 1}


def test_extract_json_invalid_returns_empty_dict():
    assert extract_json_from_response("nothing structured here") == {}


# --- column count / cell type ---------------------------------------------


def test_get_table_column_count_simple():
    html = "<table><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    assert get_table_column_count(html) == 3


def test_get_table_column_count_with_colspan():
    html = '<table><tr><td colspan="2">x</td><td>y</td></tr></table>'
    assert get_table_column_count(html) == 3


def test_get_table_column_count_empty():
    assert get_table_column_count("<table></table>") == 0


def test_infer_cell_type():
    assert infer_cell_type("") == "empty"
    assert infer_cell_type("   ") == "empty"
    assert infer_cell_type("123") == "number"
    assert infer_cell_type("$1,234.50") == "number"
    assert infer_cell_type("(1,234)") == "number"
    assert infer_cell_type("42%") == "number"
    assert infer_cell_type("Alice") == "text"


def test_get_table_column_types_counts_headers_as_text():
    html = (
        "<table>"
        "<tr><td>Name</td><td>Age</td><td>Salary</td></tr>"
        "<tr><td>Alice</td><td>30</td><td>$50,000</td></tr>"
        "<tr><td>Bob</td><td>25</td><td>$60,000</td></tr>"
        "</table>"
    )
    types = get_table_column_types(html, rows_to_check=3, from_start=True)
    assert types == ["text", "text", "text"]


def test_get_table_column_types_detects_numeric_data_without_headers():
    html = (
        "<table>"
        "<tr><td>Alice</td><td>30</td><td>$50,000</td></tr>"
        "<tr><td>Bob</td><td>25</td><td>$60,000</td></tr>"
        "</table>"
    )
    assert get_table_column_types(html, rows_to_check=2, from_start=True) == [
        "text",
        "number",
        "number",
    ]
