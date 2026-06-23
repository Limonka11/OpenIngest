# SPDX-License-Identifier: Apache-2.0
"""T1 split: the OCR'd ParseResult must cross the GPU→CPU boundary as lossless JSON.

These tests need only pydantic (conftest puts ``src/`` on the path); they do NOT
import the Tensorlake runtime, torch, or vllm, so they run in CI without a GPU.
"""

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import (
    DocumentLayout,
    PageLayout,
    PageLayoutElement,
)
from tensorlake_docai.pipeline.api import PageFragmentType, ParseRequest


def _ocrd_parse_result() -> ParseResult:
    """A ParseResult as it looks AFTER OCR — page layout with ocr_text filled in."""
    element = PageLayoutElement(
        bbox=(10.0, 20.0, 110.0, 60.0),
        fragment_type=PageFragmentType.TEXT,
        score=0.99,
        reading_order=1,
        ref_id="1.1",
        ocr_text="Reference projects",
    )
    page = PageLayout(elements=[element], shape=(595, 842), page_number=1)
    return ParseResult(
        document_layout=DocumentLayout(pages=[page], scale_factor=1.0, total_pages=1),
        request=ParseRequest(
            file_name="d.pdf", mime_type="application/pdf", ocr_model="dots-ocr"
        ),
    )


def test_parseresult_json_roundtrip_is_lossless():
    pr = _ocrd_parse_result()
    dumped = pr.model_dump()
    restored = ParseResult.model_validate(dumped)
    # Re-dump must equal the first dump: no field is dropped or mutated across the
    # GPU→CPU function boundary (handoff fidelity, plan risk R2).
    assert restored.model_dump() == dumped
    # The OCR text and bbox a downstream resume needs are intact.
    assert restored.document_layout.pages[0].elements[0].ocr_text == "Reference projects"
    assert restored.document_layout.pages[0].elements[0].bbox == (10.0, 20.0, 110.0, 60.0)


def test_ocr_only_flag_defaults_false_and_survives_roundtrip():
    pr = _ocrd_parse_result()
    # New T1 field: defaults off so existing single-call behaviour is unchanged.
    assert pr.request.ocr_only is False
    pr.request.ocr_only = True
    restored = ParseResult.model_validate(pr.model_dump())
    assert restored.request.ocr_only is True


def test_raw_file_bytes_break_json_but_base64_transport_roundtrips():
    """normalize() leaves raw PDF bytes on request.file_bytes; raw bytes break the
    JSON output boundary, so the OCR-only seam base64-encodes them and resume decodes
    them back. Lock that contract (pydantic-only, runs locally)."""
    import base64
    import pytest

    raw = b"%PDF-1.3\n\xaa\xab\xac\xad binary"
    pr = _ocrd_parse_result()
    pr.request.file_bytes = raw  # what normalize() leaves on the request

    # Raw bytes on a str-typed field can't be JSON-serialized (the observed failure).
    with pytest.raises(Exception):
        pr.model_dump_json()

    # Seam transform: bytes -> base64 str.
    pr.request.file_bytes = base64.b64encode(raw).decode("ascii")
    dumped = pr.model_dump_json()  # now JSON-safe
    assert isinstance(dumped, str)

    # Resume transform: base64 str -> original bytes.
    restored = ParseResult.model_validate_json(dumped)
    assert base64.b64decode(restored.request.file_bytes) == raw


def test_route_after_ocr_short_circuits_when_ocr_only(pytestconfig):
    """With ocr_only=True, route_after_ocr returns the OCR'd ParseResult and does
    NOT dispatch the post-OCR DAG. Needs the full dependency env (boto3/tensorlake),
    so it runs in fork CI and skips in the dep-light local env."""
    import pytest

    pytest.importorskip("boto3")
    pytest.importorskip("tensorlake")
    from tensorlake_docai.pipeline.routing import route_after_ocr

    pr = _ocrd_parse_result()
    pr.request.ocr_only = True
    out = route_after_ocr(pr, log_prefix="TEST", dots_ocr=True)
    assert out is pr  # same object back, no future / no downstream task dispatched
