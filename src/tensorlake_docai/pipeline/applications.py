# SPDX-License-Identifier: Apache-2.0
"""Split entrypoints for the GPU/CPU throughput optimization (plan T1).

The single-container pipeline runs convert → OCR → classify/extract/format in one
process, holding a GPU through the (CPU/API) post-OCR steps. To isolate the GPU to
OCR only, we split the existing DAG at the OCR boundary:

* **OCR stage** — reuse ``normalize_file_type_and_upload`` with ``ocr_only=True``
  (see :func:`build_ocr_only_request`). ``route_after_ocr`` short-circuits and the
  app returns the OCR'd :class:`ParseResult` instead of dispatching the post-OCR DAG.
  Run this on the GPU function; the A100 is released the instant OCR returns.
* **Post-OCR stage** — :func:`resume_post_ocr_app` re-injects that ``ParseResult``
  and resumes from ``route_after_ocr`` (TableMerging → VLM classify/summarize →
  StructuredExtraction → format_final_output). Run this on a CPU function; it holds
  the LLM/API waits, not the GPU.

The intermediate ``ParseResult`` is plain Pydantic (verified JSON-lossless in
``tests/test_split_roundtrip.py``), so it crosses the function boundary as a dict.
"""

from tensorlake.applications import application, function, Retries

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.vlm.workflow_images import file_convertion_image

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
]


def build_ocr_only_request(raw_request: dict) -> dict:
    """Return a copy of the request flagged so the pipeline stops after OCR.

    The GPU stage runs ``normalize_file_type_and_upload`` with this request; the
    OCR backend fills ``ocr_text`` and ``route_after_ocr`` returns the ParseResult.
    """
    return {**raw_request, "ocr_only": True}


@application()
@function(
    description=(
        "Resume the parse pipeline from a completed (OCR'd) ParseResult: TableMerging, "
        "VLM classification/summarization, StructuredExtraction, and final formatting. "
        "Runs on CPU — no GPU is held during the LLM/API steps."
    ),
    image=file_convertion_image,
    secrets=SECRETS,
    timeout=30 * 60,
    cpu=2,
    memory=4,
    retries=Retries(max_retries=2),
)
def resume_post_ocr_app(parse_result: dict) -> dict:
    """Resume the post-OCR DAG from a serialized ParseResult.

    Args:
        parse_result: ``ParseResult.model_dump()`` produced by the OCR-only stage.

    Returns:
        The OpenIngest document dict (``{"document": ..., "usage": ...}``), identical
        to a single-call ``normalize_file_type_and_upload`` run.
    """
    pr = ParseResult.model_validate(parse_result)
    # Clear the OCR-only flag so route_after_ocr resumes the DAG instead of
    # short-circuiting again.
    pr.request.ocr_only = False
    return route_after_ocr(
        pr,
        log_prefix="RESUME_POST_OCR",
        dots_ocr=(pr.request.ocr_model == "dots-ocr"),
    )
