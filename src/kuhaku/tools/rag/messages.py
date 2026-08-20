"""Domain-specific, user-facing message strings for the RAG pipeline.

Extracted from ``rag/engine.py`` and ``rag/prompts.py`` so the pipeline's control flow
stays domain- and language-agnostic: swapping this module's constants (or injecting a
different :class:`EngineMessages`) changes every user-facing string without touching the
engine's or prompt builder's logic. Defaults below are exactly the strings the assistant
used before this extraction, so behavior is unchanged unless a caller overrides them.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- RAGEngine response strings ----------------------------------------------------

EMPTY_QUERY_MESSAGE = "Please enter a question or upload a log file."
EMPTY_KB_MESSAGE = "The knowledge base appears to be empty. Please ingest documents first."
NO_CHUNKS_MESSAGE = "Sorry, no documents relevant to this question were found in the knowledge base."
ACCESS_DENIED_MESSAGE = "Sorry, you do not have permission to access documents for this request."
# Formatted with a single ``joined`` keyword -- see RAGEngine._flag_unverified_citations.
UNVERIFIED_CITATIONS_WARNING_TEMPLATE = (
    "\n\n⚠️ Warning: some source references in this response ([{joined}]) could not be verified."
)

# -- User prompt template pieces (rag/prompts.py's build_user_prompt) --------------

QUESTION_LABEL = "QUESTION:"
SOURCES_LABEL = "SOURCES:"
NO_SOURCES_FALLBACK = "(No sources found.)"
# Formatted with a single ``example_tags`` keyword.
CITE_INSTRUCTION_TEMPLATE = (
    "Answer the question based on the sources above, and indicate the sources you used "
    "in the form {example_tags}."
)

# -- Ingestion errors (rag/ingestion.py) --------------------------------------------

# Formatted with a single ``error`` keyword.
PDF_READ_FAILED_TEMPLATE = "Failed to read PDF content: {error}"
# Formatted with ``suffix``/``allowed`` keywords.
UNSUPPORTED_FILE_TYPE_TEMPLATE = "Unsupported file type '{suffix}'. Supported types: {allowed}."
EMPTY_FILE_MESSAGE = "The uploaded file is empty."
# Formatted with ``detected``/``suffix`` keywords.
CONTENT_EXTENSION_MISMATCH_TEMPLATE = (
    "File content was detected as '.{detected}', which does not match the '{suffix}' "
    "extension and cannot be accepted."
)
MISSING_PDF_SIGNATURE_MESSAGE = "File has a '.pdf' extension but does not carry a PDF signature."
UNDECODABLE_CONTENT_MESSAGE = "File could not be converted to text (binary or corrupted content)."
NO_EXTRACTABLE_TEXT_MESSAGE = "No extractable text was found in the file."
# Formatted with a single ``reason`` keyword.
DOCUMENT_SECURITY_CHECK_FAILED_TEMPLATE = "Document failed the security check: {reason}"
# Formatted with a single ``max_bytes`` keyword.
UPLOAD_SIZE_EXCEEDED_TEMPLATE = "Uploaded file exceeds the maximum allowed size of {max_bytes} bytes."


@dataclass(frozen=True)
class EngineMessages:
    """Bundle of the RAG pipeline's user-facing strings, injectable at construction."""

    empty_query: str = EMPTY_QUERY_MESSAGE
    empty_kb: str = EMPTY_KB_MESSAGE
    no_chunks: str = NO_CHUNKS_MESSAGE
    access_denied: str = ACCESS_DENIED_MESSAGE
    unverified_citations_warning: str = UNVERIFIED_CITATIONS_WARNING_TEMPLATE
    question_label: str = QUESTION_LABEL
    sources_label: str = SOURCES_LABEL
    no_sources_fallback: str = NO_SOURCES_FALLBACK
    cite_instruction_template: str = CITE_INSTRUCTION_TEMPLATE

    # -- Ingestion errors --
    pdf_read_failed_template: str = PDF_READ_FAILED_TEMPLATE
    unsupported_file_type_template: str = UNSUPPORTED_FILE_TYPE_TEMPLATE
    empty_file: str = EMPTY_FILE_MESSAGE
    content_extension_mismatch_template: str = CONTENT_EXTENSION_MISMATCH_TEMPLATE
    missing_pdf_signature: str = MISSING_PDF_SIGNATURE_MESSAGE
    undecodable_content: str = UNDECODABLE_CONTENT_MESSAGE
    no_extractable_text: str = NO_EXTRACTABLE_TEXT_MESSAGE
    document_security_check_failed_template: str = DOCUMENT_SECURITY_CHECK_FAILED_TEMPLATE
    upload_size_exceeded_template: str = UPLOAD_SIZE_EXCEEDED_TEMPLATE


DEFAULT_ENGINE_MESSAGES = EngineMessages()
