"""Extraction's shared settings, read from ``KUHAKU_EXTRACTION__``-prefixed variables.

One place for what every extraction should do alike -- the router's own schema choice
included -- so nothing has to be set per extractor to start. An extractor that needs
something different gets its own ``ExtractionSettings``; an argument passed to
:class:`Extractor` or :class:`ExtractionRouter` wins over both.

Kept out of core's :class:`~kuhaku.core.config.Settings` on purpose: core may not depend
on a tool, so a tool's settings read the environment themselves.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kuhaku.core.security.input_policy import InputSecurity


class ExtractionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KUHAKU_EXTRACTION__",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
        # KUHAKU_EXTRACTION__SECURITY__SANITIZE=true sets security.sanitize.
        env_nested_delimiter="__",
    )

    max_retries: int = Field(default=1, ge=0)
    """Further model calls allowed to fix malformed output, per provider."""

    audit_log_path: str | None = None
    """Where audit records go. Unset means no audit log -- it writes a file."""

    clarification_question: str = "Could you also tell me: {fields}?"
    """The wording of ``result.question`` -- ``{fields}`` becomes the missing fields'
    descriptions (or names), comma-separated. Write it in your users' language, e.g.
    ``KUHAKU_EXTRACTION__CLARIFICATION_QUESTION="Şunu da yazar mısınız: {fields}?"``;
    an empty string turns the question off."""

    security: InputSecurity = Field(default_factory=InputSecurity)
    """Core's input checks (:class:`~kuhaku.core.security.input_policy.InputSecurity`),
    run by :class:`SecurityStep` on every extraction. A check core adds shows up here
    without a change to extraction."""
