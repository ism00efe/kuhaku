"""Two-stage risk classification for the v2 guard pipeline.

Stage 1 is always available: a deterministic rule-based scorer combining several cheap
signal groups into a 0.0-1.0 risk score, with an optional swap-in path for a trained
``sklearn`` char-n-gram Logistic Regression model (``models/stage1_lr.joblib``) once one
exists (see ``scripts/train_stage1.py``). Stage 2 is an optional multilingual transformer
classifier (ONNX); no model ships with this change, so it runs permanently degraded until
one is deployed — see ``guard.py``'s three-zone decision.

Both stages fail open/degrade rather than raise: a missing or broken model must never
crash a request.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter as _Counter
from dataclasses import dataclass
from pathlib import Path

from ..observability.metrics import record_guard_degradation
from .guard import _PATTERNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1Result:
    """0.0-1.0 risk score plus the top contributing signals, for audit/explainability."""

    score: float
    source: str  # "rule_based" | "lr_model"
    top_features: list[tuple[str, float]]


@dataclass(frozen=True)
class Stage2Result:
    ran: bool
    degraded: bool
    label: str | None  # "safe" | "unsafe" | None (None when it didn't run)
    confidence: float | None


# A shared "Stage-2 did not run" sentinel, imported lazily by guard.py (inside
# GuardPipeline.__init__, not at module scope) to avoid a circular import — this module
# imports `guard._PATTERNS`, so guard.py cannot import this module back at its own
# top level.
NOT_RUN_STAGE2 = Stage2Result(ran=False, degraded=False, label=None, confidence=None)


# ---------------------------------------------------------------------------
# Stage 1 — rule-based scorer (default, always available)
# ---------------------------------------------------------------------------

# Each regex category from the existing legacy guard (guard._PATTERNS) contributes to the
# score as one feature group instead of triggering an immediate binary reject — reused
# (imported), not re-implemented, so the two layers never drift out of sync.
_REGEX_GROUP_WEIGHTS: dict[str, float] = {
    "instruction_override": 0.35,
    "persona_override": 0.35,
    "role_switch": 0.20,
    "special_token": 0.20,
    "prompt_leak": 0.20,
}

# Injection/jailbreak phrasing beyond the 5 regex categories, English + Turkish (Turkish
# coverage is a documented gap in the legacy regex guard). Word/phrase
# substring match on lowercased text, deliberately fuzzier and lower-weighted than the
# strict regexes.
_KEYWORD_PHRASES: tuple[str, ...] = (
    # English
    "ignore previous",
    "disregard previous",
    "act as",
    "pretend you are",
    "developer mode",
    "unlock mode",
    "bypass your",
    "override your",
    "reveal your",
    "no restrictions",
    # Turkish
    "sistem promptu",
    "önceki talimatları yoksay",
    "talimatları görmezden gel",
    "rol yap",
    "gizli talimat",
    "geliştirici modu",
    "kısıtlamaları kaldır",
    "artık sınırsız",
    "kurallarını yoksay",
    "kurallarını unut",
)
_KEYWORD_WEIGHT = 0.15
_KEYWORD_CAP = 0.30

# Structural mimicry of a chat-template turn that the strict role_switch/special_token
# regexes don't already cover (near-miss variants: JSON-shaped role fields, XML tags,
# fenced code blocks framed as a system/instruction block).
_FORMAT_MIMICRY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r'"role"\s*:\s*"(system|assistant)"', re.IGNORECASE),
    re.compile(r"<system>.*?</system>", re.IGNORECASE | re.DOTALL),
    re.compile(r"```\s*(system|instruction)", re.IGNORECASE),
)
_FORMAT_MIMICRY_WEIGHT = 0.20

_ENTROPY_THRESHOLD = 4.2
_ENTROPY_WEIGHT = 0.15
_ENTROPY_MIN_LENGTH = 20

_SCRIPT_MIXING_WEIGHT = 0.15

_LENGTH_THRESHOLD = 2000
_LENGTH_WEIGHT = 0.05

_PUNCT_DENSITY_CHARS = frozenset("{}<>|`")
_PUNCT_DENSITY_THRESHOLD = 0.03
_PUNCT_DENSITY_WEIGHT = 0.10

_TOP_FEATURES_COUNT = 5


def _shannon_entropy(text: str) -> float:
    """Bits-per-character Shannon entropy of ``text``."""

    if not text:
        return 0.0
    counts = _Counter(text)
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _script_of(ch: str) -> str | None:
    """Coarse Unicode-block classification for one alphabetic character."""

    if not ch.isalpha():
        return None
    cp = ord(ch)
    if cp < 0x0250:
        # Basic Latin + Latin-1 Supplement + Latin Extended-A/B — covers Turkish
        # (ğ, ş, ı, İ, ö, ü, ç) as plain Latin, so normal Turkish text never triggers
        # script-mixing on its own.
        return "latin"
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x4E00 <= cp <= 0x9FFF:
        return "cjk"
    return None


class RuleBasedScorer:
    """Combines several cheap, deterministic signal groups into a 0.0-1.0 risk score.

    Weights below are a documented starting point, expected to be superseded once
    ``scripts/train_stage1.py`` has real labeled data
    to retune them against. The sum of every group's maximum comfortably exceeds 1.0 by
    design: a single maxed-out signal should not alone reach HIGH_THRESH — REJECT should
    require corroborating signals, not one keyword hit. The final ``min(1.0, total)`` clip
    is load-bearing, not decorative.
    """

    def score(self, normalized_text: str) -> Stage1Result:
        contributions: dict[str, float] = {}

        for pattern, reason in _PATTERNS:
            if pattern.search(normalized_text):
                contributions[f"regex_{reason}"] = _REGEX_GROUP_WEIGHTS[reason]

        keyword_score = self._keyword_score(normalized_text)
        if keyword_score:
            contributions["keyword_hits"] = keyword_score

        if any(r.search(normalized_text) for r in _FORMAT_MIMICRY_RES):
            contributions["format_mimicry"] = _FORMAT_MIMICRY_WEIGHT

        entropy_score = self._entropy_score(normalized_text)
        if entropy_score:
            contributions["entropy"] = entropy_score

        if self._script_mixing(normalized_text):
            contributions["script_mixing"] = _SCRIPT_MIXING_WEIGHT

        if len(normalized_text) > _LENGTH_THRESHOLD:
            contributions["length_abnormality"] = _LENGTH_WEIGHT

        punct_score = self._punct_density_score(normalized_text)
        if punct_score:
            contributions["punct_density"] = punct_score

        total = min(1.0, sum(contributions.values()))
        top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        return Stage1Result(
            score=total, source="rule_based", top_features=top[:_TOP_FEATURES_COUNT]
        )

    @staticmethod
    def _keyword_score(text: str) -> float:
        low = text.lower()
        hits = sum(1 for phrase in _KEYWORD_PHRASES if phrase in low)
        return min(_KEYWORD_CAP, hits * _KEYWORD_WEIGHT)

    @staticmethod
    def _entropy_score(text: str) -> float:
        if len(text) < _ENTROPY_MIN_LENGTH:
            return 0.0
        entropy = _shannon_entropy(text)
        return min(_ENTROPY_WEIGHT, max(0.0, entropy - _ENTROPY_THRESHOLD) * _ENTROPY_WEIGHT)

    @staticmethod
    def _script_mixing(text: str) -> bool:
        scripts = {s for ch in text if (s := _script_of(ch)) is not None}
        non_latin = scripts - {"latin"}
        return "latin" in scripts and bool(non_latin)

    @staticmethod
    def _punct_density_score(text: str) -> float:
        if not text:
            return 0.0
        density = sum(1 for ch in text if ch in _PUNCT_DENSITY_CHARS) / len(text)
        return _PUNCT_DENSITY_WEIGHT if density > _PUNCT_DENSITY_THRESHOLD else 0.0


# ---------------------------------------------------------------------------
# Stage 1 — trained LR model swap-in (active once a model file exists)
# ---------------------------------------------------------------------------
class LRModelScorer:
    """Wraps a fitted ``sklearn.pipeline.Pipeline`` loaded from ``model_path`` via
    ``joblib.load``. Expected shape: ``Pipeline([("vectorizer",
    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))), ("clf",
    LogisticRegression)])`` — the exact shape ``scripts/train_stage1.py`` produces.

    Raises on construction if the file is missing/corrupt/wrong-shaped; the caller
    (``TwoStageClassifier``) is responsible for catching that and falling back to
    :class:`RuleBasedScorer` (this pipeline's fail-open requirement).
    """

    def __init__(self, model_path: str) -> None:
        import joblib

        pipeline = joblib.load(model_path)
        # Fail fast on an unexpected artifact shape rather than a confusing AttributeError
        # the first time a request comes in.
        _ = pipeline.named_steps["vectorizer"]
        _ = pipeline.named_steps["clf"]
        self._pipeline = pipeline

    def score(self, normalized_text: str) -> Stage1Result:
        vectorizer = self._pipeline.named_steps["vectorizer"]
        clf = self._pipeline.named_steps["clf"]

        proba = self._pipeline.predict_proba([normalized_text])[0]
        score = float(proba[1])

        vector = vectorizer.transform([normalized_text]).tocoo()
        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]
        contributions = [
            (str(feature_names[col]), float(coefs[col] * value))
            for col, value in zip(vector.col, vector.data, strict=True)
        ]
        contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return Stage1Result(
            score=score, source="lr_model", top_features=contributions[:_TOP_FEATURES_COUNT]
        )


# ---------------------------------------------------------------------------
# Stage 2 — optional multilingual transformer classifier (ONNX)
# ---------------------------------------------------------------------------
class Stage2Classifier:
    """Loads an ONNX sequence-classification model + HuggingFace tokenizer, if both the
    ``onnxruntime``/``tokenizers`` libraries and the model/tokenizer files are present.

    Neither library is a declared project dependency — a missing
    library is treated identically to a missing model file: both permanently degrade this
    instance for the lifetime of the process, never raise.
    """

    def __init__(self, onnx_path: str, tokenizer_path: str) -> None:
        self._degraded = False
        self._session: object | None = None
        self._tokenizer: object | None = None
        self._load(onnx_path, tokenizer_path)

    def _load(self, onnx_path: str, tokenizer_path: str) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            logger.warning(
                "Stage-2 libraries (onnxruntime/tokenizers) not installed; "
                "guard v2 Stage-2 runs permanently degraded",
                extra={"guard_component": "stage2"},
            )
            self._degraded = True
            record_guard_degradation("stage2")
            return

        if not Path(onnx_path).is_file() or not Path(tokenizer_path).is_file():
            logger.warning(
                "Stage-2 model/tokenizer file not found (%s / %s); "
                "guard v2 Stage-2 runs permanently degraded",
                onnx_path,
                tokenizer_path,
                extra={"guard_component": "stage2"},
            )
            self._degraded = True
            record_guard_degradation("stage2")
            return

        try:
            self._session = ort.InferenceSession(onnx_path)
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception:
            logger.exception("Failed to load Stage-2 ONNX model/tokenizer")
            self._degraded = True
            record_guard_degradation("stage2")

    def classify(self, normalized_text: str) -> Stage2Result:
        if self._degraded:
            return Stage2Result(ran=False, degraded=True, label=None, confidence=None)
        try:
            encoding = self._tokenizer.encode(normalized_text)  # type: ignore[union-attr]
            import numpy as np

            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
            outputs = self._session.run(  # type: ignore[union-attr]
                None, {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            logits = outputs[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            unsafe_prob = float(probs[-1])
            label = "unsafe" if unsafe_prob >= 0.5 else "safe"
            confidence = max(unsafe_prob, 1.0 - unsafe_prob)
            return Stage2Result(ran=True, degraded=False, label=label, confidence=confidence)
        except Exception:
            logger.exception(
                "Stage-2 runtime error; guard v2 Stage-2 now permanently degraded"
            )
            self._degraded = True
            record_guard_degradation("stage2")
            return Stage2Result(ran=False, degraded=True, label=None, confidence=None)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
class TwoStageClassifier:
    """Composes the Stage-1 backend (LR model if present and loadable, else rule-based)
    and the Stage-2 classifier. Constructed once (in ``service.build_service``), not
    per-request — mirrors how ``CrossEncoderReranker``/``QueryAnswerCache`` are built once
    and injected as collaborators.
    """

    def __init__(
        self, *, stage1_model_path: str, stage2_onnx_path: str, stage2_tokenizer_path: str
    ) -> None:
        self._rule_based = RuleBasedScorer()
        self._lr: LRModelScorer | None = None
        if Path(stage1_model_path).is_file():
            try:
                self._lr = LRModelScorer(stage1_model_path)
            except Exception:
                logger.warning(
                    "Failed to load Stage-1 LR model at %s; falling back to the "
                    "rule-based scorer",
                    stage1_model_path,
                    exc_info=True,
                    extra={"guard_component": "stage1"},
                )
                record_guard_degradation("stage1")
                self._lr = None
        self._stage2 = Stage2Classifier(stage2_onnx_path, stage2_tokenizer_path)

    def classify_stage1(self, normalized_text: str) -> Stage1Result:
        if self._lr is not None:
            try:
                return self._lr.score(normalized_text)
            except Exception:
                logger.warning(
                    "Stage-1 LR model inference failed; falling back to the "
                    "rule-based scorer for this request",
                    exc_info=True,
                    extra={"guard_component": "stage1"},
                )
                record_guard_degradation("stage1")
        return self._rule_based.score(normalized_text)

    def classify_stage2(self, normalized_text: str) -> Stage2Result:
        return self._stage2.classify(normalized_text)
