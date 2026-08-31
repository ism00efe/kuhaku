"""Policy that acts on an already-chosen vector store.

This is not probing -- adapters only report candidates and cost -- so it lives beside the
adapters, not among them. Two concerns:

  - *Upgrade suggestion*: once the corpus crosses a size where a heavier store starts to
    pay off, present the options and let the operator choose. Never migrate automatically
    (a §14 non-goal).
  - *Concurrent writer*: catch a second process at the head of the write path and turn
    it into a clean :class:`~kuhaku.core.exceptions.StoreConflict`, never a partially
    written store.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

from kuhaku.core.exceptions import StoreConflict
from kuhaku.core.resolve import Candidate, Cost, Environment, Registry

# chunks; above this, suggest a store upgrade. An estimate, not a measurement --
# overridable via RAGSettings.chunk_upgrade_threshold.
CHUNK_UPGRADE_THRESHOLD = 50_000


def suggest_store_upgrade(
    chunk_count: int,
    *,
    current_id: str,
    registry: Registry,
    env: Environment,
    ui,
    threshold: int = CHUNK_UPGRADE_THRESHOLD,
) -> Candidate | None:
    """When ``chunk_count`` is past ``threshold``, offer the heavier stores the registry
    knows about alongside "stay". Returns the operator's pick (or ``None``). Performs no
    migration -- the caller is handed the choice, and the announcement states that the
    existing chunks and metadata carry over with no document reprocessing."""

    if chunk_count < threshold:
        return None

    heavier = [c for c in registry.candidates("store", env) if c.id != current_id]
    if not heavier:
        # no heavier store is registered yet -- still note the size so the operator
        # knows the threshold was crossed.
        ui.announce(
            f"the store holds {chunk_count:,} chunks, past the ~{threshold:,}-chunk "
            f"point where a heavier store starts to pay off; no alternative store is "
            f"available in this build."
        )
        return None

    stay = Candidate(
        id=current_id,
        kind="store",
        label=f"stay on {current_id}",
        cost=Cost(note="no change"),
        ready=True,
        safety_rank=0,
        activate=lambda: None,
    )
    options = [stay, *heavier]
    summary = (
        f"the store holds {chunk_count:,} chunks, past the ~{threshold:,}-chunk point "
        f"where a heavier store starts to pay off. This is a suggestion, not a "
        f"migration."
    )
    ui.announce(summary)
    choice = ui.ask(summary + " Choose one:", options) if ui.is_interactive() else None
    if choice is not None and choice.id != current_id:
        ui.announce(
            f"to move to {choice.label}: {choice.cost.note}. Existing chunks, metadata "
            f"and vectors carry over -- no source documents are reprocessed.",
            prominent=True,
        )
    return choice


_LOCK_NAME = ".kuhaku-writer.lock"


@contextlib.contextmanager
def guard_single_writer(store_dir: str | Path, *, ui) -> Iterator[None]:
    """Hold an exclusive lock file under ``store_dir`` for the duration of a write.

    A second process that reaches this while the lock is held gets a clean
    :class:`StoreConflict` (or, when interactive, a chance to override) rather than a
    corrupted store. §13 fallback: an explicit lock at the head of the write path.
    ``RAG.ingest`` wraps each ingest in this; a caller composing ``RAGEngine`` directly
    is responsible for its own write path.

    If the lock directory cannot be created or written (read-only filesystem), this
    yields without a lock rather than failing the write -- best-effort, like the rest of
    the resolver's on-disk state.
    """

    directory = Path(store_dir)
    lock = directory / _LOCK_NAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        message = (
            f"another process holds the write lock at {lock}. If that process is gone, "
            f"delete the file and retry."
        )
        if ui.is_interactive() and ui.confirm(f"{message}\nBreak the lock and continue?", Cost()):
            with contextlib.suppress(OSError):
                lock.unlink()
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise StoreConflict(message) from exc
    except OSError:
        # cannot create the lock at all (read-only dir, etc.) -- proceed unguarded
        yield
        return
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()
