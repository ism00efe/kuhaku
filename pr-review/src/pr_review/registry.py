"""A tiny name-keyed registry.

Every pluggable kind in this package (review axes, depths, LLM providers, repo
discoverers, deterministic analyzers, verifiers, reporters, context strategies)
is registered here by string name. The pipeline iterates registries and never
switches on a name, so adding a component is purely additive: define a class,
call ``register``, enable it in config.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from pr_review.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a lowercase name to a factory returning a component instance."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(
        self, name: str, factory: Callable[..., T] | None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]] | Callable[..., T]:
        """Register ``factory`` under ``name``.

        Usable as ``registry.register("x", Factory)`` or as a decorator
        ``@registry.register("x")`` on a class / zero-arg callable.
        """
        key = name.lower()

        def _add(f: Callable[..., T]) -> Callable[..., T]:
            if key in self._factories:
                raise RegistryError(f"{self.kind} {name!r} is already registered")
            self._factories[key] = f
            return f

        if factory is not None:
            return _add(factory)
        return _add

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        try:
            factory = self._factories[name.lower()]
        except KeyError:
            raise RegistryError(
                f"unknown {self.kind} {name!r}; known: {sorted(self._factories)}"
            ) from None
        return factory(*args, **kwargs)  # type: ignore[call-arg]

    def has(self, name: str) -> bool:
        return name.lower() in self._factories

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())
