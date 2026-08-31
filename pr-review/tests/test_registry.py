import pytest

from pr_review.errors import RegistryError
from pr_review.registry import Registry


def test_register_and_create():
    r: Registry[str] = Registry("thing")

    @r.register("a")
    def _make_a() -> str:
        return "A"

    assert r.has("a")
    assert r.create("a") == "A"
    assert r.names() == ["a"]


def test_duplicate_rejected():
    r: Registry[int] = Registry("thing")
    r.register("x", lambda: 1)
    with pytest.raises(RegistryError):
        r.register("x", lambda: 2)


def test_unknown_name():
    r: Registry[int] = Registry("thing")
    with pytest.raises(RegistryError):
        r.create("missing")


def test_create_passes_args():
    r: Registry[int] = Registry("thing")
    r.register("add", lambda a, b: a + b)
    assert r.create("add", 2, 3) == 5
