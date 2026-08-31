"""Detect languages by file extension (and a few well-known basenames)."""

from __future__ import annotations

from pathlib import Path

from pr_review.discovery.base import DISCOVERERS, walk_files
from pr_review.models import RepoMetadata

EXT_LANG: dict[str, str] = {
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".scala": "Scala",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".swift": "Swift",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++", ".cxx": "C++",
    ".m": "Objective-C", ".mm": "Objective-C++",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".sql": "SQL", ".r": "R", ".dart": "Dart", ".ex": "Elixir", ".exs": "Elixir",
    ".clj": "Clojure", ".hs": "Haskell", ".lua": "Lua", ".pl": "Perl",
    ".yml": "YAML", ".yaml": "YAML", ".tf": "Terraform",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".vue": "Vue", ".svelte": "Svelte",
}
BASENAME_LANG = {"Dockerfile": "Dockerfile", "Makefile": "Make", "Rakefile": "Ruby"}
_CODE_LANGS = {
    "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "Kotlin", "Scala",
    "Ruby", "PHP", "C#", "Swift", "C", "C++", "Objective-C", "Objective-C++",
    "Shell", "Dart", "Elixir", "Clojure", "Haskell", "Lua", "Perl", "Vue", "Svelte",
}


def language_for(path: str) -> str:
    p = Path(path)
    return EXT_LANG.get(p.suffix.lower(), BASENAME_LANG.get(p.name, ""))


@DISCOVERERS.register("languages")
class LanguageDiscoverer:
    name = "languages"

    def discover(self, root: Path) -> RepoMetadata:
        counts: dict[str, int] = {}
        for f in walk_files(root):
            lang = EXT_LANG.get(f.suffix.lower()) or BASENAME_LANG.get(f.name)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        meta = RepoMetadata(root=str(root), languages=counts)
        code = {k: v for k, v in counts.items() if k in _CODE_LANGS}
        if code:
            meta.conventions["primary_code_language"] = max(code.items(), key=lambda kv: kv[1])[0]
        return meta
