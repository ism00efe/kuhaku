"""Detect package managers, dependency files and frameworks from manifests."""

from __future__ import annotations

from pathlib import Path

from pr_review.discovery.base import DISCOVERERS, walk_files
from pr_review.models import RepoMetadata

# filename -> (package manager, dependency-file? )
MANIFESTS: dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "requirements.txt": "pip",
    "Pipfile": "pipenv",
    "poetry.lock": "poetry",
    "uv.lock": "uv",
    "package.json": "npm",
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "go.mod": "go-modules",
    "Cargo.toml": "cargo",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "Gemfile": "bundler",
    "composer.json": "composer",
    "mix.exs": "hex",
    "Package.swift": "swiftpm",
    "CMakeLists.txt": "cmake",
    "Dockerfile": "docker",
}
DEP_FILE_NAMES = set(MANIFESTS) | {"requirements-dev.txt", "constraints.txt"}

_FRAMEWORK_MARKERS = {
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "pytest": "pytest",
    "react": "React", "next": "Next.js", "vue": "Vue", "svelte": "Svelte",
    "express": "Express", "spring-boot": "Spring Boot", "rails": "Rails",
    "gin-gonic": "Gin", "actix": "Actix", "tokio": "Tokio",
}


@DISCOVERERS.register("manifests")
class ManifestDiscoverer:
    name = "manifests"

    def discover(self, root: Path) -> RepoMetadata:
        meta = RepoMetadata(root=str(root))
        for f in walk_files(root):
            if f.name in MANIFESTS:
                rel = f.relative_to(root).as_posix()
                pm = MANIFESTS[f.name]
                if pm not in meta.package_managers:
                    meta.package_managers.append(pm)
                if f.name in DEP_FILE_NAMES:
                    meta.dependency_files.append(rel)
                self._scan_frameworks(f, meta)
        meta.dependency_files = sorted(set(meta.dependency_files))
        return meta

    def _scan_frameworks(self, path: Path, meta: RepoMetadata) -> None:
        try:
            text = path.read_text("utf-8", errors="ignore").lower()
        except OSError:
            return
        for marker, name in _FRAMEWORK_MARKERS.items():
            if marker in text and name not in meta.frameworks:
                meta.frameworks.append(name)
