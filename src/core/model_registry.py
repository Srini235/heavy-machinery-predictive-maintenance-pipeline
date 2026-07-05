"""On-disk Model Registry

Design notes:

- Purpose: provide a simple, file-backed model registry that maps artifact
    filenames -> sha256 + metadata using `model_registry/index.json`.
- Pattern: Model Registry (lightweight) supporting idempotency checks and
    runtime integrity verification by the API (compare saved sha256 with actual
    file checksum).
- Responsibility: serialization of the index, simple get/register/list API.
    Keep logic minimal to avoid coupling to remote stores; this module is easier
    to unit-test and reason about because it has a single responsibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelArtifact:
    path: str
    sha256: str
    metadata: dict[str, Any]


class ModelRegistry:
    """Minimal on-disk model registry for the hydraulic predictive-maintenance system."""

    def __init__(self, registry_dir: str | Path = "model_registry") -> None:
        self.registry_dir = str(registry_dir)
        self.registry_path = Path(registry_dir)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.registry_path / "index.json"

    def register(self, artifact: ModelArtifact) -> None:
        index = self._load_index()
        index[artifact.path] = {
            "sha256": artifact.sha256,
            "metadata": artifact.metadata,
        }
        self._save_index(index)

    def get(self, path: str) -> ModelArtifact | None:
        index = self._load_index()
        entry = index.get(path)
        if not entry:
            return None
        return ModelArtifact(path=path, sha256=entry["sha256"], metadata=entry["metadata"])

    def list(self) -> dict[str, Any]:
        return self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if not self._index_file.exists():
            return {}
        with open(self._index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_index(self, index: dict[str, Any]) -> None:
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
