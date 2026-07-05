from __future__ import annotations

from typing import Any, Protocol, Sequence


class Stage(Protocol):
    def process(self, data: Any) -> Any:
        ...


class PipeFilterPipeline:
    """Simple pipe-and-filter pipeline for data ingestion and retrieval stages."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = list(stages)

    def run(self, data: Any = None) -> Any:
        result = data
        for stage in self._stages:
            result = stage.process(result)
        return result
