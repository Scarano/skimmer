from abc import ABC

from pydantic import BaseModel



class ScoredSpan(BaseModel):
    start: int
    end: int
    score: float


class SpanScorer(ABC):
    def __call__(self, doc: str) -> list[ScoredSpan]:
        pass
