from abc import ABC

from pydantic import BaseModel



class ScoredSpan(BaseModel):
    start: int
    end: int
    score: float


class Abridger(ABC):
    def __call__(self, doc: str) -> list[ScoredSpan]:
        pass