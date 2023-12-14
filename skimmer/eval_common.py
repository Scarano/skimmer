from dataclasses import dataclass


@dataclass
class ReferenceSummarySet:
    doc: str
    summaries: list[str]
