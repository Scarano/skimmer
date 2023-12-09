import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from skimmer.abridger import ScoredSpan
from skimmer.embedding_abridger import OpenAIEmbedding, OpenAISummarizer
from skimmer.parser import RightBranchingParser
from skimmer.summary_matching_abridger import SummaryMatchingAbridger


memory = joblib.Memory('cache', mmap_mode='c', verbose=0)
embed = OpenAIEmbedding(memory=memory)
summarize = OpenAISummarizer(memory=memory)
parser = RightBranchingParser('en')
abridger = SummaryMatchingAbridger(parser, embed, summarize)


app = FastAPI()

class ScoreRequest(BaseModel):
    text: str

@app.post("/score", response_model=List[ScoredSpan])
async def score(request: ScoreRequest) -> List[ScoredSpan]:
    return abridger(request.text)

class AbridgeRequest(BaseModel):
    text: str
    keep: float

@app.post("/abridge", response_model=str)
async def abridge(request: AbridgeRequest) -> str:

    spans = abridger(request.text)

    threshold = np.percentile([span.score for span in spans], 100 - 100*request.keep)

    abridged = ' '.join(request.text[span.start:span.end]
                        for span in spans if span.score > threshold)

    return abridged
