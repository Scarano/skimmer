import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from skimmer.abridger import Abridger, TEXT_PARAGRAPH_PATTERN
from skimmer.highlighter import HTMLHighlighter
from skimmer.span_scorer import ScoredSpan
from skimmer.openai_embedding import OpenAIEmbedding
from skimmer.openai_summarizer import OpenAISummarizer
from skimmer.parser import RightBranchingParser
from skimmer.summary_matching_scorer import SummaryMatchingScorer


# This is not at all production-ready. Among the issues: intermediate results are cached to
# the local filesystem and never purged.


# TODO: use environment variable to specify cache dir
cache_dir = '/tmp/skimmer-cache'
memory = joblib.Memory(cache_dir, mmap_mode='c', verbose=0)
embed = OpenAIEmbedding(memory=memory)
summarize = OpenAISummarizer(prompt_name='few-points-1', memory=memory)
parser = RightBranchingParser('en')
scorer = SummaryMatchingScorer(parser, embed, summarize)


app = FastAPI()


class ScoreRequest(BaseModel):
    text: str

@app.post("/score", response_model=List[ScoredSpan])
async def score(request: ScoreRequest) -> List[ScoredSpan]:
    return scorer(request.text)


class AbridgeRequest(BaseModel):
    text: str
    keep: float

@app.post("/abridge", response_model=str)
async def abridge(request: AbridgeRequest) -> str:

    abridger = Abridger(scorer, request.keep, paragraph_pattern = TEXT_PARAGRAPH_PATTERN,
                        ellipsis_string = ' [...] ')

    return abridger.abridge(request.text)


class HighlightRequest(BaseModel):
    text: str
    proportion: float

@app.post("/highlight", response_model=str)
async def highlight(request: HighlightRequest) -> str:
    """
    This is a quick and dirty implementation that just adds HTML to the input text, and won't
    work well if input already contains HTML tags.
    TODO: finish the version that parses the input as HTML, which is started in branch html-highlighter
    """

    highligher = HTMLHighlighter(scorer, request.proportion)
    return highligher.highlight(request.text)
