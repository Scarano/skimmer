from typing import Optional

import joblib
import tiktoken
from openai import OpenAI

from skimmer import logger
from skimmer.util import abbrev


class OpenAISummarizer:
    """
    Summarization function (in the sense that it is __call__-able) that uses OpenAI's chat
    interface with a summarization prompt.
    """

    SUMMARIZE_PROMPT = """
        As a professional abridger, write a slightly shortened version of the provided text.
        Your version should include all the main ideas and essential information, but eliminate extraneous language, less-important points, redundant information, and redundant examples.
        It should preserve the style and jargon of the provided text.
        Rely strictly on the provided text, without including external information.
    """
        # Your version should be about three quarters of the length of the provided text.
    MAX_TOKENS = 2000  # TODO: should be based on model choice, not a constant

    client = OpenAI()  # TODO: should probably be passed in to constructor

    def __init__(self, model: str = 'gpt-3.5-turbo',
                 memory: Optional[joblib.Memory] = None):
        """
        :param model: OpenAI model to use for summarization
        :param memory: joblib Memory object to use for caching. If None, no caching will be done.
        """
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            raise Exception(f"No tiktoken encoding found for model {model}")

        self.summarize_func = lambda model, encoding, prompt, text: \
            OpenAISummarizer.uncached_summarize(
                model, encoding, prompt, text)
        if memory:
            self.summarize_func = memory.cache(self.summarize_func)

    @staticmethod
    def uncached_summarize(model, encoding, prompt, text: str) -> str:
        text = text.strip()

        tokens = encoding.encode(text)
        if len(tokens) > OpenAISummarizer.MAX_TOKENS:
            logger.warning(
                "OpenAISummarizer does not support more than %d "
                f"tokens at a time. Provided text (%s...) has %d tokens and had to be truncated.",
                OpenAISummarizer.MAX_TOKENS, text[:30], len(tokens))
            text = encoding.decode(tokens[:OpenAISummarizer.MAX_TOKENS])

        response = OpenAISummarizer.client.chat.completions.create(
            model=model, max_tokens=int(len(tokens)*.75), temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        # TODO Check finish_reason / errors
        summary = response.choices[0].message.content.strip()
        logger.debug("Summarization response for %s: %s",
                     abbrev(text, 80),
                     abbrev( summary, 80))

        return summary

    def __call__(self, text: str) -> str:
        return self.summarize_func(
            self.model, self.encoding, OpenAISummarizer.SUMMARIZE_PROMPT, text)
