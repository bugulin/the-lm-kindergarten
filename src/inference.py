import json
import logging
from collections.abc import Iterable
from os import getenv
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=getenv("LOGLEVEL", "INFO").upper())


class InputItem(TypedDict):
    """An item of input data."""

    id: str
    syllogism: str


class OutputItem(TypedDict):
    """An item of output data."""

    id: str
    validity: bool


class SyllogismSolver:
    """A syllogism solver."""

    def __init__(
        self, model: str, prompt_path: str = "prompt.j2", config: str | None = None
    ):
        self.pipe = pipeline(
            "text-generation",
            model,
        )

        env = Environment(loader=FileSystemLoader("src"))
        self.prompt = env.get_template("prompt.j2")

    def _generate_samples(self):
        with open("data/train_data.json") as fp:
            samples = json.load(fp)

        for sample in samples[:5]:
            yield {"role": "user", "content": sample["syllogism"]}
            yield {
                "role": "assistant",
                "content": ("INVALID", "VALID")[sample["validity"]],
            }

    def solve(self, data: Iterable[InputItem]):
        responses: list[OutputItem] = []
        for item in data:
            messages = [
                {
                    "role": "system",
                    "content": self.prompt.render(),
                },
                *self._generate_samples(),
                {"role": "user", "content": item["syllogism"]},
            ]
            response = self.pipe(messages)
            logger.info(response)
            answer = response[0]["generated_text"][-1]["content"]
            responses.append(
                {
                    "id": item["id"],
                    "validity": answer.startswith("VALID"),
                }
            )

        return responses
