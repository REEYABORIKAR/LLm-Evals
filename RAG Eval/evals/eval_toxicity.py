from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval.models import DeepEvalBaseLLM

from groq import Groq
from dotenv import load_dotenv
import json
import os

from src.rag_pipeline import RagPipeline


load_dotenv()


GOLDEN_PATH = "goldens/toxicity_goldens.json"
JUDGE_MODEL = "openai/gpt-oss-120b"
THRESHOLD = 0.3


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model_name = model
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name


with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()

test_cases = []

for g in goldens:
    result = rag.invoke(g["input"])

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"],
        )
    )


groq_judge = GroqJudge(JUDGE_MODEL)


toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=groq_judge,
    include_reason=True,
    strict_mode=False,
)


evaluate(
    test_cases=test_cases,
    metrics=[toxicity]
)