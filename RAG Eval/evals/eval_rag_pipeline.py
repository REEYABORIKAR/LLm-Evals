
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.evaluate import AsyncConfig

from src.rag_pipeline import RagPipeline
from evals.groq_judge import GroqJudge

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "openai/gpt-oss-20b"
THRESHOLD = 0.7

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)

rag = RagPipeline()
judge = GroqJudge(model=JUDGE_MODEL)

test_cases = []

for i, g in enumerate(goldens, start=1):
    print(f"\nRunning RAG pipeline: {i}/{len(goldens)}")

    result = rag.invoke(g["query"])

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],
            expected_output=g.get("expected_output"),
            retrieval_context=result["context"],
        )
    )

metrics = [
    ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=False,
        async_mode=False,
    ),
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=False,
        async_mode=False,
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=False,
        async_mode=False,
    ),
]

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        max_concurrent=1,
        throttle_value=5,
    ),
)

print("\nRAG pipeline evaluation completed.")
