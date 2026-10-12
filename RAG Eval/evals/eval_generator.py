from dotenv import load_dotenv

from evals.groq_judge import GroqJudge
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from deepeval.evaluate import AsyncConfig

from src.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH= "goldens/faithfulness_dataset.json"
JUDGE_MODEL= "openai/gpt-oss-120b"
judge = GroqJudge(model=JUDGE_MODEL)
THRESHOLD=  0.7

import time



def run():
    goldens = load_goldens(GOLDEN_PATH)
    all_test_results = []

    metrics = [
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

    for i, g in enumerate(goldens, start=1):
        context = g["ideal_context"]
        answer = generate(g["query"], context)

        test_case = LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context,
        )

        print(f"\nEvaluating test case {i}/{len(goldens)}")

        result = evaluate(
            test_cases=[test_case],
            metrics=metrics,
            async_config=AsyncConfig(
                max_concurrent=1,
                throttle_value=2,
            ),
        )

        all_test_results.extend(result.test_results)
        time.sleep(5)

    return summarize_by_metric(all_test_results)


if __name__ == "__main__":
    print_summary("generator", run())
