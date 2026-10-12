from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from src.reranker import RerankingRetriever
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)
from evals.groq_judge import GroqJudge


load_dotenv()


GOLDEN_PATH = "goldens/golden/retriever_goldens.json"

JUDGE_MODEL = GroqJudge(
    model="openai/gpt-oss-120b"
)
THRESHOLD = 0.7


def run(retriever):

    # 1. Load golden set
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    # 2. Run retriever for every golden
    for g in goldens:

        retrieved = retriever.invoke(
            g["query"]
        )

        retrieval_context = [
            doc.page_content
            for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. Metrics
    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=False,
            async_mode=False,
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=False,
            async_mode=False,
        ),

    ]

    # 4. Evaluate sequentially
    result = evaluate(

        test_cases=test_cases,

        metrics=metrics,

        async_config=AsyncConfig(
            run_async=False,
        ),

        hyperparameters={

            "retriever": "reranker",

            "embedding_model":
                "gemini-embedding-001",

            "reranker_model":
                "cross-encoder/ms-marco-MiniLM-L-6-v2",

            "chunk_size": 1000,

            "chunk_overlap": 150,

            "top_k": 5,

            "fetch_k": 10,

            "judge_model": "openai/gpt-oss-120b",

            "golden_set":
                GOLDEN_PATH,
        },
    )

    return summarize_by_metric(result)


def run_local():

    return run(
        RerankingRetriever()
    )


if __name__ == "__main__":

    print_summary(
        "retriever",
        run_local()
    )