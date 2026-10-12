
from src.reranker import RerankingRetriever
from src.generator import generate
from langsmith import traceable


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        self.retriever = RerankingRetriever(
            fetch_k=fetch_k,
            top_k=top_k
        )

    @traceable(run_type="chain", name="RAG Pipeline")
    def invoke(self, query: str) -> dict:
        docs = self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]
        answer = generate(query, context)

        return {
            "query": query,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    rag = RagPipeline()

    result = rag.invoke(
        "What is drift and why does it matter after deployment?"
    )

    print("QUERY:", result["query"])
    print("ANSWER:", result["answer"])
    print("CONTEXT:", result["context"])

    for i, chunk in enumerate(result["context"]):
        print(f"[{i}]: {chunk[:120]}...")
