from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm

load_dotenv()

measured_chain= prompt | llm

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEAT = 3

PRICE_INPUT_PER_1M= 0.15
PRICE_CACHED_INPUT_PER_1M= 0.075
PRICE_OUTPUT_PER_1M= 0.60

QUERIES_PER_DAY= 2000
USD_TO_INR= 95.0

COST_BUDGET_PER_QUERY_USD= 0.0015

def measured_tokens(pipeline, question):
    docs= pipeline.retriever.invoke(question)
    context_text= "\n\n".join([doc.page_content for doc in docs])

    msg= measured_chain.invoke({"question": question, "context": context_text})
    usage= msg.usage_metadata or {}

    input_tokens= usage.get("input_tokens", 0)
    output_tokens= usage.get("output_tokens", 0)

    details= usage.get("input_tokens_details") or {}
    cached_tokens= details.get("cached_read", 0) or 0

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens
    }

def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input= max(input_tokens - cached_tokens, 0)
    c_in= uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached= cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out= output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    return {
        "input": c_in,
        "cached": c_cached,
        "output": c_out,
        "total": c_in + c_cached + c_out
    }

def benchmark(pipeline):
    rows= []
    print("Measuring token usage...")
    for question in QUESTIONS:
        for _ in range(REPEAT):
            tok= measured_tokens(pipeline, question)
            cost= cost_usd(tok["input"], tok["output"], tok["cached"])
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows

def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)

def report(rows):
    n= len(rows)
    avg_in= avg(rows, "input")
    avg_out= avg(rows, "output")
    avg_cached= avg(rows, "cached") 
    avg_cost= avg(rows, "cost_total")   
    min_cost= min(r["cost_total"] for r in rows)
    max_cost= max(r["cost_total"] for r in rows)

    avg_cost_in= avg(rows, "cost_input")+ avg(rows, "cost_cached")
    avg_cost_out= avg(rows, "cost_output")
    out_share= 100 * avg_cost_out / avg_cost if avg_cost else 0

    print("\n" + "=" * 70)
    print(f"COST  (openai/gpt-oss-20b @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)")
    print("=" * 70)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
          f"<- tight range = cost is stable, unlike latency")
    print(f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output "
          f"(output is 4x the rate -> long answers dominate)")
    print("-" * 70)

    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")


def main():
    pipeline = RagPipeline()
    rows = benchmark(pipeline)
    report(rows)

if __name__ == "__main__":
    main()