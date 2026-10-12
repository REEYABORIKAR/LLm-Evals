from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

load_dotenv()

# Test Case 1
case_1= LLMTestCase(
    input="What is the capital of France?",
    actual_output="The captial of France is Paris."
)

# Test Case 2
case_2= LLMTestCase(
    input="What is the captial of France?",
    actual_output="France is a beautiful country famous for its food and wine."
)

# One metric
metric= AnswerRelevancyMetric(threshold=0.7, model="llama-3.1-8b-instant", include_reason=True)

# Run(Evaluate) both cases throughthe metrics 
evaluate(test_cases=[case_1,case_2], metrics=metric)