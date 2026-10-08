from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import AnthropicModel
from agent_instrumented import support_agent

actual_output = support_agent("Where is my order ORD-1042?")

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)
judge = AnthropicModel(model="claude-haiku-4-5")
evaluate(test_cases =[test_case],
         metrics = [TaskCompletionMetric(threshold=0.7, model=judge)])
