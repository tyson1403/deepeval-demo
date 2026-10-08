from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
from agent_instrumented import support_agent as _support_agent

judge = AnthropicModel(model="claude-haiku-4-5")

@observe(name = "support_agent")
def support_agent(user_input:str) -> str:
    current_golden = get_current_golden()
    if current_golden:
        if current_golden.expected_tools:
            update_current_trace(expected_tools=current_golden.expected_tools)
        if current_golden.expected_output:
            update_current_trace(expected_output=current_golden.expected_output)
    return _support_agent(user_input)

task_completion = TaskCompletionMetric(threshold=0.9, model = judge)

tool_correctness = ToolCorrectnessMetric(model = judge)

dataSet = EvaluationDataset(goldens= [
    Golden(input = "Where in my order ORD-1990?", expected_tools= [ToolCall(name = "get_order_status") ]),
    Golden(input = "What is the refund policy for electronics", expected_tools=[ToolCall(name="get_refund_policy")]),
])

for golden in dataSet.evals_iterator(metrics = [task_completion, tool_correctness]):
    support_agent(golden.input)