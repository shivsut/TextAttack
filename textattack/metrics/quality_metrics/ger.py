import language_tool_python

from textattack import Metric
from textattack.attack_results import FailedAttackResult, SkippedAttackResult


class GERMetric(Metric):
    def __init__(self, **kwargs):
        self.lang_tool = language_tool_python.LanguageTool('eng')



    def calculate(self, results):
        num_results = 0
        num_err = 0
        for result in results:
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                num_err += len(self.lang_tool.check(result.perturbed_result.attacked_text))
        return {"avg_grammar_errs" : round(num_err/num_results, 2)}