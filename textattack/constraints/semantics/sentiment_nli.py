from transformers import pipeline

from textattack.constraints import Constraint
from textattack.shared.validators import transformation_consists_of_word_swaps

class sentiment_nli(Constraint):
    def __init__(self, compare_against_original=True):
        super().__init__(compare_against_original)
        self.pipeline = pipeline("sentiment-analysis")

    def _check_constraint(self, transformed_text, reference_text):
        ref_sent = self.pipeline(reference_text.tokenizer_input[0])
        tran_sent = self.pipeline(transformed_text.tokenizer_input[0])
        return ref_sent[0]['label'] == tran_sent[0]['label']
    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)