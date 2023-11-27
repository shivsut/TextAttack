import transformers

from textattack.attack import Attack
from textattack.attack_recipes import AttackRecipe
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, InputColumnModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.grammaticality import PartOfSpeech, LanguageTool
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM, WordSwapWordNet
from textattack.constraints.semantics.lm_liklihood import lm_liklihood_constraint

class TextFoolerJin2019Adjusted_LM(AttackRecipe):
#def TextFoolerJin2019Adjusted_LM(model, SE_thresh=0.98, sentence_encoder='bert'):

    """
        Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019).

        Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment.

        https://arxiv.org/abs/1907.11932

        Constraints adjusted from paper to align with human evaluation.
    """

    @staticmethod
    def build(model_wrapper):


        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=60)
        # shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained(
        #     "distilroberta-base"
        # )
        # shared_tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     "distilroberta-base"
        # )
        # transformation = WordSwapMaskedLM(
        #     method="bert-attack", max_candidates=50, min_confidence=5e-4
        # )
        #transformation = WordSwapWordNet()
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone",
             "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow",
             "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been",
             "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by",
             "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down",
             "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything",
             "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven",
             "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
             "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it",
             "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
             "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely",
             "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not",
             "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others",
             "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she",
             "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than",
             "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
             "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru",
             "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn",
             "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where",
             "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
             "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn",
             "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        # constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        # use_constraint = UniversalSentenceEncoder(
        #     threshold=0.936338023,
        #     metric="cosine",
        #     compare_against_original=True,
        #     window_size=15,
        #     skip_text_shorter_than_window=True,
        # )
        # constraints.append(use_constraint)
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        #constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        #constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        # use_constraint = UniversalSentenceEncoder(
        #     threshold=0.840845057,
        #     metric="angular",
        #     compare_against_original=False,
        #     window_size=15,
        #     skip_text_shorter_than_window=True,
        # )
        # constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        # constraints.append(
        #     WordEmbeddingDistance(min_cos_sim=0.7)
        # )
        # import os
        # path = 'C:\\git\\nlp-fall-2023\\assignments\\fp\\fp-dataset-artifacts-main\\content\\fp-dataset-artifacts\\roberta-base-snli\\'
        # if not os.path.isdir(path):
        #     path = '/gdrive/MyDrive/fp/llmtrain/roberta-base-snli/'
        # constraints.append(
        #     lm_liklihood_constraint(model_path=path)
        # )
        # constraints.append(
        #     LanguageTool(1)
        # )
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
#     #
#     # Swap words with their embedding nearest-neighbors.
#     #
#     # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
#     #
#     # 50 nearest-neighbors with a cosine similarity of at least 0.5.
#     # (The paper claims 0.7, but analysis of the code and some empirical
#     # results show that it's definitely 0.5.)
#     #
#     transformation = WordSwapEmbedding(max_candidates=50)
#     #
#     # Don't modify the same word twice or stopwords
#     #
#     constraints = [
#         RepeatModification(),
#         StopwordModification()
#     ]
#     #
#     # Minimum word embedding cosine similarity of 0.9.
#     #
#     # constraints.append(
#     #     WordEmbeddingDistance(min_cos_sim=0.9)
#     # )
#     #
#     # Universal Sentence Encoder with a minimum angular similarity of ε = 0.7.
#     #
#     # if sentence_encoder == 'bert':
#     #     se_constraint = BERT(threshold=SE_thresh,
#     #                          metric='cosine', compare_against_original=False, window_size=15,
#     #                          skip_text_shorter_than_window=False)
#     # else:
#     #     se_constraint = UniversalSentenceEncoder(threshold=SE_thresh,
#     #                                              metric='cosine', compare_against_original=False, window_size=15,
#     #                                              skip_text_shorter_than_window=False)
#     # constraints.append(se_constraint)
#     #
#     # Do grammar checking
#     #
#     constraints.append(
#         lm_liklihood_constraint('C:\\git\\nlp-fall-2023\\assignments\\fp\\fp-dataset-artifacts-main\\content\\fp-dataset-artifacts\\roberta-base-snli\\')
#     )
#     constraints.append(
#         LanguageTool(0)
#     )
#
#     #
#     # Untargeted attack
#     #
#     goal_function = UntargetedClassification(model)
#
#     #
#     # Greedily swap words with "Word Importance Ranking".
#     #
#     search_method = GreedyWordSwapWIR()
#
#     return Attack(goal_function, constraints, transformation, search_method)
#
#
# attack = TextFoolerJin2019Adjusted_LM
