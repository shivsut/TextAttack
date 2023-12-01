import copy
import json
from datetime import datetime

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

from textattack.constraints import Constraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class lm_liklihood_constraint(Constraint):
    def __init__(self, model_path, compare_against_original=True):
        super().__init__(compare_against_original)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.vocab = {}
        for i, (k,v) in enumerate(self.tokenizer.get_vocab().items()):
            self.vocab[v] = (k, i)
        self.label_tag = {0:"<ent>", 1:"<neu>", 2: "<con>" }
        self.label_token = {"<ent>" : self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<ent>"))[0],
                            "<neu>" : self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<neu>"))[0],
                            "<con>": self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<con>"))[0]}
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            self.model = self.model.to("cuda")
            self.device = "cuda"
        self.model.eval()
        self.debug_dict = {}
        self.debug_list = ['', '', -1]
        self.tmp_dict_below = {}
        self.tmp_dict_above = {}
        FORMAT = '%Y%m%d%H%M%S'
        self.file_name = 'lm_liklihood_debug_%s.json' % (datetime.now().strftime(FORMAT))
    def get_masked_sents(self, label, transformed_text, reference_text):
        #reference_text.attack_attrs['ground_truth']
        labelled_prem_hyp = label + transformed_text.tokenizer_input[0] + "</s></s>" + transformed_text.tokenizer_input[1]
        modified_indices = list(transformed_text.attack_attrs['modified_indices'])
        modified_word = ""
        replaced_word = ""
        modified_indices  = modified_indices[0]
        if modified_indices < len(transformed_text.words_per_input[0]):
            modified_word = reference_text.words_per_input[0][modified_indices]
            replaced_word = transformed_text.words_per_input[0][modified_indices]
        else:
            modified_word = reference_text.words_per_input[1][modified_indices-len(transformed_text.words_per_input[0])]
            replaced_word = transformed_text.words_per_input[1][modified_indices-len(transformed_text.words_per_input[0])]

        replaced_word_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' +replaced_word))
        #replaced_word_token = self.tokenizer.encode(replaced_word)
        labelled_prem_hyp_tokens = self.tokenizer.encode(labelled_prem_hyp)
        all_nasked_sents = []
        replaced_idx = []
        for i in range(0, len(labelled_prem_hyp_tokens) - len(replaced_word_token) + 1):
            if labelled_prem_hyp_tokens[i:i + len(replaced_word_token)] == replaced_word_token:
                for j in range(i, i + len(replaced_word_token)):
                    masked_sent = labelled_prem_hyp_tokens[:]
                    masked_sent[j] = self.tokenizer.mask_token_id
                    all_nasked_sents.append(masked_sent)
                    replaced_idx.append(j)
                #labelled_prem_hyp_tokens[i:i + len(replaced_word_token)] = mask_tokens
                break
        return all_nasked_sents, replaced_idx, replaced_word_token

    def replace_label(self, new_token, old_token, masked_sents):
        for sent in masked_sents:
            for i in range(len(sent)):
                if sent[i] == old_token:
                    sent[i] = new_token
        return masked_sents

    def check_if_new_dump(self, transformed_text, reference_text):
        if self.debug_list[0] != reference_text.tokenizer_input[0] or self.debug_list[1] != reference_text.tokenizer_input[1] or self.debug_list[2] != reference_text.attack_attrs['ground_truth']:
            if len(self.debug_dict) > 0:
                with open(self.file_name, 'a') as f:
                    self.debug_dict['can'] = {k: self.tmp_dict_above[k] for k in (sorted(self.tmp_dict_above, reverse=False)[:5])}
                    self.debug_dict['can'].update({k:self.tmp_dict_below[k] for k in (sorted(self.tmp_dict_below, reverse=True)[:5])})
                    json.dump(self.debug_dict, f, indent=1)
                    f.write('\n')
            self.debug_dict.clear()
            self.tmp_dict_above.clear()
            self.tmp_dict_below.clear()
            self.debug_list[0] = reference_text.tokenizer_input[0]
            self.debug_list[1] = reference_text.tokenizer_input[1]
            self.debug_list[2] = reference_text.attack_attrs['ground_truth']
            self.debug_dict['inp'] = self.debug_list

    def _check_constraint(self, transformed_text, reference_text):
        ent_masked_sents, replaced_idx, replaced_token = self.get_masked_sents("<ent>",transformed_text, reference_text)
        if not ent_masked_sents:
            return False
        neu_masked_sents = self.replace_label(self.label_token["<neu>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(ent_masked_sents))
        con_masked_sents = self.replace_label(self.label_token["<con>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(ent_masked_sents))

        self.check_if_new_dump(transformed_text, reference_text)

        input_ids = torch.tensor( [sent for sent in ent_masked_sents + neu_masked_sents + con_masked_sents], dtype=torch.long).to(self.device)
        mask_ids = torch.tensor([[1] * len(sent) for sent in ent_masked_sents + neu_masked_sents + con_masked_sents]).to(self.device)
        with torch.no_grad():
            self.model.eval()
            model_out = self.model(input_ids=input_ids, attention_mask=mask_ids)

            logits = model_out[0].detach()

            pos_idx = torch.tensor(replaced_idx).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, logits.size(-1))
            probs_idx = torch.gather(logits, 1, pos_idx).softmax(dim=-1).squeeze(1)
            tokens_tensor = torch.tensor(replaced_token).unsqueeze(1).repeat(3,1)
            probs = torch.gather(probs_idx, 1, tokens_tensor).squeeze(-1)

        lable_idx_map = {}
        idx = 0
        for i in range(3):
            indices = []
            for _ in range(len(ent_masked_sents)):
                indices.append(idx)
                idx += 1
            lable_idx_map[i] = indices
        gold_label = reference_text.attack_attrs['ground_truth']
        gold_probs = probs[lable_idx_map[gold_label]].prod(0)

        lable_idx_map.pop(gold_label)
        other_probs = []
        for label, idx in lable_idx_map.items():
            other_probs.append(probs[idx].prod(0))
        other_prob = max(other_probs)
        ratio = gold_probs/other_prob
        if ratio >= 1:
            self.tmp_dict_above[ratio.item()] = transformed_text.tokenizer_input[0]
            return True
        else:
            self.tmp_dict_below[ratio.item()] = transformed_text.tokenizer_input[0]
            return False

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)


