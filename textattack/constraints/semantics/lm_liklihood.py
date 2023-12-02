import copy
import gc
import json
from datetime import datetime

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

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
        FORMAT = '%Y%m%d%H%M%S'
        self.file_name = 'lm_liklihood_debug_%s.json' % (datetime.now().strftime(FORMAT))

    def get_masked_sents_all(self, label, transformed_texts, reference_text):
        res = []
        max_sent_token_len = 0
        for t_text in transformed_texts:
            all_masked_sents, replaced_idx, replaced_word_token, sent_len = self.get_masked_sents(label, t_text, reference_text)
            res.append([all_masked_sents, replaced_idx, replaced_word_token])
            max_sent_token_len = max(max_sent_token_len, sent_len)
        return res, max_sent_token_len
    def get_masked_sents(self, label, transformed_text, reference_text):
        #reference_text.attack_attrs['ground_truth']
        labelled_prem_hyp = label + transformed_text.tokenizer_input[0] + "</s></s>" + transformed_text.tokenizer_input[1]
        modified_indices = list(transformed_text.attack_attrs['modified_indices'])
        modified_word = ""
        replaced_word = ""
        modified_idx = modified_indices[0]
        if modified_idx < len(transformed_text.words_per_input[0]):
            modified_word = reference_text.words_per_input[0][modified_idx]
            replaced_word = transformed_text.words_per_input[0][modified_idx]
        else:
            modified_word = reference_text.words_per_input[1][modified_idx-len(transformed_text.words_per_input[0])]
            replaced_word = transformed_text.words_per_input[1][modified_idx-len(transformed_text.words_per_input[0])]

        replaced_word_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize((' ' if modified_idx > 0 else '') + replaced_word))
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
        return all_nasked_sents, replaced_idx, replaced_word_token, len(labelled_prem_hyp_tokens)
    def replace_label(self, new_token, old_token, masked_sents):
        for sent in masked_sents:
            for i in range(len(sent)):
                if sent[i] == old_token:
                    sent[i] = new_token
        return masked_sents
    def _check_constraint_many(self, transformed_texts, reference_text):
        #ent_masked_sents, replaced_idx, replaced_token = self.get_masked_sents("<ent>",transformed_text, reference_text)
        sent_info_list, max_sent_token_len = self.get_masked_sents_all("<ent>", transformed_texts, reference_text)
        all_replaced_idx_list = []
        all_replaced_word_token_list = []
        all_sent_tokens = []
        for sent_info in sent_info_list:
            all_replaced_idx_list.append(sent_info[1])
            all_replaced_word_token_list.append(sent_info[2])
            all_sent_tokens.extend(sent_info[0])
            all_sent_tokens.extend(self.replace_label(self.label_token["<neu>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(sent_info[0])))
            all_sent_tokens.extend(self.replace_label(self.label_token["<con>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(sent_info[0])))
        # ent_masked_sents = sent_info_list[0]
        #
        # if not ent_masked_sents:
        #     return False
        # neu_masked_sents = self.replace_label(self.label_token["<neu>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(ent_masked_sents))
        # con_masked_sents = self.replace_label(self.label_token["<con>"], self.label_token["<ent>"], masked_sents=copy.deepcopy(ent_masked_sents))

        #self.check_if_new_dump(transformed_text, reference_text)

        input_ids = torch.tensor( [sent + [self.tokenizer.pad_token_id] * (max_sent_token_len - len(sent)) for sent in all_sent_tokens], dtype=torch.long).to(self.device)
        mask_ids = torch.tensor([[1] * len(sent) + [0] * (max_sent_token_len - len(sent)) for sent in all_sent_tokens]).to(self.device)
        del all_sent_tokens
        del sent_info
        del sent_info_list
        gc.collect()
        if len(input_ids)  == 0:
            return []
        dataset = TensorDataset(input_ids, mask_ids)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=len(input_ids))

        all_probs = []
        each_t_text_list_idx = 0
        with torch.no_grad():
            self.model.eval()
            for input_ids, input_mask in data_loader:
                model_out = self.model(input_ids=input_ids, attention_mask=input_mask)
                logits = model_out[0].detach()

            pos_idx = torch.tensor(data = [],  dtype=torch.int64)
            tokens_tensor = torch.tensor(data = [],  dtype=torch.int64)
            for i in range(len(all_replaced_idx_list)):
                pos_idx = torch.cat((pos_idx, torch.tensor(all_replaced_idx_list[i]).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, logits.size(-1))), 0)
                tokens_tensor = torch.cat((tokens_tensor, torch.tensor(all_replaced_word_token_list[i]).unsqueeze(1).repeat(3, 1)), 0)
            #pos_idx = torch.tensor(replaced_idx).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, logits.size(-1))
            try:
                probs_idx = torch.gather(logits, 1, pos_idx).softmax(dim=-1).squeeze(1)
            except Exception as e:
                print(e)
                raise e
            # tokens_tensor = torch.tensor(replaced_token).unsqueeze(1).repeat(3,1)
            probs = torch.gather(probs_idx, 1, tokens_tensor).squeeze(-1)

        del logits
        del model_out
        gc.collect()

        curr_prob_idx = 0
        res = []
        debug_dict_above = {}
        debug_dict_below = {}
        for i in range(len(transformed_texts)):
            l = len(all_replaced_idx_list[i])
            t_probs = probs[curr_prob_idx:curr_prob_idx+l*3]
            curr_prob_idx += l*3
            ent_prob = t_probs[0:l].prod(0)
            neu_prob = t_probs[l:2*l].prod(0)
            con_prob = t_probs[2*l:3*l].prod(0)
            gold_prob = 1
            other_prob = 1
            gold_label = reference_text.attack_attrs['ground_truth']
            if gold_label == 0:
                gold_prob = ent_prob
                other_prob = max(neu_prob, con_prob)
            elif gold_label == 1:
                gold_prob = neu_prob
                other_prob = max(ent_prob, con_prob)
            else:
                gold_prob = con_prob
                other_prob = max(ent_prob, neu_prob)
            ratio = gold_prob/other_prob
            if ratio >= 1:
                debug_dict_above[ratio.item()] = transformed_texts[i].tokenizer_input[0]
                res.append(transformed_texts[i])
            else:
                debug_dict_above[ratio.item()] = transformed_texts[i].tokenizer_input[0]
        self.logdebug(debug_dict_above, debug_dict_below, reference_text)
        return res
        #
        #
        # lable_idx_map = {}
        # idx = 0
        # for i in range(3):
        #     indices = []
        #     for _ in range(len(ent_masked_sents)):
        #         indices.append(idx)
        #         idx += 1
        #     lable_idx_map[i] = indices
        # gold_label = reference_text.attack_attrs['ground_truth']
        # gold_probs = probs[lable_idx_map[gold_label]].prod(0)
        #
        # lable_idx_map.pop(gold_label)
        # other_probs = []
        # for label, idx in lable_idx_map.items():
        #     other_probs.append(probs[idx].prod(0))
        # other_prob = max(other_probs)
        # ratio = gold_probs/other_prob
        # if ratio >= 1:
        #     self.tmp_dict_above[ratio.item()] = transformed_text.tokenizer_input[0]
        #     return True
        # else:
        #     self.tmp_dict_below[ratio.item()] = transformed_text.tokenizer_input[0]
        #     return False

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    def _check_constraint(self, transformed_text, reference_text):
        pass

    def logdebug(self, debug_dict_above, debug_dict_below, reference_text):
        l = [reference_text.tokenizer_input[0], reference_text.tokenizer_input[1], reference_text.attack_attrs['ground_truth']]
        d = {'inp': l}
        d['can'] = {k: debug_dict_above[k] for k in (sorted(debug_dict_above, reverse=False)[:5])}
        d['can'].update({k: debug_dict_below[k] for k in (sorted(debug_dict_below, reverse=True)[:5])})
        try:
            with open(self.file_name, 'a') as f:
                json.dump(d, f, indent=1)
                f.write('\n')
        except:
            pass