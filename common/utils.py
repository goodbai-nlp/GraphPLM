# coding:utf-8
import os
import re
import json
import math
import torch
import itertools
import random
import numpy as np
from pathlib import Path
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer, EvalPrediction
from typing import Callable, Dict, Iterable, List, Tuple, Union
from collections import Counter
from common.constant import ROUGE_KEYS
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Standard MLM task: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_nodes_edges(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Mask AMR nodes and edges, ignore other special tokens """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    mask_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)                         # special tokens
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer.amr_bos_token_id)                 # don't mask AMR_bos_token
        probability_matrix.masked_fill_(special_token_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer._convert_token_to_id("Ġ("))       # don't mask left bracket
        probability_matrix.masked_fill_(special_token_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer._convert_token_to_id("Ġ)"))       # don't mask right bracket
        probability_matrix.masked_fill_(special_token_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100                                                      # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    mask_inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    mask_inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return mask_inputs, labels


def sequence_infilling(tokenizer, inp, mlm_prob=0.15):
    token_length = len([int(itm != tokenizer.pad_token_id) for itm in inp])
    masking_length = math.floor(token_length * mlm_prob)
    masked_length = 1
    masked_inputs = inp.clone().tolist()
    while masked_length < masking_length:
        span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
        start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
        masked_inputs = masked_inputs[:start_index] + [tokenizer.mask_token_id] + masked_inputs[start_index + span_length:]
        token_length -= span_length - 1
        masked_length += span_length
    return torch.LongTensor(masked_inputs)


def batch_infilling(inp, tokenizer, mlm_prob=0.15):
    res = []
    for sents in inp:
        res.append(sequence_infilling(tokenizer, sents, mlm_prob=mlm_prob))
    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def save_dummy_batch(batch, tokenizer, output_dir):
    print("Saving dummy inputs...")
    json_out_path = open(output_dir + "/dummy_input.json", "w", encoding="utf-8")
    ith_dict = {}
    for k, v in batch.items():
        if "_ids" in k and v is not None:
            ith_dict[k] = str(v.tolist())
            ith_dict[k.replace("ids", "tokens")] = tokenizer.batch_decode(v.tolist(), clean_up_tokenization_spaces=False)
        elif "labels" in k:
            ith_dict[k] = str(v.tolist())
            label_data_new = [[idx if idx != -100 else tokenizer.pad_token_id for idx in ith_label] for ith_label in v.tolist()]
            ith_dict[k+"_tokens"] = tokenizer.batch_decode(label_data_new, clean_up_tokenization_spaces=False)
        else:
            print(f"Skiping {k}...")
    json.dump(ith_dict, json_out_path, indent=4)


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # nll_loss = nll_loss.sum()                  
    # smooth_loss = smooth_loss.sum()
    nll_loss = nll_loss.mean()                   
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def smart_emb_init(tokenizer, model):
    print("Initializing AMR Vocab according to similar tokens ...")
    INIT = "Ġ"
    for tok, idx in tokenizer.encoder.items():
        tok = tok.lstrip(INIT)

        if idx < tokenizer.old_enc_size:
            continue

        elif tok.startswith("<pointer:") and tok.endswith(">"):
            tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

        elif tok.startswith("<"):
            continue

        elif tok.startswith(":"):

            if tok.startswith(":op"):
                tok_split = ["relation", "operator", str(int(tok[3:]))]

            elif tok.startswith(":snt"):
                tok_split = ["relation", "sentence", str(int(tok[4:]))]

            elif tok.startswith(":ARG"):
                tok_split = ["relation", "argument", str(int(tok[4:]))]

            else:
                tok_split = ["relation"] + tok.lstrip(":").split("-")

        else:
            tok_split = tok.split("-")

        tok_split_ = tok_split
        tok_split = []
        for s in tok_split_:
            s_ = INIT + s
            if s_ in tokenizer.encoder:
                # print(f"{s_} in tokenizer vocabulary")
                tok_split.append(s_)
            elif s in tokenizer.encoder:
                tok_split.append(s)
            else:
                tok_split.extend(s.split("_"))  #

        vecs = []
        for s in tok_split:
            idx_split = tokenizer.encoder.get(s, -1)
            if idx_split > -1:
                vec_split = model.model.shared.weight.data[idx_split].clone()
                vecs.append(vec_split)

        if vecs:
            vec = torch.stack(vecs, 0).mean(0)
            noise = torch.empty_like(vec)
            noise.uniform_(-0.1, +0.1)
            model.model.shared.weight.data[idx] = vec + noise

    return model