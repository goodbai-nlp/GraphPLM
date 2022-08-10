# coding:utf-8
import os
import torch
import inspect
import importlib
import pytorch_lightning as pl
from copy import deepcopy
from torch.utils.data import Dataset
from contextlib import contextmanager
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union, Any, Tuple
from common.utils import shift_tokens_right


class Seq2SeqDataSet(Dataset):
    def __init__(
        self, tokenizer, args, model_args
    ):
        super().__init__()
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.src_prefix = args.source_prefix
        self.tgt_prefix = args.target_prefix
        self.cache_dir = model_args.cache_dir
        self.tokenizer = tokenizer

        self.max_src_length = min(args.max_source_length, self.tokenizer.model_max_length)
        self.max_tgt_length = min(args.max_target_length, self.tokenizer.model_max_length)

        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # exit()
        self.datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py",
            data_files=data_files,
            keep_in_memory=False,
        )
        column_names = self.datasets["train"].column_names
        print("datasets:", self.datasets)
        print("colums:", column_names)

    def tokenize_function(self, examples):
        sent = examples["src"]  # text tokens
        amr = examples["tgt"]  # amr tokens

        sent = [self.src_prefix + inp for inp in sent]
        amr = [self.tgt_prefix + inp for inp in amr]

        sent_ids = self.tokenizer(
            sent, max_length=self.max_src_length, padding=False, truncation=True
        )
        amr_output = self.tokenizer(
            amr, max_length=self.max_tgt_length, padding=False, truncation=True
        )
        
        amr_output["input_ids"] = [[self.tokenizer.amr_bos_token_id] + itm[1:-1] + [self.tokenizer.amr_eos_token_id] for itm in amr_output["input_ids"]]
        amr_output["sent_ids"] = sent_ids["input_ids"]
        return amr_output


def padding_func(features, padding_side="right", pad_token_id=1, key="label", pad_to_multiple_of=1, max_length=None):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    if pad_to_multiple_of > 1:
        if max_length is not None:
            max_label_length = min(max_length,
                (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            )
        else:
            max_label_length = (max_label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
            
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    mlm_probability: float = 0.15

    def __call__(self, features):
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="sent_ids",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # features.pop("attention_mask")
        features.pop("sent_ids")                                            # todo: consider sentences during pre-training
        features["labels"] = features["input_ids"].clone()[:, 1:]
        features["decoder_input_ids"] = features["input_ids"].clone()[:, :-1]
        features["input_ids"], _ = self.torch_mask_tokens(
            features["input_ids"], special_tokens_mask=None
        )
        return features

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # if special_tokens_mask is None:                                   # todo
        #     special_tokens_mask = [
        #         self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        #     ]
        #     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # else:
        #     special_tokens_mask = special_tokens_mask.bool()

        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
            special_token_mask = labels.eq(self.tokenizer.amr_bos_token_id)                # don't mask AMR_bos_token
            probability_matrix.masked_fill_(special_token_mask, value=0.0)
            special_token_mask = labels.eq(self.tokenizer.amr_eos_token_id)                # don't mask AMR_eos_token
            probability_matrix.masked_fill_(special_token_mask, value=0.0)
            # special_token_mask = labels.eq(self.tokenizer._convert_token_to_id("Ġ("))       # don't mask left bracket
            # probability_matrix.masked_fill_(special_token_mask, value=0.0)
            # special_token_mask = labels.eq(self.tokenizer._convert_token_to_id("Ġ)"))       # don't mask right bracket
            # probability_matrix.masked_fill_(special_token_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels