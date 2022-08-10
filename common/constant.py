# coding:utf-8
import os
import json
from tokenize import Special
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5Model,
    T5ForConditionalGeneration,
)

# amr_tokens = ["dummy", "multi-sentence", ':snt1', "<sep>", "<url>", ':wiki', ':arg0-of', ':arg1-of', ':arg2-of', ':arg3-of', ':arg4-of', ':arg5-of', ':arg5', ':path-of', ':source-of', ':age-of', ':part-of',  ':location-of', ':example-of', ':subevent-of', ':instrument-of', ':quant-of', ':medium-of', ':op1-of', ':destination-of', ':ord-of', ':purpose-of', ':duration-of', ':accompanier-of', ':time-of', 
#                     ':age', ':duration', ':year', ':mod', ':manner-of',
#                     ':snt7', ':year2', ':op5', ':subset-of', ':dayperiod', ':quant', ':season', ':subevent',
#                     ':op9', ':accompanier', ':op6', ':li', ':direction-of', ':domain-of', ':op13', ':op11', ':op4',
#                     ':condition', ':op16', ':condition-of', ':arg8', ':domain', ':time', ':weekday', ':arg2',
#                     ':poss', ':beneficiary-of', ':prep-by', ':snt2', ':prep-in', ':snt8', ':concession-of',
#                     ':topic-of', ':scale', ':snt6', ':arg3', ':prep-for', ':medium', ':op2', ':prep-on',
#                     ':beneficiary', ':snt11', ':op7', ':prep-as', ':frequency-of', ':arg7', ':unit',
#                     ':op1', ':path', ':value', ':degree-of', ':direction', ':poss-of', ':ord', ':month', ':op10',
#                     ':quarter', ':op14', ':prep-under', ':snt3', ':prep-against', ':arg6', ':location',
#                     ':destination', ':consist-of', ':purpose', ':degree', ':extent-of', ':extent',
#                     ':op8', ':conj-as-if', ':prep-from', ':snt10', ':snt9',
#                     ':topic', ':calendar', ':prep-at', ':polite',
#                     ':example', ':prep-out-of', ':day', ':name-of', ':prep-amid', ':prep-into', ':concession',
#                     ':part', ':arg9', ':arg0', ':op19', ':century', ':prep-among',
#                     ':instrument', ':source', ':op17', ':prep-with', ':compared-to',
#                     ':prep-in-addition-to', ':snt5', ':frequency',
#                     ':timezone', ':op3', ':prep-toward', ':prep-on-behalf-of', ':prep-without',
#                     ':name', ':op15', ':prep-along-with', ':arg4', ':mode', ':prep-to', ':decade',
#                     ':op12', ':polarity', ':range', ':snt4', ':P',
#                     ':manner', ':op18', ':op20', ':era', ':arg1']

# special_tokens = ["dummy", "multi-sentence", ':snt1', "<sep>", "<url>", ':wiki', ':ARG0-of', ':ARG1-of', ':ARG2-of', ':ARG3-of', ':ARG4-of', ':ARG5-of', ':ARG5', ':path-of', ':source-of', ':age-of', ':part-of',  ':location-of', ':example-of', ':subevent-of', ':instrument-of', ':quant-of', ':medium-of', ':op1-of', ':destination-of', ':ord-of', ':purpose-of', ':duration-of', ':accompanier-of', ':time-of', 
#                           ':age', ':duration', ':year', ':mod', ':manner-of',
#                           ':snt7', ':year2', ':op5', ':subset-of', ':dayperiod', ':quant', ':season', ':subevent',
#                           ':op9', ':accompanier', ':op6', ':li', ':direction-of', ':domain-of', ':op13', ':op11', ':op4',
#                           ':condition', ':op16', ':condition-of', ':ARG8', ':domain', ':time', ':weekday', ':ARG2',
#                           ':poss', ':beneficiary-of', ':prep-by', ':snt2', ':prep-in', ':snt8', ':concession-of',
#                           ':topic-of', ':scale', ':snt6', ':ARG3', ':prep-for', ':medium', ':op2', ':prep-on',
#                           ':beneficiary', ':snt11', ':op7', ':prep-as', ':frequency-of', ':ARG7', ':unit',
#                           ':op1', ':path', ':value', ':degree-of', ':direction', ':poss-of', ':ord', ':month', ':op10',
#                           ':quarter', ':op14', ':prep-under', ':snt3', ':prep-against', ':ARG6', ':location',
#                           ':destination', ':consist-of', ':purpose', ':degree', ':extent-of', ':extent',
#                           ':op8', ':conj-as-if', ':prep-from', ':snt10', ':snt9',
#                           ':topic', ':calendar', ':prep-at', ':polite',
#                           ':example', ':prep-out-of', ':day', ':name-of', ':prep-amid', ':prep-into', ':concession',
#                           ':part', ':ARG9', ':ARG0',  ':op19', ':century', ':prep-among',
#                           ':instrument', ':source', ':op17',  ':prep-with', ':compared-to',
#                           ':prep-in-addition-to',  ':snt5', ':frequency',
#                           ':timezone', ':op3', ':prep-toward',  ':prep-on-behalf-of', ':prep-without',
#                           ':name', ':op15', ':prep-along-with', ':ARG4', ':mode', ':prep-to', ':decade', 
#                           ':op12', ':polarity', ':range', ':snt4', ':P', 
#                           ':manner', ':op18', ':op20', ':era', ':ARG1', 'G2T']

# sent_tokens = []
# max_turn = 32
# for idx in range(max_turn):         # ma
#     amr_tokens.append(f"speaker{idx+1}")
#     sent_tokens.append(f"speaker{idx+1}")
#     amr_tokens.append(f":speaker{idx+1}")
#     amr_tokens.append(f"u<{idx+1}>")
#     sent_tokens.append(f"u<{idx+1}>")

# special_tokens = []
raw_special_tokens = json.load(open(f"{os.path.dirname(__file__)}/addition-tokens.json", 'r', encoding="utf-8"))
special_tokens = [itm.lstrip("Ġ") for itm in raw_special_tokens]

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

arg_to_tokenizer = {
    "AutoTokenizer": AutoTokenizer,
    "BartTokenizer": BartTokenizer,
    "T5Tokenizer": T5Tokenizer,
}
arg_to_plm_model = {
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    "BartForConditionalGeneration": BartForConditionalGeneration,
    "T5Model": T5Model,
    "T5ForConditionalGeneration": T5ForConditionalGeneration,
}