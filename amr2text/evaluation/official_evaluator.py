# coding:utf-8
import sys
from collections import Counter
import math
sys.path.append('../')


def calc_f1(data):
    """
    calc_f1
    """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in data:
        # golden_response = "".join(golden_response).decode("utf8")
        # response = "".join(response).decode("utf8")
        golden_response = "".join(golden_response)
        response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1 = 2 * p * r / (p + r)
    return f1


def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i: (i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0]  # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        count(pred_tokens, gold_tokens, ngram, result)
    cover_rate = result[0] / result[1]
    # print(cover_rate, str(ngram) + "-gram", result[0], result[1])
    return cover_rate


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        c_count += len(pred_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    cover_rate3 = calc_cover_rate(pair_list, 3)
    cover_rate4 = calc_cover_rate(pair_list, 4)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp(
            (math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3
        )
    if cover_rate4 > 0:
        bleu4 = bp * math.exp(
            (math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3) + math.log(cover_rate4)) / 4
        )
    # print('BP', bp)
    return [bleu1, bleu2, bleu3, bleu4]


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
        # if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    return [distinct1, distinct2]
