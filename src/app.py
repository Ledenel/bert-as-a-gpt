import functools
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import scipy.special as sp
import zhon.hanzi
import zhon.pinyin
import zhon.cedict
from random import sample, randint, choice
from stopwordsiso import stopwords

class FormatException(Exception):
    pass


def word_entropy(logits):
    softmaxes = xr.apply_ufunc(lambda t: -sp.log_softmax(t, axis=logits.dims.index("word")) ,logits)
    return softmaxes

def translate(top_indexes):
    word_maps = {v:k for k,v in tokenizer.vocab.items()}
    translated = xr.apply_ufunc(np.vectorize(lambda x: word_maps[x]), top_indexes)
    return translated

def pick_words(logit):
    kws = {}
    kws["seq"] = logit.coords["seq_idx"]
    if "word" in logit.dims:
        kws["word"] = logit.coords["seq_words"]
    self_logits = logit.sel(**kws)
    return self_logits

def my_model(*texts):
    splited_texts = [tokenizer.tokenize(text) for text in texts]
    splited_ids = [tokenizer.convert_tokens_to_ids(x) for x in splited_texts]
    splited_ranges = [list(range(len(x))) for x in splited_texts]
    output = model(torch.tensor(splited_ids), output_hidden_states=True)
    bias = model.cls.predictions.decoder.bias
    token_strs = ["<ERROR%s>" % i if x is None else x for i,x in enumerate(tokenizer.convert_ids_to_tokens(range(len(bias))))]

    logits = xr.DataArray(
        output.logits.detach().numpy(),
        coords={
            "word": token_strs,
            "seq_texts": ("batch", list(texts)),
            "seq_words": (("batch","seq"), splited_texts),
            "seq_word_ids": (("batch","seq"), splited_ids),
            "seq_idx": (("batch","seq"), splited_ranges),
            "seq": splited_ranges[0],
        },
        dims=["batch", "seq", "word"],
    )
    features = xr.DataArray(
        torch.stack(output.hidden_states).detach().numpy(),
        coords={
            "seq_texts": ("batch", list(texts)),
            "seq_words": (("batch","seq"), splited_texts),
            "seq_word_ids": (("batch","seq"), splited_ids),
            "seq_idx": (("batch","seq"), splited_ranges),
        },
        dims=["layers", "batch", "seq", "hidden"],
    )
    bias = xr.DataArray(
        bias.detach().numpy(),
        coords={
            "word": token_strs,
        },
        dims=["word"],
    )
    return logits, features, bias

def each_min(logit):
    for i, w in enumerate(logit.coords["seq_words"]):
        if w == "[MASK]":
            ind = logit.sel(seq=i).idxmin(dim="word")
            yield i, ind

@functools.lru_cache()
def extra_ban():
    tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer.vocab)))
    # banned_words = list(zhon.hanzi.punctuation) + list(tokenizer.all_special_tokens) + ["...", "．"] + list(zhon.pinyin.non_stops) + list(zhon.pinyin.stops) + ["、"]
    # banned_words += [x for x in tokens if x.startswith("#")]
#     banned_words += [x for x in tokens if any(c in set("""
#     あ ア	い イ	う ウ	え エ	お オ
# 	か カ	き キ	く ク	け ケ	こ コ
# 	さ サ	し シ	す ス	せ セ	そ ソ
# 	た タ	ち チ	つ ツ	て テ	と ト
# 	な ナ	に ニ	ぬ ヌ	ね ネ	の ノ
# 	は ハ	ひ ヒ	ふ フ	へ ヘ	ほ ホ
# 	ま マ	み ミ	む ム	め メ	も モ
# 	や ヤ		ゆ ユ		よ ヨ
# 	ら ラ	り リ	る ル	れ レ	ろ ロ
# 	わ ワ				を ヲ
# ん ン
# """) for c in x)]

    banned_words = [x for x in tokens if x not in set(zhon.cedict.all)] + ["、"]
    return banned_words
import torch

# workaround from https://github.com/pytorch/pytorch/issues/46307#issuecomment-708492144
def sparse_mul(a, S):
    return torch.mul(a.expand(S.shape).sparse_mask(S.coalesce()), S)

def torch_solve(func, x, opt=torch.optim.Adam, epoch=24):
    with torch.enable_grad():
        x = torch.tensor(x, requires_grad=True)
        opt = opt(lr=0.1, params=[x])
        for _ in range(epoch):
            opt.zero_grad()
            result = func(x)
            result.backward()
            opt.step()
            # print(x)
        return x

import math
def fill_mask(text, banned_words=(), allowed_words=(), unique=False, top_k=64, soft_unique=False, top_rate=1):
    banned_words = list(banned_words) + extra_ban()
    filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
    special_ids = tokenizer.convert_tokens_to_ids(list(tokenizer.special_tokens_map.values()))
    ent = []
    i = 1
    mask_str = tokenizer.special_tokens_map["mask_token"]
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_str)
    seq_id = None
    with torch.no_grad():
        while "[MASK]" in text:
            filter_ids = tokenizer.convert_tokens_to_ids(banned_words)
            print("processing", text)
            tokenized_batch = tokenizer([text])
            logits = model(**tokenized_batch.convert_to_tensors("pt"))
            logits = logits.logits[0] # pick first item, only one

            """
            temperature = torch.ones((logits.shape[0], 1))
            thr = math.log(0.8)
            def temperature_equation(temp):
                # temp = torch.nn.functional.sigmoid(temp)
                logits_mod = temp * logits
                softmaxed = torch.nn.functional.log_softmax(logits_mod, 1)
                max_item = softmaxed.max(axis=1).values
                delta = torch.clip(max_item - thr, 0)
                return delta.sum()  
            temp = torch_solve(
                temperature_equation,
                temperature
            )
            """

            thr = math.log(0.9)
            temp = torch.clamp(torch.nn.functional.log_softmax(logits, 1).max(axis=1).values / thr, min=0, max=1)
            temp = temp.unsqueeze(-1)
            print(temp.flatten())
            logits = logits * temp

            neg_inf = logits.min() - 10000000 # margin
            print("neg inf", neg_inf)
            mask_location_pt = tokenized_batch.convert_to_tensors("pt").input_ids[0] == mask_token_id
            logits[:, filter_ids] = neg_inf # remove banned words
            logits[:, special_ids] = neg_inf # remove special tokens
            # FIXME: can't remove here, since softmaxed result is equal-probability.
            # logits[~mask_location_pt, :] = neg_inf # remove un-masked words
            topk = torch.topk(logits, k=top_k, dim=1)
            topk_i_ind = topk.indices.clone()
            topk_i_ind[:, :] = torch.arange(topk_i_ind.shape[0]).unsqueeze(-1)
            topk_coo = torch.stack([topk_i_ind.view(-1), topk.indices.view(-1)])
            logits = torch.sparse_coo_tensor(topk_coo, topk.values.view(-1), logits.shape) # clip logits to top-k
            # logits = sparse_mul(temp, logits)
            logits = torch.sparse.log_softmax(logits, 1) # softmax it

            decreased_by_word_pow = torch.bincount(tokenized_batch.convert_to_tensors("pt").input_ids[0], minlength=logits.shape[-1]).unsqueeze(0) + 1

            if soft_unique:
                if seq_id is not None:
                    seq_pow = torch.abs(torch.arange(logits.shape[0]) - seq_id).unsqueeze(-1)
                    decreased_by_word_pow = seq_pow @ decreased_by_word_pow
                logits = sparse_mul(decreased_by_word_pow, logits) # make word with n-count P(word) ^ n, which is equalivant to n * log(P(word))
                logits = torch.sparse.log_softmax(logits, 1) # re-softmax the logits

            ent_index_sr = pd.Series(index=list(logits.coalesce().indices().detach().numpy()), data=logits.coalesce().values().detach().numpy())

            mask_location_pt_where, *_ = torch.where(mask_location_pt)
            ent_index_sr_is_mask = pd.Series(ent_index_sr.index.get_level_values(0)).isin(mask_location_pt_where.detach().numpy()).values
            ent_index_sr = ent_index_sr[ent_index_sr_is_mask]
            top_k_val = ent_index_sr.sort_values(ascending=False)
            if top_rate < 0: 
                exp_item = np.exp(top_k_val)
                exp_item_rate = exp_item / exp_item.max()
                exp_item_mask = exp_item_rate >= 1 - top_rate
                print("top_rate from", len(exp_item_mask), "to", exp_item_mask.sum())
                top_k_val = top_k_val[exp_item_mask]
                
            top_k_val = top_k_val[:top_k]
            top_k_val_item = top_k_val.sample(n=1, weights=np.exp(top_k_val))
            top_k_val_item = list(top_k_val_item.to_dict().items())[0]
            idx_pack, log_p = top_k_val_item
            seq_id, word_id = idx_pack
            word_text = tokenizer.convert_ids_to_tokens(word_id)
            seq_ids_origin = tokenized_batch[0].ids.copy()
            seq_ids_origin[seq_id] = word_id # replace word with generated, before seq id changed
            seq_ids_origin = seq_ids_origin[1:-1] # remove [CLS] and [SEP]
            seq_texts = tokenizer.convert_ids_to_tokens(seq_ids_origin)

            ent.append((
                seq_id,
                i,
                word_text,
                float(np.exp(log_p)))
            )
            if unique:
                banned_words.append(word_text)
            text = "".join(seq_texts)
            # text[min_item.coords["seq"]] = min_item.coords["word"]
            # text = "".join(str(x) for x in text.data)
            i += 1
    ent.sort()
    return text, " ".join('{}{}{:.3}'.format(*x[1:]) for x in ent)

import itertools

def check_partitions(mode, lens, part):
    for mode_value, left, right in zip(mode, part[:-1], part[1:]):
        if sum(lens[left:right]) > mode_value:
            return False
    return True

def partition_indexes(k, sum, allow_zero=False):
    if allow_zero:
        comb = itertools.combinations_with_replacement(range(0, sum+1), k)
    else:
        comb = itertools.combinations(range(1, sum), k)
    return (([0] + list(partitions) + [sum]) for partitions in comb)

def partition_counts(k, sum, allow_zero=False):
    for part in partition_indexes(k, sum, allow_zero=allow_zero):
        yield [right - left for left, right in zip(part[:-1], part[1:])]


config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    # proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)

import re
number_pattern = re.compile("([0-9]+)(\\-([0-9]+))?(\\-([0-9]+))?")

from itertools import permutations
import traceback

import random
def shuffled(l):
    l = l.copy()
    random.shuffle(l)
    return l

def make_sentence(mode_str, keywords, ban_self=False, unique=False, top_k=16, rand=False, **kwargs):
    mode_matched_indexes = [i for matched in number_pattern.finditer(mode_str) for i in matched.span()]
    mode_matched_indexes.append(0)
    mode_matched_indexes.append(len(mode_str))
    mode_matched_indexes = list(set(mode_matched_indexes))
    mode_matched_indexes.sort()
    mode = [mode_str[left:right] for left, right in zip(mode_matched_indexes[:-1], mode_matched_indexes[1:])]
    mode = [x for x in mode if x.strip()]
    mode_matched = [(i, number_pattern.match(x).groups()) for i, x in enumerate(mode) if number_pattern.match(x)]

    mode_match_index, mode_match_nums = zip(*mode_matched)
    mode_mask_min_nums = []
    mode_keyword_max_nums = []
    mode_mask_max_nums = []
    for mode_mask_min, _, mode_keyword_max, _, mode_mask_max in mode_match_nums:
        if mode_keyword_max is None:
            mode_keyword_max = mode_mask_min
        if mode_mask_max is None:
            mode_mask_max = mode_mask_min
        mode_mask_min = int(mode_mask_min)
        mode_keyword_max = int(mode_keyword_max)
        mode_mask_max = int(mode_mask_max)
        if mode_mask_max < mode_mask_min:
            mode_mask_max, mode_mask_min = mode_mask_min, mode_mask_max
        mode_mask_min_nums.append(mode_mask_min)
        mode_keyword_max_nums.append(mode_keyword_max)
        mode_mask_max_nums.append(mode_mask_max)
    assert sum(mode_mask_max_nums) <= 100
    print(mode_mask_min_nums, mode_keyword_max_nums, mode_mask_max_nums)
    #mode = ui
    if rand:
        keywords_empty = ["" for _ in mode_matched]
        keywords_empty[:len(keywords)] = keywords
        # find a solution: greedy match
        mode_matched_sorted = sorted(enumerate(mode_keyword_max_nums), key=lambda t: -t[1])
        keyword_lens = [len(alphanums.sub("#", x)) for x in keywords_empty]
        keyword_lens_sorted = sorted(enumerate(keyword_lens), key=lambda t: -t[1])
        matched = all(kw_max >= kw_len for (_, kw_max), (_, kw_len) in zip(mode_matched_sorted, keyword_lens_sorted))
        if not matched:
            keyword_groups = []
        else:
            for _ in range(100):
                keyword_lens_sorted_new = keyword_lens_sorted.copy()

                # shuffle
                i, j = random.sample(list(range(len(keyword_lens_sorted_new))), 2)
                keyword_lens_sorted_new[i], keyword_lens_sorted_new[j] = keyword_lens_sorted_new[j], keyword_lens_sorted_new[i]
                if all(kw_max >= kw_len for (_, kw_max), (_, kw_len) in zip(mode_matched_sorted, keyword_lens_sorted_new)):
                    keyword_lens_sorted = keyword_lens_sorted_new
        final_group = [(group_id, keywords_empty[kw_id]) for (group_id, _),  (kw_id, _) in zip(mode_matched_sorted, keyword_lens_sorted)]
        final_group.sort()
        keyword_groups = [[kw for _, kw in final_group]]
    else:
        keyword_groups = [keywords]
    print(keyword_groups)
    gen_templates = [t for kw in keyword_groups for t in generate_templates(kw, mode_keyword_max_nums, mode_mask_max_nums, mode_mask_min_nums, mode, mode_match_index)]
    print(gen_templates)
    try:
        word_template = choice(gen_templates)
    except IndexError as e:
        raise FormatException("无法按照给定的约束和关键词填入模板。请尝试缩减关键词字数，拆分关键词，或者使用rand选项来允许无序填入模板。")#traceback.format_exc())
    if ban_self:
        extra = tokenizer.tokenize(word_template)
    else:
        extra = []
    word_template = word_template
    return fill_mask(word_template, banned_words=extra, unique=unique, top_k=top_k, **kwargs)

import re
alphanums = re.compile("[a-zA-Z0-9]+")

def generate_templates(keywords, mode_keyword_max_nums, mode_mask_max_nums, mode_mask_min_nums, mode, mode_match_index):
    keyword_lens = [len(alphanums.sub("#", x)) for x in keywords]
    valid_parts = list(partition_indexes(len(mode_keyword_max_nums) - 1, len(keywords), allow_zero=False))
    print(valid_parts)
    if not valid_parts:
        assert len(mode_keyword_max_nums) - len(keywords) > 0
        # not enough keyword to fill in mode, fill left-to-right by default
        valid_parts = list(range(len(keywords)))
        valid_parts.extend([valid_parts[-1] + 1] * (len(mode_keyword_max_nums) - len(keywords) + 1))
        valid_parts = [valid_parts]
        print(valid_parts)
    valid_parts = [x for x in valid_parts if check_partitions(mode_keyword_max_nums, keyword_lens, x)]
    gen_templates = []
    for part in valid_parts:
        all_gen = []
        for max_mode_value, min_mode_value, left, right in zip(mode_mask_max_nums, mode_mask_min_nums, part[:-1], part[1:]):
            max_remain_mask_count = max(max_mode_value - sum(keyword_lens[left:right]), 0)
            min_remain_mask_count = max(min_mode_value - sum(keyword_lens[left:right]), 0)
            current = keywords[left:right]
            for _ in range(randint(min_remain_mask_count, max_remain_mask_count)):
                spin = randint(0, len(current))
                current[spin:spin] = ["[MASK]"]
            all_gen.append("".join(current).replace("?", "[MASK]").replace("？", "[MASK]"))
        template = mode.copy()
        for index, part in zip(mode_match_index, all_gen):
            template[index] = part
        gen_templates.append("".join(template))
    return gen_templates

# @st.cache(allow_output_mutation=True)

def main():
    from init import init
    import streamlit as st
    global tokenizer, model
    tokenizer, model = st.cache(allow_output_mutation=True)(init)()
    with torch.no_grad():
        text = st.text_area("Input keywords:")
        if st.checkbox("hint mode?"):
            text = text.strip()
            text = text.replace(":", "：")
            text = text.replace(";", "；")
            text = text.replace(",", "，")
            text = text.replace(".", "。")
            text = text.replace("?", "？")
            if text[-1] not in set("？。，：；") and randint(0,1) < 1:
                text = text + choice("？。，：；")
            template = "[MASK]" * 2 + choice(list("！"))
            # st.code(set("".join(stopwords("zh"))))
            text, score = fill_mask(text+template, allowed_words=tokenizer.tokenize(text))
            st.code((text, score))
        else:
            banned_self = st.checkbox("Banned self?")
            mode = [5,7,5]
            keywords = [x for x in text.split() if x.strip() != ""]
            st.code([keywords, make_sentence(mode, keywords, ban_self=banned_self)])

if __name__ == "__main__":
    main()
else:
    from init import tokenizer, model

extra_ban()

from flask import Flask
app = Flask(__name__)
from flask import request

@app.route('/', methods=['GET'])
def make_sentences_serve():
    text, score = make_sentence([5,7,5], [x for x in request.args.get('keywords', '').split(",")])
    return "%s %s" % (text, score)

import os
if "FLASK_DEBUG" in os.environ and str(os.environ["FLASK_DEBUG"]) == "1":
    pass
else:
    @app.errorhandler(Exception)
    def exception_handler(e):
        return str(e), 400

# @app.errorhandler(500)
# def exception_handler(e):
#     return str(e), 400

@app.route('/no_self', methods=['GET'])
def make_sentences_no_self():
    top_k = int(request.args.get('rand_top', '16'))
    template = request.args.get('template', "5，7，5。")
    unique_mode = request.args.get('mode', "soft")
    top_rate = float(request.args.get('top_rate', "0.99"))
    rand = {"on": True, "off": False}[request.args.get('rand', "off")]
    
    ban_self = unique_mode == "hard"
    unique = unique_mode == "hard"
    soft_unique = unique_mode == "soft"

    text, score = make_sentence(
        template, [x for x in request.args.get('keywords', '').split(",")],
         ban_self=ban_self,
         unique=unique, 
         top_k=top_k, 
         soft_unique=soft_unique, 
         top_rate=top_rate, 
         rand=rand,
    )
    return "%s %s" % (text, score)
        
@app.route('/hint', methods=['GET'])
def make_hint():
    text = request.args.get('question', '')
    if text[-1] not in "？，：；“”":
        text = text + "？"
    template = "[MASK]" * randint(2,3) + choice("。！")
    if  text[-1] not in "”" and randint(0, 1) < 1:
        text = f"“{text}”"
        template = f"“{template}”"

    text, score = fill_mask(text+template, banned_words=tokenizer.tokenize(text))
    return "%s %s" % (text, score)
