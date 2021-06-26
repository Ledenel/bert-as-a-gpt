import functools
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import numpy as np
import xarray as xr
import scipy.special as sp
import zhon.hanzi
import zhon.pinyin
import zhon.cedict
from random import sample, randint, choice
from stopwordsiso import stopwords

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

def fill_mask(text, banned_words=(), allowed_words=(), unique=False):
    banned_words = list(banned_words) + extra_ban()
    ent = []
    i = 1
    while "[MASK]" in text:
        logits, features, bias = my_model(text)
        logits = word_entropy(logits[0])
        origin = logits.coords["seq_words"]
        mask_locations = logits.coords["seq_words"] == "[MASK]"
        if allowed_words:
            logits = logits.sel(word=list(allowed_words))
        else:
            logits = logits.drop_sel(word=banned_words, errors='ignore')
        logits = logits.sel(seq=mask_locations)
        val = logits.min(dim=["seq", "word"])
        min_item = logits.where(logits==val, drop=True).squeeze()
        ent.append((
            min_item.coords["seq"].data,
            i,
            min_item.coords["word"].data,
            float(min_item))
        )
        if unique:
            banned_words.append(str(min_item.coords["word"].data))
        text = origin
        text[min_item.coords["seq"]] = min_item.coords["word"]
        text = "".join(str(x) for x in text.data)
        i += 1
    ent.sort()
    return text, " ".join('{}{}{:.2}'.format(*x[1:]) for x in ent)

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

def make_sentence(mode, keywords, ban_self=False, unique=False):
    keyword_lens = [len(x) for x in keywords]
    valid_parts = list(partition_indexes(len(mode) - 1, len(keywords), allow_zero=True))
    valid_parts = [x for x in valid_parts if check_partitions(mode, keyword_lens, x)]
    gen_templates = []
    for part in valid_parts:
        all_gen = []
        for mode_value, left, right in zip(mode, part[:-1], part[1:]):
            mask_spin_counts = right - left + 1
            remain_mask_count = mode_value - sum(keyword_lens[left:right])
            current = keywords[left:right]
            for _ in range(remain_mask_count):
                spin = randint(0, len(current))
                current[spin:spin] = ["[MASK]"]
            all_gen.append("".join(current))
        gen_templates.append("[CLS]" + "，".join(all_gen) + "。[SEP]")
    extra = []
    word_template = choice(gen_templates)
    if ban_self:
        extra = tokenizer.tokenize(word_template)
    return fill_mask(word_template, banned_words=extra, unique=unique)

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

@app.route('/no_self', methods=['GET'])
def make_sentences_no_self():
    text, score = make_sentence([5,7,5], [x for x in request.args.get('keywords', '').split(",")], ban_self=True, unique=True)
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
