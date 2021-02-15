from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import pprint
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from torch.nn.functional import log_softmax
from transformers.utils.dummy_pt_objects import AutoModel
import h5py
import vaex
import xarray as xr
import scipy.special as sp
import xarray_extras.sort as xsort
import zhon.hanzi
import zhon.pinyin
from stopwordsiso import stopwords

config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)

@st.cache(allow_output_mutation=True)
def init():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", **config_dict)
    model = AutoModelWithLMHead.from_pretrained("bert-base-chinese", **config_dict)
    return tokenizer, model

tokenizer, model = init()

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
    token_strs = tokenizer.convert_ids_to_tokens(range(len(bias)))
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


def fill_mask(text):
    ent = 0
    while "[MASK]" in text:
        logits, features, bias = my_model(text)
        logits = word_entropy(logits[0])
        origin = logits.coords["seq_words"]
        mask_locations = logits.coords["seq_words"] == "[MASK]"
        # st.code(stopwords("zh"))
        # st.code(mask_locations)
        logits = logits.drop_sel(word=list(zhon.hanzi.punctuation) + list(tokenizer.all_special_tokens) + ["..."] + list(zhon.pinyin.non_stops) + list(zhon.pinyin.stops), errors='ignore')
        logits = logits.sel(seq=mask_locations)
        # st.code(logits)
        val = logits.min(dim=["seq", "word"])
        # st.code(logits.min(dim=("seq", "word")))
        min_item = logits.where(logits==val, drop=True).squeeze()
        ent += min_item
        # st.code(min_item)
        # st.code(origin)
        text = origin
        text[min_item.coords["seq"]] = min_item.coords["word"]
        # st.code(text)
        text = "".join(str(x) for x in text.data)
        # st.code(text)
    return text, float(ent)

        # min_ent, min_w, min_i = np.inf, None, None
        # for i, w in enumerate(logits.coords["seq_words"]):
        #     seq_i = logits.sel(seq=i)
        #     if w == "[MASK]":
        #         ind = seq_i.argmin(dim="word")
        #         st.code(seq_i, ind)
        #         if seq_i[ind] < min_ent:
        #             min_ent, min_w, min_i = seq_i[ind], ind, i
        # st.code((min_ent, min_w, min_i))



def main():
    with torch.no_grad():
        text = st.text_area("Input sentence:")
        # logits, features, bias = my_model(*text.splitlines())
        # result = logits
        # st.code(result)
        # result.coords["seq_words"]
        # target_text = translate(xsort.argtopk(result, k=7, dim="word"))
        # st.code(target_text[0])
        # ents = pick_words(word_entropy(result))
        # st.code(ents)
        # st.code(pick_words(features))
        st.code(fill_mask(text))

if __name__ == "__main__":
    main()

        

