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
    self_logits = logit.sel(seq=logit.coords["seq_idx"], word=logit.coords["seq_words"])
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
        },
        dims=["batch", "seq", "word"],
    )
    # features = xr.DataArray(
    #     torch.stack(output.hidden_states).detach().numpy(),
    #     coords={
    #         "seq_tokens": ("seq", splited_text),
    #         "seq_ids": ("seq", splited_ids),
    #     },
    #     dims=["layers", "batch", "seq", "hidden"],
    # )
    # bias = xr.DataArray(
    #     bias.detach().numpy(),
    #     coords={
    #         "word": token_strs,
    #     },
    #     dims=["word"],
    # )
    return logits

with torch.no_grad():
    text = st.text_area("Input sentence:")
    result = my_model(*text.splitlines())
    st.code(result)
    result.coords["seq_words"]
    target_text = translate(xsort.argtopk(result, k=7, dim="word"))
    st.code(target_text[0])
    ents = pick_words(word_entropy(result))
    st.code(ents)



        
