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

with torch.no_grad():
    text = st.text_input("Input sentence:")
    splited_text = tokenizer.tokenize(text)
    st.write(splited_text)
    splited_ids = tokenizer.convert_tokens_to_ids(splited_text)
    st.write(splited_ids)
    output = model(torch.tensor(splited_ids).unsqueeze(0), output_hidden_states=True)
    st.write(output)
    bias = model.cls.predictions.decoder.bias
    st.write(bias)
    token_strs = tokenizer.convert_ids_to_tokens(range(len(bias)))
    # st.write(token_strs)
    logits = xr.DataArray(
        output.logits.detach().numpy(),
        coords={
            "word": token_strs,
            "seq_tokens": ("seq", splited_text),
            "seq_ids": ("seq", splited_ids),
        },
        dims=["batch", "seq", "word"],
    )
    features = xr.DataArray(
        torch.stack(output.hidden_states).detach().numpy(),
        coords={
            "seq_tokens": ("seq", splited_text),
            "seq_ids": ("seq", splited_ids),
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
    # logits
    # features
    # bias
    # logits - bias
    # sorted_logits = logits.argsort(axis="word", ascending=False)
    # sorted_logits
    # ids = tokenizer.convert_ids_to_tokens(sorted_logits.sel(word="123").squeeze("batch"))
    tops = xsort.argtopk(logits, k=10, dim="word")
    word_maps = {v:k for k,v in tokenizer.vocab.items()}
    tops
    translated = xr.apply_ufunc(np.vectorize(lambda x: word_maps[x]), tops)
    translated
    # self_logits = xr.apply_ufunc(lambda t: np.diagonal(t, axis1=logits.dims.index('seq'), axis2=logits.dims.index('word')), logits.sel(word=splited_text))
    softmaxes = xr.apply_ufunc(lambda t: -sp.log_softmax(t, axis=logits.dims.index("word")) ,logits)
    softmaxes.loc[dict(word=tokenizer.all_special_tokens)] = 0
    self_logits = softmaxes.sel(seq=xr.DataArray(range(len(splited_text)), dims=["logits"]), word=xr.DataArray(splited_text, dims=["logits"]))
    self_logits
    most_message = xsort.argtopk(self_logits, k=3, dim="logits")
    most_message
    # self_logits.isel(most_message)
    # ids = tops.squeeze("batch").apply(tokenizer.convert_ids_to_tokens)


        
