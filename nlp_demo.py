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

config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)

def print_token(x):
    return f"{tokenizer.decode([x])}[{x}]"

@st.cache(allow_output_mutation=True)
def init():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", **config_dict)
    model = AutoModelWithLMHead.from_pretrained("bert-base-chinese", **config_dict)
    return tokenizer, model

tokenizer, model = init()

sequence = st.text_input("Sentence", value=f"生活的真谛是美")

import matplotlib.pyplot as plt


def words_score(logits, **kwargs):
    key, actual_words = list(kwargs.items())[0]
    origin = pd.Series(logits[(range(len(actual_words)), actual_words)], name=f"{key}")
    origin_words = pd.Series(tokenizer.convert_ids_to_tokens(actual_words), name=f"{key}_words")
    return origin, origin_words

def word_embeddings(tokenizer: AutoTokenizer, model: AutoModelWithLMHead):
    s = pd.Series(tokenizer.vocab)
    id_to_word = pd.Series(s.index.values, index=s.values, name="word")
    id_to_word.sort_index(inplace=True)
    assert id_to_word.unique()[-1] == len(id_to_word) - 1
    first_emb_layer = model.bert.embeddings.word_embeddings
    st.text(pprint.pformat(first_emb_layer))
    last_emb_layer = model.cls.predictions.decoder
    st.text(pprint.pformat(last_emb_layer))

    return None
    

with torch.no_grad():
    s = pd.Series(tokenizer.vocab)
    id_to_word = pd.Series(s.index.values, index=s.values, name="word")
    id_to_word.sort_index(inplace=True)
    assert id_to_word.index[-1] == len(id_to_word.unique()) - 1
    first_emb_layer = model.bert.embeddings.word_embeddings.weight
    st.text(pprint.pformat(first_emb_layer.shape))
    last_emb_layer = model.cls.predictions.decoder
    bias_layer = last_emb_layer.bias
    last_emb_layer = last_emb_layer.weight
    st.text(pprint.pformat(last_emb_layer))
    first_layer_map = pd.Series(list(first_emb_layer.detach().numpy()), index=id_to_word, name="first_emb")
    last_layer_map = pd.Series(list(last_emb_layer.detach().numpy()), index=id_to_word, name="last_emb")
    bias_map = pd.Series(bias_layer.detach().numpy(), index=id_to_word, name="bias_value")
    st.write(first_layer_map[list("生活的真谛是爱")])
    st.write(last_layer_map[list("生活的真谛是爱")])
    st.write((first_layer_map-last_layer_map).apply(lambda x:np.sqrt(np.sum(x**2)))[list("生活的真谛是爱")])
    st.write(bias_map[list("生活的真谛是爱")])
    
    sequence_input = tokenizer.encode(sequence, return_tensors="pt")
    mask_token_index, real_token = list(enumerate(sequence_input[0]))[1]
    if tokenizer.convert_ids_to_tokens([real_token])[0] not in tokenizer.all_special_tokens:
        masked_input = sequence_input.clone()
        masked_input[0, mask_token_index] = tokenizer.mask_token_id
        print(tokenizer.decode(masked_input[0]))
        print(tokenizer.convert_ids_to_tokens([real_token])[0])
        token_logits = model(masked_input).logits
        print(token_logits[0, :, :].shape)
        mask_token_logits = -log_softmax(token_logits[0, :, :], dim=1)
        print(mask_token_logits.shape)
        logits = mask_token_logits.detach().numpy()
        df_logits = pd.DataFrame(logits)
        s = pd.Series(tokenizer.vocab)
        r2 = pd.Series(s.index.values, index=s.values, name="word")
        df_logits = df_logits.T.join(r2)
        st.dataframe(df_logits.describe().T.style.background_gradient())
        df_stats = pd.DataFrame(
            words_score(logits, origin=sequence_input[0])
            + words_score(logits, masked=masked_input[0])
            + words_score(logits, max=list(np.argmin(logits, axis=1)))
        ).T
        st.dataframe(df_stats.style.format("{:.4}"))
        # st.plotly_chart(go.Figure(data=go.Heatmap(z=df_logits)))
        
        # st.line_chart({

        # })
        
