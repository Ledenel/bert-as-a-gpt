from altair.vegalite.v4.api import sequence
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import pprint
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from torch.nn.functional import log_softmax

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

with torch.no_grad():
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
        df_logits = mask_token_logits.detach().numpy()
        actual_words = sequence_input[0][1:-1]
        df_logits = df_logits.T[actual_words].T
        df_logits = pd.DataFrame(df_logits, columns=tokenizer.convert_ids_to_tokens(actual_words))
        st.write(df_logits)
        # st.plotly_chart(go.Figure(data=go.Heatmap(z=df_logits)))
        
        # st.line_chart({

        # })
        
