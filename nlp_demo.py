from altair.vegalite.v4.api import sequence
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import pprint
import streamlit as st
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

input = tokenizer.encode(sequence, return_tensors="pt")
for mask_token_index, real_token in enumerate(input[0]):
    if tokenizer.convert_ids_to_tokens([real_token])[0] not in tokenizer.all_special_tokens:
        masked_input = input.clone()
        # masked_input[0, mask_token_index] = tokenizer.mask_token_id
        print(tokenizer.decode(masked_input[0]))
        print(tokenizer.convert_ids_to_tokens([real_token])[0])
        token_logits = model(masked_input).logits
        mask_token_logits = log_softmax(token_logits[0, mask_token_index, :])
        top_5_tokens = torch.topk(mask_token_logits, 5).indices.tolist()
        st.text(
            ",".join(f"{print_token(x)}:{float(mask_token_logits[x])}" for x in [real_token] + top_5_tokens)
        )
    
