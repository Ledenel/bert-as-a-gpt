from transformers import AutoModelWithLMHead, AutoTokenizer
import os
import torch
import time
import pprint
time.sleep(10)
config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", **config_dict)
model = AutoModelWithLMHead.from_pretrained("bert-base-chinese", **config_dict)

sequence = f"生活的真谛是美。"

time.sleep(10)
input: torch.Tensor = tokenizer.encode(sequence, return_tensors="pt")

def print_token(x):
    return f"{tokenizer.decode([x])}[{x}]"

pprint()
sequence = f"生活的真谛是美。"
from pprint import pprint
input = tokenizer.encode(sequence, return_tensors="pt")
for mask_token_index, real_token in enumerate(input[0]):
    time.sleep(10)
    masked_input = input.clone()
    masked_input[0, mask_token_index] = tokenizer.mask_token_id
    print(tokenizer.decode(masked_input[0]))
    token_logits = model(masked_input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5).indices.tolist()
    pprint([(print_token(x),float(mask_token_logits[x])) for x in [real_token] + top_5_tokens])
