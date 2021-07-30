import functools
from transformers import AutoModelForMaskedLM, AutoTokenizer

config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    # proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)

import os

# @st.cache(allow_output_mutation=True)
def init():
    root_dir = "./model"
    model_str = "nghuyong/ernie-1.0"
    joined_path = os.path.join(root_dir, model_str)
    if not os.path.exists(joined_path):
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0", **config_dict)
        model = AutoModelForMaskedLM.from_pretrained("nghuyong/ernie-1.0", **config_dict)
        os.makedirs(joined_path, exist_ok=True)
        tokenizer.save_pretrained(joined_path)
        model.save_pretrained(joined_path)
        os.system("rm -rf cache")
    tokenizer = AutoTokenizer.from_pretrained(joined_path)
    model = AutoModelForMaskedLM.from_pretrained(joined_path)
    return tokenizer, model
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")
    # model = TFAutoModelWithLMHead.from_pretrained("t5-small")
    # You can then save them locally via:

    # tokenizer.save_pretrained('./local_model_directory/')
    # model.save_pretrained('./local_model_directory/')
    # And then simply load from the directory:

    # tokenizer = AutoTokenizer.from_pretrained('./local_model_directory/')
    # model = TFAutoModelWithLMHead.from_pretrained('./local_model_directory/')
    
tokenizer, model = init()

