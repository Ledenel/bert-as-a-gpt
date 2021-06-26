import functools
from transformers import AutoModelForMaskedLM, AutoTokenizer

config_dict = dict(
    cache_dir="cache",
    # force_download=True,
    # resume_download=True,
    # proxies={'http': os.environ["HTTP_PROXY"], 'https': os.environ["HTTPS_PROXY"]}
)

# @st.cache(allow_output_mutation=True)
def init():
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", **config_dict)
    model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext", **config_dict)
    return tokenizer, model

tokenizer, model = init()

