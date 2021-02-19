from networkx.readwrite.edgelist import read_weighted_edgelist
from numpy.lib.function_base import iterable
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
import bellmanford as bf
import plotly.figure_factory as ff
import h5py
import vaex

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


import matplotlib.pyplot as plt


def words_score(logits, **kwargs):
    key, actual_words = list(kwargs.items())[0]
    origin = pd.Series(logits[(range(len(actual_words)), actual_words)], name=f"{key}")
    origin_words = pd.Series(tokenizer.convert_ids_to_tokens(actual_words), name=f"{key}_words")
    return origin, origin_words

def tokenizer_id_array(tokenizer):
    s = pd.Series(tokenizer.vocab)
    id_to_word = pd.Series(s.index.values, index=s.values, name="word")
    id_to_word.sort_index(inplace=True)

def word_embeddings(tokenizer: AutoTokenizer, model: AutoModelWithLMHead):
    id_to_word = tokenizer_id_array(tokenizer)
    assert id_to_word.unique()[-1] == len(id_to_word) - 1
    first_emb_layer = model.bert.embeddings.word_embeddings
    st.text(pprint.pformat(first_emb_layer))
    last_emb_layer = model.cls.predictions.decoder
    st.text(pprint.pformat(last_emb_layer))

    return None

def embedding_series(tokenizer, tensor, name):
    first_layer_map = pd.Series(list(tensor.detach().numpy()), index=tokenizer_id_array(tokenizer), name=name)
    return first_layer_map

# def encoded_embs_func(tokenizer, model):
#     @st.cache(hash_funcs={
#         type(tokenizer): id,
#         type(model): id
#     })
#     def encode_embs_work(sentence):
#         print(f"working for {sentence}")
#         sequence_input = tokenizer.encode(sentence, return_tensors="pt")
#         tokenzied_inputs = tokenizer.convert_ids_to_tokens(sequence_input[0])
#         token_output_states, *_ = model(sequence_input, output_hidden_states=True).hidden_states
#         return pd.Series(list(token_output_states.squeeze().detach().numpy()), index=tokenzied_inputs)
#     return encode_embs_work

tokenizer, model = init()

@st.cache(hash_funcs={
    type(tokenizer): id,
    type(model): id
})
def encode_embs(sentence):
    print(f"working for {sentence}")
    sequence_input = tokenizer.encode(sentence, return_tensors="pt")
    tokenzied_inputs = tokenizer.convert_ids_to_tokens(sequence_input[0])
    token_output_states, *_ = model(sequence_input, output_hidden_states=True).hidden_states
    return pd.Series(list(token_output_states.squeeze().detach().numpy()), index=tokenzied_inputs)

def normalise(A):
    lengths = (A**2).sum()**0.5
    return A/lengths

def length(A):
    return (A**2).sum()**0.5

def distance(left, right):
    if isinstance(left, str) and isinstance(right, str):
        left_vec, right_vec = [encode_embs(t).iloc[1:-1].mean() for t in [left, right]]
        # return length(left_vec - right_vec)
        # print(left_vec[0], right_vec[0])
        # FIXME: why left is equal to right? [CLS] is always same, [SEP] only differ when length change.
        # assert not np.array_equal(left_vec, right_vec)
        # diff = left_vec - right_vec
        # print(np.max(diff), np.min(diff))
        # print(np.abs(left_vec - right_vec))
        # return np.max(np.abs(left_vec - right_vec))
        # return np.sum((left_vec - right_vec) ** 2) ** 0.5
        return -np.dot(left_vec, right_vec)
    else:
        return 0

import networkx as nx

def as_plain(path):
    if isinstance(path, str):
        yield path
    elif iterable(path):
        for item in path:
            yield from as_plain(item)
        



def main():
    with torch.no_grad():
        # tokenizer, model = init()
        # encode_embs = encoded_embs_func(tokenizer, model)
        list_num = st.number_input("numbers of list", value=3)
        cols = st.beta_columns(list_num)
        texts = []
        for i, col in enumerate(cols):
            with col:
                texts.append(st.text_area(f"list {i+1}"))
        nodes = [[line.strip() for line in group.split() if line.strip() != ""] for group in texts]
        nodes, G, length, negative_cycle = list_maximum_likehood_on_embeddings(list_num, nodes)
        pick_n = st.number_input("get top k?", 3)
        for i, path in zip(range(pick_n), nx.shortest_simple_paths(G, (0,0,0), (list_num+1,0,1), weight='distance')):
            st.write(" ".join(as_plain(path)))
        #TODO add negative spanning tree on complete graph to deal with non-sequence product(projection).
        #TODO negative shortest path is a specific spanning tree.
        st.write(bf.negative_edge_cycle(G))
        st.write(length, negative_cycle)
        st.write(nodes)

def list_maximum_likehood_on_embeddings(list_num, nodes):
    nodes.insert(0, [0])
    nodes.append([1])
    st.write(nodes)
    G = nx.DiGraph()
    dis_df = []
    for gid, (lefts, rights) in enumerate(zip(
            nodes[:-1],
            nodes[1:]
    )):
        for lid, left in enumerate(lefts):
            for rid, right in enumerate(rights):
                v = distance(left, right)
                dis_df.append({"from": left, "to": right, "dis": v})
                G.add_edge((gid, lid, left), (gid+1, rid, right), distance=v)
    # dis_df = pd.DataFrame(dis_df)
    # dis_df = dis_df[(dis_df["from"] != 0) & (dis_df["to"] != 1)]
    # st.dataframe(dis_df.sort_values(by="dis"), width=800)
    length, nodes, negative_cycle = bf.bellman_ford(G, (0,0,0), (list_num+1,0,1), weight='distance')
    return nodes,G,length,negative_cycle
        # st.graphviz_chart(
        #     nx.nx_pydot.to_pydot(G).to_string()
        # )
        
        
if __name__ == "__main__":
    main()