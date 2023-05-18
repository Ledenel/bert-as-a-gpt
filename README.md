# bert-as-a-gpt
This project is aimed to make MLM generate texts like gpt in an unsupervized, consistent way.

It's used in https://github.com/nightingu/damebot to generate formatted memes, because MLM is better to fill in blanks in an appropriate context, while GPT only support left context.
the limit of MLM is that it must know the total length (however it can be solved via brute-force enumerate lengths and calculate total loss).

some related theory:

https://arxiv.org/abs/1902.04094


