"""
# TODO: adapt current augment func to allow passing in pipeline? rename?
# TODO: see if fill-mask pipeline can accept kwargs like top_p, beams, etc.
# TODO: add option to return tensors?
# TODO: adjust both to work as transforms in a torch dataset
# TODO: add CLI to generate data and save to csv
"""
import numpy as np
from transformers import pipeline


MASK = '<mask>'
fill_mask = pipeline('fill-mask')


def mask(txt, n=1):
    tokens = txt.split(' ')
    idx = np.random.randint(0, len(tokens), n)
    return ' '.join(t if i not in idx else MASK for i, t in enumerate(tokens))



def augment(t, n=1, return_all=False):
    # Each item will be a list of strings. Each string in res[i]
    # will have i words changed.
    res = [[t]]
    for i in range(n+1):
        t = [aug_dict['sequence'].replace('<s>', '').replace('</s>', '')
             for row in res[-1] for aug_dict in fill_mask(mask(row))]
        res.append(t)
    if not return_all: res = res[n]
    return res


# Equivalent of mask() for text-generation pipeline. May want to rename these
# later.
def truncate(text, drop=None, drop_pct=None, rand_low=None, rand_high=None,
	     min_keep=3):
    tokens = text.split()
    if len(tokens) <= min_keep:
        n = 0
    else:
        if drop:
            n = drop
        elif drop_pct:
            n = int(drop_pct * len(tokens))
        elif rand_low is not None:
            n = np.random.randint(rand_low, rand_high)
        else:
            n = 10
        n = np.clip(n, 0, len(tokens) - min_keep)
        tokens = tokens[:-n]
    return ' '.join(tokens), n


def generative_augment(text, min_length=5, max_length=15, **generate_kwargs):
    # generate counts current length as part of min_length. For now, we're just splitting words
    # so I'm naively inflating n_curr a bit since pipeline does some form of sub-word tokenization.
    n_curr = int(len(text.split()) * 1.1)
    return generate(text, min_length=n_curr + min_length,
    max_length=n_curr + max_length, **generate_kwargs)
