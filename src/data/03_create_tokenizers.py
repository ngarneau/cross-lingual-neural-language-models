import os
import logging
import shutil
from collections import defaultdict
import json

from transformers import *
from tqdm import tqdm
import click
import numpy as np

from src.data import LANGUAGES
from src.models.evaluate_language_models import get_vectors


def get_dictionary(path):
    src_tgt = defaultdict(list)
    tgt_src = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as fhandle:
        for line in fhandle:
            try:
                src, tgt = line[:-1].split()
                src_tgt[src].append(tgt)
                tgt_src[tgt].append(src)
            except:
                logging.warn("Could not parse following line:")
                logging.warn(line)
    return src_tgt, tgt_src


def complement_vectors(vocab, lang_vectors, english_vectors):
    for word in vocab:
        if word not in lang_vectors:
            if word in english_vectors:
                lang_vectors[word] = english_vectors[word]
    return lang_vectors


def create_tokenizer(lang):
    logging.info(f"Working on {lang}.")

    english_tokenizer = TransfoXLTokenizer.from_pretrained('./data/processed/europarl-en-tokenizer')
    english_vocab = english_tokenizer.get_vocab()
    english_vocab_size = len(english_vocab)
    logging.info("English vocab size: {}".format(english_vocab_size))

    # english_vectors = get_vectors('./data/raw/embeddings/wiki.en.align.vec', english_tokenizer.get_vocab().keys())

    src_tgt_dict, tgt_src_dict = get_dictionary(f'./data/raw/dictionaries/en-{lang}.txt')

    joint_vocab = english_tokenizer.get_vocab().keys() & src_tgt_dict.keys()
    logging.info("Joint vocab size: {} ({})".format(len(joint_vocab), len(joint_vocab) / english_vocab_size))
    filtered_lang_vocab = {w for k, v in src_tgt_dict.items() for w in v if k in joint_vocab}

    train_file = f'./data/processed/europarl/europarl.tokenized.{lang}.split/train/data.txt'
    output_path = f'./data/processed/europarl-{lang}-tokenizer/'

    logging.info(f"Loading word vectors.")
    ft_vectors = get_vectors(f'./data/raw/embeddings/wiki.{lang}.align.vec', vocab=filtered_lang_vocab)
    logging.info("Words with lang vectors: {}".format(len(ft_vectors)))

    # types = defaultdict(int)

    # # We will grab every word that has a vector either in the train or valid
    # with open(train_file) as fhandle:
    #     for line in tqdm(fhandle):
    #         tokens = line.split()
    #         for t in tokens:
    #             t = t.lower()
    #             ts = t.split('-')
    #             for a in ts:
    #                 if a in ft_vectors:
    #                     types[a] +=1

    # with open(train_file.replace("train", "valid")) as fhandle:
    #     for line in tqdm(fhandle):
    #         tokens = line.split()
    #         for t in tokens:
    #             t = t.lower()
    #             ts = t.split('-')
    #             for a in ts:
    #                 if a in ft_vectors:
    #                     types[a] +=1


    # Write vocab file
    vocab = {
        "<pad>": 0,
        '<unk>': 1,
        "</s>": 2,
        '!': 3,
        '#': 4,
        '$': 5,
        '%': 6,
        '&': 7,
        "'": 8,
        '(': 9,
        ')': 10,
        '+': 11,
        '+/': 12,
        ',': 13,
        '.': 14,
        '/': 15,
    }
    for word in sorted(ft_vectors.keys()):
        vocab[word] = len(vocab)
    with open(f"./data/processed/europarl-{lang}.vocab", 'w') as fhandle:
        for word, idx in vocab.items():
            fhandle.write("{} {}\n".format(word, idx))

    tokenizer = TransfoXLTokenizer(
        vocab_file=f"./data/processed/europarl-{lang}.vocab",
        eos_token="</s>"
    )

    if os.path.exists(output_path) and os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    tokenizer.save_pretrained(output_path)


def main():
    for lang in sorted(LANGUAGES):
        if lang == 'fr':
            create_tokenizer(lang)
    # create_tokenizer('en')



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
