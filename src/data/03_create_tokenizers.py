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


def get_vectors(path):
    vectors = dict()
    with open(path, 'r', encoding='utf-8') as fhandle:
        for i, line in enumerate(fhandle):
            elements = line.split()
            if len(elements) > 2:
                try:
                    word = elements[0].lower()
                    vector = np.asarray([float(i) for i in elements[1:]])
                    vectors[word] = vector
                except:
                    print("Could not process line {}".format(i))
    return vectors

def create_tokenizer(lang):
    logging.info(f"Working on {lang}.")
    train_file = f'./data/processed/europarl/europarl.tokenized.{lang}.split/train/data.txt'
    output_path = f'./data/processed/europarl-{lang}-tokenizer/'

    logging.info(f"Loading word vectors.")
    ft_vectors = get_vectors(f'./data/raw/embeddings/wiki.{lang}.align.vec')

    types = defaultdict(int)

    # We will grab every word that has a vector either in the train or valid
    with open(train_file) as fhandle:
        for line in tqdm(fhandle):
            tokens = line.split()
            for t in tokens:
                t = t.lower()
                ts = t.split('-')
                for a in ts:
                    if a in ft_vectors:
                        types[a] +=1

    with open(train_file.replace("train", "valid")) as fhandle:
        for line in tqdm(fhandle):
            tokens = line.split()
            for t in tokens:
                t = t.lower()
                ts = t.split('-')
                for a in ts:
                    if a in ft_vectors:
                        types[a] +=1


    logging.info("Total types: {}".format(len(types)))

    # Write vocab file
    vocab = {"<pad>": 0, '<unk>': 1, "</s>": 2}
    for word in sorted(types):
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
    for lang in LANGUAGES:
        create_tokenizer(lang)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
