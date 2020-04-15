import os
import logging
import shutil
from collections import defaultdict
import json

from transformers import *
from tqdm import tqdm
import click
import numpy as np


def get_vectors(path):
    vectors = dict()
    with open(path, 'r') as fhandle:
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


@click.command()
@click.option('--train-file', default='./data/processed/europarl/europarl.tokenized.en.split/train/data.txt')
@click.option('--output-path', default='./data/processed/gpt2-europarl-tokenizer/')
def main(train_file, output_path):
    types = defaultdict(int)
    ft_vectors = get_vectors('./data/raw/embeddings/wiki.en.align.vec')

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
    with open("./data/processed/europarl.vocab", 'w') as fhandle:
        for word, idx in vocab.items():
            fhandle.write("{} {}\n".format(word, idx))

    tokenizer = TransfoXLTokenizer(
        vocab_file="./data/processed/europarl.vocab",
        eos_token="</s>"
    )

    shutil.rmtree(output_path)
    os.makedirs(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
