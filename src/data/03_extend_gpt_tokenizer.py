import os
import logging
import shutil
from collections import defaultdict
import json

from transformers import *
from tqdm import tqdm
import click


@click.command()
@click.option('--train-file', default='./data/processed/europarl/europarl.tokenized.en.split/train/data.txt')
@click.option('--output-path', default='./data/processed/gpt2-europarl-tokenizer/')
def main(train_file, output_path):
    types = defaultdict(int)

    with open(train_file) as fhandle:
        for line in tqdm(fhandle):
            tokens = line.split()
            for t in tokens:
                types[t.lower()] +=1


    logging.info("Total types: {}".format(len(types)))
    logging.info("Reducing types that has less then 100 occurrences...")

    filtered_types = {k: v for k, v in types.items() if v >= 2}
    logging.info("Filtered types: {}".format(len(filtered_types)))

    # Write vocab file
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for word in sorted(filtered_types):
        vocab[word] = len(vocab)
    with open("./data/processed/europarl.vocab", 'w') as fhandle:
        for word, idx in vocab.items():
            fhandle.write("{} {}\n".format(word, idx))

    tokenizer = TransfoXLTokenizer(
        vocab_file="./data/processed/europarl.vocab"
    )

    shutil.rmtree(output_path)
    os.makedirs(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
