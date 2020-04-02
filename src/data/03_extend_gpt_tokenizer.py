import os
import logging
import shutil
from collections import defaultdict
import json

from transformers import *
from tqdm import tqdm
import click


@click.command()
@click.option('--train-file', default='./data/raw/europarl/es-en/europarl-v7.es-en.en.split/train/data.txt')
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
    vocab = {word: i for i, word in enumerate(sorted(filtered_types))}
    import pdb; pdb.set_trace()
    with open("./data/processed/europarl.vocab", 'w') as fhandle:
        json.dump(vocab, fhandle)

    # Write an empty merge file
    with open("./data/processed/europarl.merges", 'w') as fhandle:
        fhandle.write("#version: 0.2\n")

    tokenizer = GPT2Tokenizer(
        vocab_file="./data/processed/europarl.vocab",
        merges_file="./data/processed/europarl.merges",
        unk_token="<|unknown|>"
    )

    shutil.rmtree(output_path)
    os.makedirs(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
