import os
import logging
import shutil
from collections import defaultdict

from transformers import *
from tqdm import tqdm
import click


@click.command()
@click.option('--train-file', default='./data/processed/europarl.tokenized.en.train')
@click.option('--output-path', default='./data/processed/gpt2-europarl-tokenizer/')
def main(train_file, output_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    types = defaultdict(int)

    with open(train_file) as fhandle:
        for line in tqdm(fhandle):
            tokens = line.split()
            for t in tokens:
                types[t.lower()] +=1

    logging.info("Total types: {}".format(len(types)))
    logging.info("Reducing types that has less then 100 occurrences...")

    filtered_types = {k: v for k, v in types.items() if v >= 10}
    logging.info("Filtered types: {}".format(len(filtered_types)))

    num_tokens_added = tokenizer.add_tokens(filtered_types)
    logging.info("Num tokens added to tokenizer: {}".format(num_tokens_added))

    shutil.rmtree(output_path)
    os.makedirs(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
