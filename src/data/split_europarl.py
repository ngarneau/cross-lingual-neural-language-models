import os
import logging
from tqdm import tqdm
import click
import numpy as np


languages = {
    'bg',
    'cs',
    'da',
    'de',
    'en',
    # 'el',  # we nous have encoding problems with el
    'es',
    'et',
    'fi',
    'fr',
    'hu',
    'it',
    'lt',
    'lv',
    'nl',
    'pl',
    'pt',
    'ro',
    'sk',
    'sl',
    'sv'
}


def write_split(filename, split, lines):
    with open("{}/{}".format(filename, split), 'w') as fhandle:
        for line in lines:
            fhandle.write(line)


def split_dataset(filename, train_ratio, valid_ratio):
    lines = list()
    with open(filename) as fhandle:
        for line in tqdm(fhandle):
            lines.append(line)
    # np.random.shuffle(lines)  # I think we should'nt shuffle for vocab concern

    train_split_index = int(len(lines)*train_ratio)
    valid_split_index = train_split_index + int(len(lines)*valid_ratio)

    train_lines = lines[:train_split_index]
    valid_lines = lines[train_split_index:valid_split_index]
    test_lines = lines[valid_split_index:]

    os.makedirs(filename + ".split/train", exist_ok=True)
    os.makedirs(filename + ".split/valid", exist_ok=True)
    os.makedirs(filename + ".split/test", exist_ok=True)
    write_split(filename + ".split/train", 'data.txt', train_lines)
    write_split(filename + ".split/valid", 'data.txt', valid_lines)
    write_split(filename + ".split/test", 'data.txt', test_lines)


@click.command()
@click.option('--train-ratio', default=0.6)
@click.option('--valid-ratio', default=0.2)
def main(train_ratio, valid_ratio):

    # Special case for english
    logging.info("Tokenizing en corpus")
    split_dataset(
        './data/raw/europarl/es-en/europarl-v7.es-en.en',
        train_ratio,
        valid_ratio
    )

    for lang in LANGUAGES:
        logging.info("Splitting {} corpus".format(lang))
        split_dataset(
            './data/raw/europarl/{}-en/europarl-v7.{}-en.{}'.format(lang, lang, lang),
            train_ratio,
            valid_ratio
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
