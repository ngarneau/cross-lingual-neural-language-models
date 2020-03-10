import logging
from tqdm import tqdm
import click
import numpy as np


LANGUAGES = {
    'bg',
    'cs',
    'da',
    'de',
    'en',
    # 'el',  # We nous have encoding problems with el
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
    with open("{}.{}".format(filename, split), 'w') as fhandle:
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

    write_split(filename, 'train', train_lines)
    write_split(filename, 'valid', valid_lines)
    write_split(filename, 'test', test_lines)


@click.command()
@click.option('--train-ratio', default=0.6)
@click.option('--valid-ratio', default=0.2)
def main(train_ratio, valid_ratio):
    for lang in LANGUAGES:
        logging.info("Splitting {} corpus".format(lang))
        split_dataset(
            './data/processed/europarl.tokenized.{}'.format(lang),
            train_ratio,
            valid_ratio
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
