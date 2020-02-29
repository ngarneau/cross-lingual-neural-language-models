from tqdm import tqdm
import click
import numpy as np


def write_split(filename, split, lines):
    with open("{}.{}".format(filename, split), 'w') as fhandle:
        for line in lines:
            fhandle.write(line)


@click.command()
@click.option('--file', default='./data/processed/europarl.tokenized.en', help='File to split in train/valid/test.')
@click.option('--train-ratio', default=0.6)
@click.option('--valid-ratio', default=0.2)
def main(file, train_ratio, valid_ratio):
    lines = list()
    with open(file) as fhandle:
        for line in tqdm(fhandle):
            lines.append(line)
    np.random.shuffle(lines)

    train_split_index = int(len(lines)*train_ratio)
    valid_split_index = train_split_index + int(len(lines)*valid_ratio)

    train_lines = lines[:train_split_index]
    valid_lines = lines[train_split_index:valid_split_index]
    test_lines = lines[valid_split_index:]

    write_split(file, 'train', train_lines)
    write_split(file, 'valid', valid_lines)
    write_split(file, 'test', test_lines)


if __name__ == '__main__':
    main()
