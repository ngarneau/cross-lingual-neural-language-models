from transformers import *
from tqdm import tqdm
import click

def load_dataset(filename):
    lines = list()
    with open(filename) as fhandle:
        for line in fhandle:
            lines.append(line[:-1].lower())  # Remove the trailing \n
    return lines


def load_splits(file_prefix):
    train_split = load_dataset(file_prefix + ".train")
    valid_split = load_dataset(file_prefix + ".valid")
    test_split = load_dataset(file_prefix + ".test")
    return train_split, valid_split, test_split


@click.command()
@click.option('--data-file-prefix', default='./data/processed/europarl.tokenized.en')
@click.option('--tokenizer-path', default='./data/processed/gpt2-europarl-tokenizer/')
def main(data_file_prefix, tokenizer_path):
    logging.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('./data/processed/gpt2-europarl-tokenizer', pad_token='<PAD>', pad_token_id=-100)

    logging.info("Loading splits...")
    train_split, valid_split, test_split = load_splits(data_file_prefix)

    logging.info("Preparing train data...")
    import pdb; pdb.set_trace()
    encoded_train = [tokenizer.encode(l, max_length=1024, pad_to_max_length=True, return_tensors='pt') for l in tqdm(train_split)]


    # model = GPT2LMHeadModel.from_pretrained('gpt2')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
