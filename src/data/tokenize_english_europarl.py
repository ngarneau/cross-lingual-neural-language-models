from nltk import word_tokenize
from tqdm import tqdm

def main():
    with open('./data/processed/europarl.tokenized.en', 'w') as fwrite:
        with open('./data/raw/europarl/es-en/europarl-v7.es-en.en') as fhandle:
            for line in tqdm(fhandle):
                tokens = word_tokenize(line)
                fwrite.write("{}\n".format(" ".join(tokens)))

if __name__ == '__main__':
    main()
