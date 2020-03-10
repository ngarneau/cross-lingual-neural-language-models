import logging
import os

from nltk import word_tokenize
from tqdm import tqdm

LANGUAGES = {
    'bg',
    'cs',
    'da',
    'de',
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

def tokenize_lang(input_file, output_file):
    if not os.path.exists(output_file):
        with open(output_file, 'w') as fwrite:
            with open(input_file) as fhandle:
                for line in tqdm(fhandle):
                    tokens = word_tokenize(line)
                    fwrite.write("{}\n".format(" ".join(tokens)))
    else:
        logging.info("{} already exists, skipping.".format(output_file))

def main():

    # Special case for english
    logging.info("Tokenizing en corpus")
    tokenize_lang(
        './data/raw/europarl/es-en/europarl-v7.es-en.en',
        './data/processed/europarl.tokenized.en'
    )

    for lang in LANGUAGES:
        logging.info("Tokenizing {} corpus".format(lang))
        tokenize_lang(
            './data/raw/europarl/{}-en/europarl-v7.{}-en.{}'.format(lang, lang, lang),
            './data/processed/europarl.tokenized.{}'.format(lang)
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
