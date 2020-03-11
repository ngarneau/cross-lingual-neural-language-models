from fastai.text import *
from fastai.text.data import *

import logging


LANGUAGES = {
    'bg',
    'cs',
    'da',
    'de',
    'en',
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


def get_databunch(path):
    databunch = TextLMDataBunch.from_folder(path)
    return databunch


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    for lang in LANGUAGES:
        logging.info("Starting training for {}".format(lang))
        split_path = './data/raw/europarl/{}-en/europarl-v7.{}-en.{}.split'.format(lang, lang, lang)
        d = get_databunch(split_path)
        learn = language_model_learner(d, AWD_LSTM, pretrained=False, metrics=[Perplexity()])
        learn.fit_one_cycle(90, 5e-3, moms=(0.8,0.7,0.8), div=10)
