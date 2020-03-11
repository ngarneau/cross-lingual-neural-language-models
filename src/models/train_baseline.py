from fastai.text import *
from fastai.text.data import *


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
    for lang in LANGUAGES:
        split_path = './data/raw/europarl/{}-en/europarl-v7.{}-en.{}.split'.format(lang, lang, lang)
        d = get_databunch(split_path)
        learn = language_model_learner(d, AWD_LSTM, pretrained=False)
        learn.fit_one_cycle(10, 1e-2)
