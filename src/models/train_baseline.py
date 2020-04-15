from functools import partial

from fastai.text import *
from fastai.text.data import *
from fastai.callbacks import CSVLogger

import logging


LANGUAGES = [
    # 'bg',
    # 'cs',
    # 'da',
    # 'de',
    'en',
    # 'es',
    # 'et',
    # 'fi',
    # 'fr',
    # 'hu',
    # 'it',
    # 'lt',
    # 'lv',
    # 'nl',
    # 'pl',
    # 'pt',
    # 'ro',
    # 'sk',
    # 'sl',
    # 'sv'
]


def get_databunch(path):
    databunch = TextLMDataBunch.from_folder(path)
    return databunch


CUTOFF = 1000000

def get_vectors(path):
    vectors = dict()
    with open(path, 'r') as fhandle:
        for i, line in enumerate(fhandle):
            if i > CUTOFF:
                break
            elements = line.split()
            if len(elements) > 2:
                try:
                    word = elements[0].lower()
                    vector = np.asarray([float(i) for i in elements[1:]])
                    vectors[word] = vector
                except:
                    print("Could not process line {}".format(i))
    return vectors


def set_item_embedding(model, idx, embedding):
    model.encoder.weight.data[idx] = torch.FloatTensor(embedding)

def load_words_embeddings(model, vec_model, word_to_idx):
    for word in vec_model:
        if word in word_to_idx and word_to_idx[word] != 0:
            idx = word_to_idx[word]
            embedding = vec_model[word]
            set_item_embedding(model, idx, embedding)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = {'emb_sz': 300, 'n_hid': 1152, 'n_layers': 3, 'pad_token': 1, 'qrnn': False, 'bidir': False, 'output_p': 0.1, 'hidden_p': 0.15, 'input_p': 0.25, 'embed_p': 0.02, 'weight_p': 0.2, 'tie_weights': True, 'out_bias': True}
    for lang in LANGUAGES:
        logging.info("Starting training for {}".format(lang))
        split_path = './data/processed/europarl/europarl.tokenized.{}.split'.format(lang)
        d = get_databunch(split_path)
        vectors = get_vectors('./data/raw/embeddings/wiki.en.align.vec')
        learn = language_model_learner(
            d,
            AWD_LSTM,
            pretrained=False,
            config=config,
            metrics=[Perplexity()],
            callback_fns=[partial(CSVLogger, filename='history_freezed_{}'.format(lang))]
        )
        load_words_embeddings(learn.model[0], vectors, d.vocab.stoi)
        learn.model[0].encoder.weight
        learn.model[0].encoder.weight.requires_grad = False
        learn.fit_one_cycle(90, 5e-3)
