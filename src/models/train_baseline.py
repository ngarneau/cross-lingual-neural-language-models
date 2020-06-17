import collections
from functools import partial

from fastai.text import *
from fastai.text.data import *
from fastai.callbacks import CSVLogger, SaveModelCallback

from tqdm import tqdm

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


def get_vectors(path):
    vectors = dict()
    with open(path, 'r') as fhandle:
        for i, line in enumerate(fhandle):
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
    set_item_embedding(model, 2, vec_model['</s>'])


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config = {
        'emb_sz': 300,
        'n_hid': 1152,
        'n_layers': 3,
        'pad_token': 1,
        'qrnn': False,
        'bidir': False,
        'output_p': 0.1,
        'hidden_p': 0.15,
        'input_p': 0.25,
        'embed_p': 0.02,
        'weight_p': 0.2,
        'tie_weights': True,
        'out_bias': False
    }
    for lang in LANGUAGES:

        vectors = get_vectors('./data/raw/embeddings/wiki.en.align.vec')

        train_file = f'./data/processed/europarl/europarl.tokenized.{lang}.split/train/data.txt'
        # We will grab every word that has a vector either in the train or valid
        words = set()
        with open(train_file) as fhandle:
            for line in tqdm(fhandle):
                tokens = line.split()
                for t in tokens:
                    t = t.lower()
                    ts = t.split('-')
                    for a in ts:
                        if a in vectors:
                            words.add(a)

        itos = list()
        itos.append("<unk>")
        itos.append("<pad>")
        itos.append("xxbos")
        for word in words:
            itos.append(word)

        vocab = Vocab(itos)
        stoi = {k: v for k, v in vocab.stoi.items()}
        split_path = './data/processed/europarl/'
        csv_name = 'europarl.tokenized.{}.csv.lower'.format(lang)
        databunch = TextLMDataBunch.from_csv(split_path, csv_name, delimiter='\t', text_cols=0, bs=128, vocab=vocab, num_workers=4)

        logging.info("Starting training for {}".format(lang))
        learn = language_model_learner(
            databunch,
            AWD_LSTM,
            pretrained=False,
            config=config,
            metrics=[Perplexity()],
            callback_fns=[
                partial(CSVLogger, filename='history_freezed_{}'.format(lang)),
                partial(SaveModelCallback)
            ]
        )
        load_words_embeddings(learn.model[0], vectors, stoi)
        learn.model[0].encoder.weight
        learn.model[0].encoder.weight.requires_grad = False
        learn.fit_one_cycle(90, 5e-3)

