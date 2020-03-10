from fastai.text import *

if __name__ == '__main__':
    sentences = list()
    with open("./data/processed/europarl.tokenized.sl.train") as fhandle:
        for line in fhandle:
            sentences.append(line[:-1])
    d = TextList(sentences).split_none().label_for_lm().databunch()
    learn = language_model_learner(d, AWD_LSTM, pretrained=False)
    learn.fit_one_cycle(10, 1e-2)
