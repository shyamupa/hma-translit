import logging
import pickle
import random

import numpy as np
import torch

from utils.arguments import PARSER
from readers.aligned_reader import load_aligned_data, read_examples, subsample_examples
from seq2seq.lang import Lang
from seq2seq.constants import STEP


def index_vocab(examples, fr_lang, en_lang):
    for ex in examples:
        raw_x, raw_y, xs, ys, weight, is_eng = ex
        fr_lang.index_words(xs)
        en_lang.index_words(ys)
    logging.info("train size %d", len(examples))


def load_vocab_and_examples(vocabfile, aligned_file):
    with open(vocabfile + ".frvoc", 'rb') as f:
        fr_lang = pickle.load(f)
    with open(vocabfile + ".envoc", 'rb') as f:
        en_lang = pickle.load(f)
    with open(aligned_file, 'rb') as f:
        examples = pickle.load(f)
    return fr_lang, en_lang, examples


def load_vocab(vocabfile):
    with open(vocabfile + ".frvoc", 'rb') as f:
        fr_lang = pickle.load(f)
    with open(vocabfile + ".envoc", 'rb') as f:
        en_lang = pickle.load(f)
    return fr_lang, en_lang


def save_vocab_and_examples(fr_lang, en_lang, examples, vocabfile, aligned_file):
    with open(vocabfile + ".frvoc", 'wb') as f:
        pickle.dump(fr_lang, file=f)
    with open(vocabfile + ".envoc", 'wb') as f:
        pickle.dump(en_lang, file=f)
    with open(aligned_file, 'wb') as f:
        pickle.dump(examples, file=f)


langcodes = {"hi": "hindi", "fa": "farsi", "ta": "tamil", "ba": "bengali", "ka": "kannada", "he": "hebrew",
             "th": "thai"}

if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    logging.info(args)
    # batch_first = args["batch_first"]
    # device_id = args["device_id"]
    seed = args["seed"]
    native_or_eng = args["nat_or_eng"]
    single_token = args["single_token"]

    remove_spaces = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    lang = langcodes[args["lang"]]

    trainpath = "data/%s/%s_train_annotateEN" % (lang, lang) if args["ftrain"] is None else args["ftrain"]
    testpath = "data/%s/%s_test_annotateEN" % (lang, lang) if args["ftest"] is None else args["ftest"]

    examples = read_examples(fpath=trainpath,
                             native_or_eng=native_or_eng,
                             remove_spaces=remove_spaces)

    examples = subsample_examples(examples=examples, frac=args["frac"], single_token=single_token)

    fr_lang, en_lang = Lang(name="fr"), Lang(name="en")
    examples = load_aligned_data(examples=examples,
                                 mode="mcmc",
                                 seed=seed)
    index_vocab(examples, fr_lang, en_lang)
    en_lang.index_word(STEP)
    fr_lang.compute_maps()
    en_lang.compute_maps()
    save_vocab_and_examples(fr_lang, en_lang, examples, vocabfile=args["vocabfile"], aligned_file=args["aligned_file"])
