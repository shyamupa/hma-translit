import random
import logging
import sys

import torch
import torch.nn as nn
import numpy as np

from utils.arguments import PARSER
from readers.aligned_reader import load_aligned_data, read_examples
from seq2seq.constants import STEP
from seq2seq.evaluators.reporter import AccReporter, get_decoded_words
from seq2seq.lang import Lang
from seq2seq.runner import run
from seq2seq.trainers.monotonic_train import MonotonicTrainer
from seq2seq.model_utils import load_checkpoint, model_builder, setup_optimizers

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)


def subsample_examples(examples, frac, single_token):
    new_examples = []
    for ex in examples:
        fr, en, weight, is_eng = ex
        frtokens, entokens = fr.split(" "), en.split(" ")
        if len(frtokens) != len(entokens): continue
        if single_token:
            if len(frtokens) > 1 or len(entokens) > 1: continue
        for frtok, entok in zip(frtokens, entokens):
            new_examples.append((frtok, entok, weight, is_eng))
    examples = new_examples
    logging.info("new examples %d", len(examples))
    # subsample if needed
    random.shuffle(examples)
    if frac < 1.0:
        tmp = examples[0:int(frac * len(examples))]
        examples = tmp
    elif frac > 1.0:
        tmp = examples[0:int(frac)]
        examples = tmp
    return examples


def index_vocab(examples, fr_lang, en_lang):
    for ex in examples:
        raw_x, raw_y, xs, ys, weight, is_eng = ex
        fr_lang.index_words(xs)
        en_lang.index_words(ys)
    logging.info("train size %d", len(examples))


langcodes = {"hi": "hindi", "fa": "farsi", "ta": "tamil", "ba": "bengali", "ka": "kannada", "he": "hebrew",
             "th": "thai"}

if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    logging.info(args)
    batch_first = args["batch_first"]
    device_id = args["device_id"]
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
    # see_phrase_alignments(examples=examples)
    logging.info(fr_lang.word2index)
    logging.info(en_lang.word2index)
    # ALWAYS READ ALL TEST EXAMPLES
    test = read_examples(fpath=testpath)
    train = read_examples(fpath=trainpath)

    train = [ex for ex in train if '  ' not in ex[0] and '  ' not in ex[1]]
    logging.info("input vocab: %d", fr_lang.n_words)
    logging.info("output vocab: %d", en_lang.n_words)
    logging.info("beam width: %d", args["beam_width"])

    # Initialize models
    encoder, decoder, evaler = model_builder(args, fr_lang=fr_lang, en_lang=en_lang)
    enc_opt, dec_opt, enc_sch, dec_sch = setup_optimizers(args=args, encoder=encoder, decoder=decoder)
    criterion = nn.NLLLoss()

    trainer = MonotonicTrainer(encoder=encoder, decoder=decoder,
                               enc_opt=enc_opt, dec_opt=dec_opt,
                               enc_sch=enc_sch, dec_sch=dec_sch,
                               fr_lang=fr_lang, en_lang=en_lang)

    # Begin!
    test_reporter = AccReporter(args=args,
                                dump_file=args["dump"])
    train_reporter = AccReporter(args=args,
                                 dump_file=args["dump"] + ".train.txt" if args["dump"] is not None else None)

    if args["restore"]:
        if "," in args["restore"]:
            logging.info("ensembling ...")
            pass
        else:
            load_checkpoint(encoder=encoder, decoder=decoder,
                            enc_opt=enc_opt, dec_opt=dec_opt,
                            ckpt_path=args["restore"])
            if args["interactive"]:
                try:
                    while True:
                        surface = input("enter surface:")
                        surface = " ".join(list(surface))
                        print(surface)
                        x, y, weight, is_eng = surface, None, 1.0, False
                        decoded_outputs = evaler.infer_on_example(sentence=x)
                        scores_and_words = get_decoded_words(decoded_outputs)
                        decoded_words = [w for s, w in scores_and_words]
                        scores = [s for s, w in scores_and_words]
                        print(scores_and_words)
                except KeyboardInterrupt:
                    print('interrupted!')
                    sys.exit(0)
            else:
                logging.info(20 * "-" + "TEST" + 20 * "-")
                test_reporter.report_eval(epoch=-1, seen=-1, evaler=evaler, examples=test)

    else:
        run(args=args,
            examples=examples,
            trainer=trainer, evaler=evaler, criterion=criterion,
            train=train, test=test,
            train_reporter=train_reporter, test_reporter=test_reporter)
