import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from readers.aligned_reader import read_examples
from seq2seq.evaluators.reporter import AccReporter, get_decoded_words
from seq2seq.model_utils import load_checkpoint, model_builder, setup_optimizers
from seq2seq.prepare_data import langcodes, load_vocab_and_examples
from seq2seq.runner import run
from seq2seq.trainers.monotonic_train import MonotonicTrainer
from utils.arguments import PARSER

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)

if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    logging.info(args)
    seed = args["seed"]

    remove_spaces = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    lang = langcodes[args["lang"]]

    testpath = args["ftest"]

    fr_lang, en_lang, examples = load_vocab_and_examples(vocabfile=args["vocabfile"], aligned_file=args["aligned_file"])
    logging.info(fr_lang.word2index)
    logging.info(en_lang.word2index)

    # ALWAYS READ ALL TEST EXAMPLES
    test = read_examples(fpath=testpath)

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
            test=test,test_reporter=test_reporter)
