import logging
import sys

from seq2seq.evaluators.reporter import get_decoded_words
from seq2seq.model_utils import load_checkpoint, model_builder, setup_optimizers
from seq2seq.prepare_data import load_vocab
from utils.arguments import PARSER

logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)

if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    logging.info(args)

    fr_lang, en_lang = load_vocab(vocabfile=args["vocabfile"])
    logging.info(fr_lang.word2index)
    logging.info(en_lang.word2index)

    logging.info("input vocab: %d", fr_lang.n_words)
    logging.info("output vocab: %d", en_lang.n_words)
    logging.info("beam width: %d", args["beam_width"])

    # Initialize models
    encoder, decoder, evaler = model_builder(args, fr_lang=fr_lang, en_lang=en_lang)
    enc_opt, dec_opt, enc_sch, dec_sch = setup_optimizers(args=args, encoder=encoder, decoder=decoder)

    if args["restore"]:
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
            testpath = args["ftest"]
            with open(args["dump"], "w") as out:
                for idx, line in enumerate(open(testpath)):
                    surface = line.strip()
                    x, y, weight, is_eng = surface, None, 1.0, False
                    if idx > 0 and idx % 200 == 0:
                        logging.info("running infer on example %d", idx)
                    decoded_outputs = evaler.infer_on_example(sentence=x)
                    scores_and_words = get_decoded_words(decoded_outputs)
                    # decoded_words = [w for s, w in scores_and_words]
                    # scores = [s for s, w in scores_and_words]
                    beam_outputs = ";".join([word for score, word in scores_and_words])
                    beam_scores = ";".join([str(score) for score, word in scores_and_words])
                    buf = f"{x}\t{beam_outputs}\t{beam_scores}\n"
                    out.write(buf)
