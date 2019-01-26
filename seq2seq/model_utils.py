import logging
import sys, shutil
import os
from seq2seq.monotonic_decoder import MonotonicDecoder
from seq2seq.inferences.monotonic_infer import MonotonicInference
from seq2seq.constants import STEP
from seq2seq.encoder import EncoderRNN
import torch
from torch import optim

__author__ = 'Shyam'


def setup_optimizers(args, encoder, decoder):
    learning_rate = args["lr"]
    reduction_factor = args['reduction_factor']
    patience = args['patience']

    enc_opt = optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_opt = optim.Adam(decoder.parameters(), lr=learning_rate)
    enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(enc_opt,
                                                         factor=reduction_factor,
                                                         patience=patience,
                                                         verbose=True)
    dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dec_opt,
                                                         factor=reduction_factor,
                                                         patience=patience,
                                                         verbose=True)
    return enc_opt, dec_opt, enc_scheduler, dec_scheduler


def model_builder(args, fr_lang, en_lang):
    bidi = args["bidi"]
    device_id = args["device_id"]
    batch_first = args["batch_first"]
    vector_size = args["wdim"]
    hidden_size = args["hdim"]
    beam_width = args["beam_width"]
    norm_by_length = args["norm_by_length"]
    if args["mono"]:
        decoder_input_size = 2 * 2 * hidden_size if bidi else 2 * hidden_size
    else:
        decoder_input_size = vector_size

    decoder_hidden_size = 2 * hidden_size if bidi else hidden_size
    # print("hidden_size", hidden_size)
    # print("decoder_hidden_size", decoder_hidden_size)
    dropout_p = args["wdrop"]

    if args["mono"]:
        en_lang.index_word(STEP)

    invoc_size = len(fr_lang.word2index)  # 20
    outvoc_size = len(en_lang.word2index)  # 30

    encoder = EncoderRNN(invoc_size=invoc_size,
                         vector_size=vector_size,
                         hidden_size=hidden_size,
                         bidi=bidi,
                         batch_first=batch_first)

    # if args["mono"]:
    decoder = MonotonicDecoder(input_size=decoder_input_size,
                               batch_first=batch_first,
                               outvoc_size=outvoc_size,
                               hidden_size=decoder_hidden_size)
    evaler = MonotonicInference(encoder=encoder,
                                decoder=decoder,
                                fr_lang=fr_lang,
                                en_lang=en_lang,
                                beam_width=beam_width,
                                norm_by_length=norm_by_length)
    logging.info(encoder)
    logging.info(decoder)
    # Move models to GPU
    if device_id is not None:
        encoder.cuda(device_id)
        decoder.cuda(device_id)
    return encoder, decoder, evaler


def load_checkpoint(encoder, decoder, enc_opt, dec_opt, ckpt_path):
    if os.path.isfile(ckpt_path):
        logging.info("=> loading checkpoint %s", ckpt_path)
        checkpoint = torch.load(ckpt_path)
        encoder.load_state_dict(checkpoint['enc_state_dict'])
        decoder.load_state_dict(checkpoint['dec_state_dict'])
        if enc_opt is not None:
            enc_opt.load_state_dict(checkpoint['enc_opt_state_dict'])
        if dec_opt is not None:
            dec_opt.load_state_dict(checkpoint['dec_opt_state_dict'])
        logging.info("=> loaded checkpoint!")
        return checkpoint
        # any other relevant state variables can be extracted from the checkpoint dict
    else:
        logging.info("=> no checkpoint at %s !!!", ckpt_path)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        From https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3

        """
    logging.info("saving model to %s", filename)
    torch.save(state, filename)
    if is_best:
        logging.info("copying to best ...")
        shutil.copyfile(filename, filename + '_best.pth.tar')
