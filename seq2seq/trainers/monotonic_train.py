import torch
import torch.nn as nn
from torch.autograd import Variable as V
from seq2seq.constants import SOS_token, SOS_ID
from seq2seq.constants import EOS_token
from seq2seq.constants import ALIGN_SYMBOL
from seq2seq.constants import STEP
from seq2seq.constants import UNK
from seq2seq.trainers.seq2seq_attn_trainer import Seq2SeqAttnTrainer


def make_target(word, word2idx):
    return torch.LongTensor([word2idx[word]])


class MonotonicTrainer(Seq2SeqAttnTrainer):
    def __init__(self, encoder, decoder, enc_opt, dec_opt, enc_sch, dec_sch, fr_lang, en_lang, clip=0.5, teacher_forcing_ratio=0.5,
                 device_id=None):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_opt = enc_opt
        self.dec_opt = dec_opt
        self.enc_sch = enc_sch
        self.dec_sch = dec_sch
        self.fr_lang = fr_lang
        self.en_lang = en_lang
        self.clip = clip
        self.device_id = device_id

    def prepare_example(self, example):
        raw_x, raw_y, x, y, weight, is_eng = example
        raw_x, raw_y, x, y = raw_x.split(" "), raw_y.split(" "), x.split(" "), y.split(" ")
        example = raw_x, raw_y, x, y, weight
        return example

    def compute_loss(self, example, criterion, profile):
        raw_x, raw_y, aligned_x, aligned_y, weight = example
        # i is input index, j is output index
        i = 0
        j = 0
        padded_raw_x = [SOS_token] + raw_x + [EOS_token]
        hidden = self.encoder.init_hidden()
        padded_lemma_idx = [self.fr_lang.word2index[w] for w in padded_raw_x]
        input_word = V(torch.LongTensor(padded_lemma_idx))
        encoder_outputs, encoder_hidden_state = self.encoder(input_word, hidden)
        aligned_x += [EOS_token]
        aligned_y += [EOS_token]

        # start decoding, keeping track of sequence loss
        decoder_hidden = self.decoder.init_hidden()
        prev_word = V(torch.LongTensor([SOS_ID]))
        loss = []  # V(torch.FloatTensor([0.0]))

        for a, (input_char, output_char) in enumerate(zip(aligned_x, aligned_y)):
            possible_outputs = []
            if output_char == EOS_token:
                decoder_hidden, scores = self.step_decoder(prev_word=prev_word, i=i,
                                                           decoder_hidden=decoder_hidden,
                                                           encoder_outputs=encoder_outputs)
                target = V(make_target(word=EOS_token, word2idx=self.en_lang.word2index))
                ex_loss = criterion(input=scores, target=target)
                loss.append(ex_loss)
                continue

            if padded_raw_x[i] == SOS_token and aligned_x[a] != ALIGN_SYMBOL:
                decoder_hidden, scores = self.step_decoder(prev_word=prev_word, i=i,
                                                           decoder_hidden=decoder_hidden,
                                                           encoder_outputs=encoder_outputs)
                target = V(make_target(word=STEP, word2idx=self.en_lang.word2index))
                ex_loss = criterion(input=scores, target=target)
                loss.append(ex_loss)

                prev_word = V(make_target(word=STEP, word2idx=self.en_lang.word2index))
                i += 1

            if aligned_y[a] != ALIGN_SYMBOL:
                decoder_hidden, scores = self.step_decoder(prev_word=prev_word, i=i,
                                                           decoder_hidden=decoder_hidden,
                                                           encoder_outputs=encoder_outputs)

                if aligned_y[a] in self.en_lang.word2index:
                    target = V(make_target(word=aligned_y[a], word2idx=self.en_lang.word2index))
                    ex_loss = criterion(input=scores, target=target)
                    prev_word = V(make_target(word=aligned_y[a], word2idx=self.en_lang.word2index))
                else:
                    target = V(make_target(word=UNK, word2idx=self.en_lang.word2index))
                    ex_loss = criterion(input=scores, target=target)
                    prev_word = V(make_target(word=UNK, word2idx=self.en_lang.word2index))

                loss.append(ex_loss)

                j += 1

            if i < len(padded_raw_x) - 1 and aligned_x[a + 1] != ALIGN_SYMBOL:
                decoder_hidden, scores = self.step_decoder(prev_word=prev_word, i=i,
                                                           decoder_hidden=decoder_hidden,
                                                           encoder_outputs=encoder_outputs)
                target = V(torch.LongTensor([self.en_lang.word2index[STEP]]))
                ex_loss = criterion(input=scores, target=target)
                loss.append(ex_loss)
                prev_word = V(torch.LongTensor([self.en_lang.word2index[STEP]]))
                # whenever you step, attend to next position
                i += 1
        return weight*sum(loss)/len(loss)

    def step_decoder(self, prev_word, i, decoder_hidden, encoder_outputs):
        decoder_output, decoder_hidden = self.decoder(prev_word,
                                                      i,
                                                      decoder_hidden,
                                                      encoder_outputs)
        # compute local loss
        scores = self.decoder.out(decoder_output)
        log_softmax = nn.LogSoftmax(dim=2)
        scores = log_softmax(scores)
        scores = scores.squeeze(1)
        return decoder_hidden, scores
