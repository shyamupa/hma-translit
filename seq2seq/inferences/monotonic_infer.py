import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V

from seq2seq.constants import EOS_token
from seq2seq.constants import SOS_token, SOS_ID, UNK_ID
from seq2seq.constants import STEP
from seq2seq.inferences.evaluate import Inference

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
__author__ = 'Shyam'


def encode_string(input_str, word2index):
    ans = []
    for w in input_str:
        if w in word2index:
            t = word2index[w]
        else:
            t = UNK_ID
        ans.append(t)
    return ans


class MonotonicInference(Inference):
    def __init__(self, encoder, decoder, fr_lang, en_lang, device_id=None, beam_width=1, norm_by_length=False):
        self.encoder = encoder
        self.decoder = decoder
        self.fr_lang, self.en_lang = fr_lang, en_lang
        self.K = beam_width
        self.norm_by_length = norm_by_length
        self.device_id = device_id

    def run_inference(self, x, max_length=60):
        # list of tokens
        padded_lemma = [SOS_token] + x.split(' ') + [EOS_token]
        padded_lemma_idx = encode_string(input_str=padded_lemma, word2index=self.fr_lang.word2index)
        input_word = V(torch.LongTensor(padded_lemma_idx))
        enc_hid = self.encoder.init_hidden()
        enc_outs, enc_hid = self.encoder(input_word, enc_hid)

        # initialize the decoder rnn
        dec_hid = self.decoder.init_hidden()

        # set prev_output_vec for first lstm step as BEGIN_WORD
        # prev_word = V(torch.LongTensor([SOS_ID]))

        # i is input index, j is output index
        i = 0
        num_outputs = 0
        beam = [(0, i, [SOS_ID], dec_hid)]
        outputs = []
        for beam_idx in range(3 * max_length):
            next_beam = []
            for score, att_pos, ys, dec_hid in beam:
                prev_word = V(torch.LongTensor([ys[-1]]))
                decoder_output, next_dec_hid = self.decoder(prev_word, att_pos, dec_hid, enc_outs)
                scores = self.decoder.out(decoder_output)
                probs = F.softmax(scores, dim=-1)
                # print("probs",probs)
                topk_probs, topk_ints = torch.topk(probs, self.K, dim=2)
                # print(topk_probs,topk_ints)
                for k in range(self.K - len(outputs)):
                    top_score = np.log(topk_probs.data[0, 0, k])
                    top_y = topk_ints.data[0, 0, k]
                    next_ys = ys + [top_y]
                    next_score = score + top_score
                    next_att_pos = att_pos
                    # print(top_y,self.en_lang.word2index[STEP])
                    if top_y == self.en_lang.word2index[STEP]:
                        if att_pos < len(padded_lemma) - 1:
                            next_att_pos = att_pos + 1
                    else:
                        next_att_pos = att_pos
                    if top_y == self.en_lang.word2index[EOS_token] or len(next_ys) == 3 * max_length:
                        # if not self.min_output_length or len(next_ys) >= self.min_output_length:
                        outputs.append((next_score, next_ys))
                    else:
                        next_beam.append((next_score, next_att_pos, next_ys, next_dec_hid))
            if len(outputs) >= self.K:
                break
            # sort beam in descending order by score.
            beam = list(sorted(next_beam, key=lambda tup: -tup[0]))[:self.K - len(outputs)]

        predicted_output_sequences = []
        for score, output in outputs:
            seq = []
            for i in output:
                seq.append(self.en_lang.index2word[i])
            if self.norm_by_length:
                score /= len(seq)
            predicted_output_sequences.append((score, seq))

        predicted_output_sequences = sorted(predicted_output_sequences, key=lambda tup: -tup[0])
        prediction = predicted_output_sequences
        return prediction

    def get_llh(self, x, y, max_length=60):
        padded_lemma = [SOS_token] + x.split(' ') + [EOS_token]
        padded_lemma_idx = encode_string(input_str=padded_lemma, word2index=self.fr_lang.word2index)
        input_word = V(torch.LongTensor(padded_lemma_idx))
        enc_hid = self.encoder.init_hidden()
        enc_outs, enc_hid = self.encoder(input_word, enc_hid)
        y = y.split(' ') + [EOS_token]
        # initialize the decoder rnn
        dec_hid = self.decoder.init_hidden()

        # i is input index, j is output index
        i = 0
        num_outputs = 0
        outputs = []
        y_idx = 0
        beam = [(0, i, y_idx, [SOS_ID], dec_hid)]
        for idx in range(3 * max_length):
            next_beam = []
            for score, att_pos, y_pos, ys, dec_hid in beam:
                prev_word = V(torch.LongTensor([ys[-1]]))
                decoder_output, next_dec_hid = self.decoder(prev_word,
                                                            att_pos,
                                                            dec_hid,
                                                            enc_outs)
                scores = self.decoder.out(decoder_output)
                probs = F.softmax(scores, dim=-1)
                if y_pos == len(y):
                    seq = []
                    for i in ys:
                        seq.append(self.en_lang.index2word[i])
                    # print("finished seq:", [s for s in seq if s != STEP and s != SOS_token])
                    outputs.append((score, ys))
                    continue
                yo = y[y_pos]
                # print("yo:", yo)
                # print(scores.size())
                yo_score = np.log(probs.data[0][0][self.en_lang.word2index[yo]])
                st_score = np.log(probs.data[0][0][self.en_lang.word2index[STEP]])
                if ys[-1] == self.en_lang.word2index[STEP] and ys[-2] == self.en_lang.word2index[STEP]:
                    possible_actions = [yo]
                else:
                    possible_actions = [yo, STEP]
                for action in possible_actions:
                    next_ys = ys + [self.en_lang.word2index[action]]
                    next_att_pos = att_pos
                    next_y_pos = y_pos
                    if action == STEP:
                        next_score = score + st_score
                        if att_pos < len(padded_lemma) - 1:
                            next_att_pos = att_pos + 1
                    else:
                        next_score = score + yo_score
                        next_att_pos = att_pos
                        next_y_pos = y_pos + 1
                    # print(next_ys)
                    # print(next_score)
                    new_state = (next_score, next_att_pos, next_y_pos, next_ys, next_dec_hid)
                    # print(new_state[:-1])
                    next_beam.append(new_state)
            # sort beam in descending order by score.
            # print("next_beam", len(next_beam))
            beam = list(sorted(next_beam, key=lambda tup: -tup[0]))[:self.K]

        outputs = sorted(outputs, key=lambda tup: -tup[0])
        predicted_output_sequences = []
        for score, output in outputs:
            seq = []
            for i in output:
                seq.append(self.en_lang.index2word[i])
            print("seq:", seq)
            # print("seq:", [s for s in seq if s != STEP and s != SOS_token])
            print("sco:", score)
            predicted_output_sequences.append((score, seq))
        print(x, y)
