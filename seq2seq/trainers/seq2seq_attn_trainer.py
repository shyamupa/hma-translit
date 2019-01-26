import time
import logging

import torch
from torch.autograd import Variable as V
import torch.nn.functional as F

from seq2seq.constants import EOS_token, SOS_ID
from seq2seq.torch_utils import variables_from_pair


class Seq2SeqAttnTrainer:
    def __init__(self, encoder, decoder, fr_lang, en_lang, enc_opt, dec_opt, clip=0.5, teacher_forcing_ratio=0.5,
                 device_id=None):
        self.clip = clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device_id = device_id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.enc_opt = enc_opt
        self.dec_opt = dec_opt
        self.fr_lang = fr_lang
        self.en_lang = en_lang

    def train_on_example(self, example,
                         criterion, profile=False):

        # y_length = len(y)
        prep_ex = self.prepare_example(example)

        # Zero gradients of both optimizers
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()

        loss = self.compute_loss(prep_ex, criterion, profile)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)
        self.enc_opt.step()
        self.dec_opt.step()

        return loss.data[0]

    def compute_loss(self, prep_ex, criterion, profile):
        loss = []  # Added onto for each word

        x, y = prep_ex
        # Get size of input and target sentences
        x_length = x.size()[0]
        y_length = y.size()[0]

        # Run words through encoder
        tic = time.time()
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(word_inputs=x,
                                                       hidden=encoder_hidden,
                                                       )
        toc = time.time()
        if profile: logging.info("encoding time %.2f", toc - tic)
        # print("encoder_outputs",encoder_outputs.size(),encoder_hidden.size())
        # return
        # Prepare input and output variables
        tic = time.time()
        decoder_input = V(torch.LongTensor([[SOS_ID]]))
        # Use last hidden state from encoder to start decoder
        decoder_hidden = self.decoder.init_hidden(encoder_hidden)
        # print("decoder_hidden",decoder_hidden.size())
        if self.device_id:
            decoder_input = decoder_input.cuda(self.device_id)

        # Choose whether to use teacher forcing
        use_teacher_forcing = True  # random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(y_length):
                # print("decoder_input",decoder_input.size())
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              encoder_outputs,
                                                              decoder_hidden)
                # print(decoder_output[0].size(),target_variable[di].size())
                ex_loss = criterion(input=decoder_output, target=y[di])
                loss.append(ex_loss)
                # decoder_input = y[di]  # Next target is next input
                decoder_input = y[di].unsqueeze(0)  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(y_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              encoder_outputs,
                                                              decoder_hidden)
                # print(decoder_output[0].size(),target_variable[di].size())
                ex_loss = criterion(input=decoder_output, target=y[di])
                loss.append(ex_loss)

                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = V(torch.LongTensor([[ni]]))  # Chosen word is next input
                if self.device_id:
                    decoder_input = decoder_input.cuda()

                # Stop at end of sentence (not necessary when using known targets)
                if ni == EOS_token:
                    break

        toc = time.time()
        if profile: logging.info("decoding time %.2f", toc - tic)
        return sum(loss) / len(loss)

    def compute_loss_old(self, prep_ex, criterion, profile):
        loss = []  # Added onto for each word

        x, y = prep_ex
        # Get size of input and target sentences
        x_length = x.size()[0]
        y_length = y.size()[0]

        # Run words through encoder
        tic = time.time()
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(word_inputs=x,
                                                       hidden=encoder_hidden,
                                                       )
        toc = time.time()
        if profile: logging.info("encoding time %.2f", toc - tic)
        # print("encoder_outputs",encoder_outputs.size(),encoder_hidden.size())
        # return
        # Prepare input and output variables
        tic = time.time()
        decoder_input = V(torch.LongTensor([[SOS_ID]]))
        decoder_context = V(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = torch.cat([encoder_hidden[0, :, :], encoder_hidden[1, :, :]], dim=-1).unsqueeze(
            0)  # Use last hidden state from encoder to start decoder
        # decoder_hidden = self.decoder.init_hidden()  # Use last hidden state from encoder to start decoder
        # print("decoder_hidden",decoder_hidden.size())
        if self.device_id:
            decoder_input = decoder_input.cuda(self.device_id)
            decoder_context = decoder_context.cuda(self.device_id)

        # Choose whether to use teacher forcing
        use_teacher_forcing = True  # random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(y_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                                  decoder_context,
                                                                                                  decoder_hidden,
                                                                                                  encoder_outputs)
                # print(decoder_output[0].size(),target_variable[di].size())
                ex_loss = criterion(input=decoder_output, target=y[di])
                loss.append(ex_loss)
                decoder_input = y[di]  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(y_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                                  decoder_context,
                                                                                                  decoder_hidden,
                                                                                                  encoder_outputs)
                # print(decoder_output[0].size(),target_variable[di].size())
                ex_loss = criterion(input=decoder_output, target=y[di])
                loss.append(ex_loss)

                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = V(torch.LongTensor([[ni]]))  # Chosen word is next input
                if self.device_id:
                    decoder_input = decoder_input.cuda()

                # Stop at end of sentence (not necessary when using known targets)
                if ni == EOS_token:
                    break

        toc = time.time()
        if profile: logging.info("decoding time %.2f", toc - tic)
        return sum(loss) / len(loss)

    def prepare_example(self, example):
        raw_x, raw_y, x, y, weight, is_eng = example
        training_pair = variables_from_pair(x, y,
                                            input_lang=self.fr_lang,
                                            output_lang=self.en_lang)
        vx = training_pair[0]
        vy = training_pair[1]
        return vx, vy, weight
