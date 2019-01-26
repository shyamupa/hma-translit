import torch
import torch.nn as nn
from torch.autograd import Variable as V


class MonotonicDecoder(nn.Module):
    def __init__(self, input_size, outvoc_size, hidden_size, n_layers=1, device_id=None, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.device_id = device_id
        self.batch_first = batch_first
        self.n_layers = n_layers
        # concatenated_input_dim = input_size + hidden_size
        # print("concatenated_input_dim",concatenated_input_dim)
        self.decoder_rnn = nn.GRU(input_size=input_size,
                                  hidden_size=hidden_size,
                                  batch_first=True)
        self.char_lookup = nn.Embedding(outvoc_size, hidden_size)
        self.out = nn.Linear(in_features=hidden_size,
                             out_features=outvoc_size)

    def forward(self, prev_word, idx, last_hidden, encoder_outputs):
        # set prev_output_vec for first lstm step as BEGIN_WORD
        if self.batch_first:
            encoder_outputs = encoder_outputs.transpose(0, 1)
        prev_word_vec = self.char_lookup(prev_word)
        attended_vec = encoder_outputs[idx]
        decoder_input = torch.cat((prev_word_vec, attended_vec), dim=1)
        decoder_output, hidden = self.decoder_rnn(decoder_input.unsqueeze(0), last_hidden)
        return decoder_output, hidden

    def init_hidden(self, batch_size=1):
        k = self.n_layers * 1
        hidden = V(torch.zeros(k, batch_size, self.hidden_size))
        if self.device_id is not None:
            hidden = hidden.cuda(self.device_id)
        return hidden

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m
