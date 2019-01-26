import torch
import torch.nn as nn
from torch.autograd import Variable as V


class EncoderRNN(nn.Module):
    def __init__(self, invoc_size: int, vector_size: int, hidden_size: int, n_layers: int = 1, batch_first: bool = True,
                 bidi=True, device_id=None) -> None:
        super(EncoderRNN, self).__init__()
        self.input_size = invoc_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device_id = device_id
        self.embedding = nn.Embedding(num_embeddings=invoc_size,
                                      embedding_dim=vector_size)
        self.bidi = bidi
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size=vector_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=batch_first,
                          bidirectional=bidi
                          )
        self.no_pack_padded_seq = False

    def forward(self, word_inputs, hidden):
        # Note: works with only batch_size = 1
        # Note: we run this all at once (over the whole input sequence)
        max_len = len(word_inputs)
        # L x D
        embedded = self.embedding(word_inputs)
        # 1 x L x D, batch first is True
        embedded = embedded.view(1, max_len, -1)
        # 1 x L x D, 1 x H x D
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        if self.bidi:
            k = self.n_layers * 2
        else:
            k = self.n_layers * 1
        hidden = V(torch.zeros(k, batch_size, self.hidden_size))
        if self.device_id is not None:
            hidden = hidden.cuda(self.device_id)
        return hidden

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m


if __name__ == '__main__':
    encoder = EncoderRNN(invoc_size=10, vector_size=13, hidden_size=20)
    print(encoder)
    hidden = encoder.init_hidden()
    input_words = torch.LongTensor([1, 2, 3, 4])
    output, hidden = encoder(V(input_words), hidden)
    print('Output size:', output.size())
    print('Hidden size:', [h.size() for h in hidden])
