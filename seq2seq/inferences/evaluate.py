import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch import optim
import torch.nn.functional as F
from seq2seq.constants import EOS_token, SOS_ID, EOS_ID
from seq2seq.constants import SOS_token
from seq2seq.torch_utils import variable_from_sentence


class Inference:
    def __init__(self, encoder, decoder, input_lang, output_lang, device_id=None):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang, self.output_lang = input_lang, output_lang
        self.device_id = device_id

    def infer_on_example(self, sentence):
        self.encoder.eval()
        self.decoder.eval()
        ans = self.run_inference(sentence)
        self.encoder.train()
        self.decoder.train()
        return ans

    def run_inference(self, sentence, max_length=100):
        raise NotImplementedError
