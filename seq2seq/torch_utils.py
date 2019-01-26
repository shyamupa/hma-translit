import torch
from torch.autograd import Variable as V

from seq2seq.constants import EOS_ID, EOS_token


def variables_from_pair(x, y, input_lang=None, output_lang=None):
    input_variable = variable_from_sentence(input_lang, x)
    target_variable = variable_from_sentence(output_lang, y)
    return input_variable, target_variable


def variable_from_sentence(lang, sentence, device_id=None):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_ID)
    var = V(torch.LongTensor(indexes).view(-1, 1))
    #     print('var =', var)
    if device_id is not None:
        var = var.cuda(device_id)
    return var


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def pad_batch(batch, pad_unit):
    lengths = [len(i) for i in batch]
    max_length = max(lengths)
    for ex in batch:
        padding = (max_length - len(ex)) * [pad_unit]
        ex += padding
    return batch, lengths
