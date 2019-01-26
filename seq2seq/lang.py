from seq2seq.constants import SOS_token, EOS_token, SOS_ID, EOS_ID, UNK, UNK_ID


class Lang:
    def __init__(self, name):
        self.name = name
        self.vocab = set()
        self.word2index = {SOS_token: SOS_ID, EOS_token: EOS_ID, UNK: UNK_ID}
        self.word2count = {}
        self.index2word = {SOS_ID: SOS_token, EOS_ID: EOS_token, UNK_ID: UNK}
        self.n_words = len(self.word2index)  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        self.vocab.add(word)

    def compute_maps(self):
        words = sorted(list(self.vocab))
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
