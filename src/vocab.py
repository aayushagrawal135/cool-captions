from nltk.tokenize import word_tokenize
import torch.nn as nn

class Vocabulary():

    idx = 4

    def __init__(self):
        self.sos = "<SOS>"
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        self.unk = "<UNK>"
        self.max_sequence_len = 0
        self.word_to_idx = {self.sos: 0, self.eos: 1, self.pad: 2, self.unk: 3}
        self.idx_to_word = {0: self.sos, 1: self.eos, 2: self.pad, 3: self.unk}

    def add_sentence_tokens_to_corpus(self, sentence):
        tokens = word_tokenize(sentence)
        if (len(tokens) > self.max_sequence_len):
            self.max_sequence_len = len(tokens)
        for token in tokens:
            if token.lower() not in self.word_to_idx.keys():
                self.word_to_idx[token.lower()] = self.idx
                self.idx_to_word[self.idx] = token.lower()
                self.idx += 1

    def get_sequence_of_token_indices(self, sentence, pad_window):
        tokens = word_tokenize(sentence)
        seq_tokens = [self.word_to_idx[token.lower()] for token in tokens]
        padding_len = self.max_sequence_len - len(seq_tokens)


    def __len__(self):
        return len(self.word_to_idx)

    def get_embedding(self, dims):
        return nn.Embedding(self.__len__(), dims, padding_idx = self.word_to_idx["<PAD>"])
