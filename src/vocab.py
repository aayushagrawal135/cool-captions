from nltk.tokenize import word_tokenize
import torch.nn as nn

class Vocabulary():

    idx = 4

    def __init__(self):
        self.word_to_idx = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.idx_to_word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}

    def add_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token.lower() not in self.word_to_idx.keys():
                self.word_to_idx[token.lower()] = self.idx
                self.idx_to_word[self.idx] = token.lower()
                self.idx += 1

    def __len__(self):
        return len(self.word_to_idx)

    # def get_embedding(self, dims):
    #     return nn.Embedding(self.__len__(), dims, padding_idx = self.word_to_idx["<PAD>"])
