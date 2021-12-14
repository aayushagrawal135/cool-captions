import torch
from torch import nn

class Embedding():

    embeddings = None

    @classmethod
    def set(self, vocab):
        Embedding.embeddings = nn.Embedding(len(vocab), 300, padding_idx = vocab.word_to_idx[vocab.pad])


    # def __init__(self, vocab):
    #     # self.model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    #     self.embeddings = nn.Embedding(len(vocab), 300, padding_idx = vocab.word_to_idx[vocab.pad])
    #     pass

    # Batch x dims
    @classmethod
    def get(self, batch_indices):
        embeds = list()
        for indices in batch_indices:
            embeds.append(torch.index_select(Embedding.embeddings.weight, 0, indices))
        embeds = torch.stack(embeds, dim=0)
        return embeds
    
    def similar(self, word):
        pass
