import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder

class Net(nn.Module):

    def __init__(self, vocab):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(encoding_dim=4096, embedding_dim=300, hidden_dim=400, vocab=vocab)

    def forward(self, input_batch):
        (images, captions, lengths) = input_batch
        # For vgg16, [4, 4096]
        encoded_images = self.encoder.forward(images)
        # list of tensors of [batch, 1, hidden] of len max(lengths)
        output_sequence = self.decoder.forward(encoded_images, lengths)
        return output_sequence

    # size of list will be the batch size
    # so far we have 3 emotions only - positive, negative and neutral
    # return one-hot encoded tensors
    def encode_personality(self, personality_list):
        pers_tensors = list()
        for pers in personality_list:
            if pers == "positive":
                pers_tensors.append(torch.FloatTensor([0, 0, 1]))
            elif pers == "neutral":
                pers_tensors.append(torch.FloatTensor([0, 1, 0]))
            else:
                pers_tensors.append(torch.FloatTensor([1, 0, 0]))
        return torch.stack(pers_tensors)

    def build_text_encoder(self):
        pass
