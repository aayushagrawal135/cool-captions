import torch
from torch import nn
from encoder import Encoder

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        pass

    def forward(self, input_batch):
        (images, texts) = input_batch
        encoded_personalities = self.encode_personality(texts['personality'])
        encoded_images = self.encoder.forward(images)
        pass

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
