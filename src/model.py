import torch
import torchvision
from torchvision import models
from torch import nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.image_encoder = self.build_image_encoder()
        pass

    def forward(self, input_batch):
        (images, texts) = input_batch
        encoded_personality = self.encode_personality(texts['personality'])
        pass

    def encode_image(self, image):
        return self.image_encoder(image)
    
    def build_image_encoder(self):
        image_encoder = models.vgg16(pretrained=True)
        for param in image_encoder.parameters():
            param.requires_grad = False
        image_encoder.classifier = image_encoder.classifier[:2]
        return image_encoder

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
