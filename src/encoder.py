from torchvision import models
from torch import nn

class Encoder(nn.Module):
    def __init__(self, model_name = 'vgg16'):
        super(Encoder, self).__init__()
        self.image_encoder = self.get_pretrained_model(model_name)

    def forward(self, images):
        return self.image_encoder(images)

    # Pick model
    # View layer desciption and shave off last few layers accordingly
    # Set params to be non-trainable
    def get_pretrained_model(self, model_name = 'vgg16'):
        if model_name == 'vgg16':
            encoder = models.vgg16(pretrained=True)
            encoder.classifier = encoder.classifier[:2]
            for param in encoder.parameters():
                param.requires_grad = False
            return encoder
