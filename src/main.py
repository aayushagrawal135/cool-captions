# %%
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
from dataset import CaptionDataset
from embedding import Embedding
from model import Net
from utils import custom_collate_fn
from vocab import Vocabulary

# Hardcoded path variables for now
base_path = "../../ParlAI/data/"
captions_path = base_path + "personality_captions/train.json"
personalities_path = base_path + "personality_captions/personalities.json"
images_path = base_path + "yfcc_images/"
pretrained_embedding_file = base_path + "GoogleNews-vectors-negative300.bin"

# We need to `Resize` since images are of varying spatial dimensions
# Before, `Resize`, we need to make it PIL image - since Resize does not work on numpy arrays
# Eventually, we convert it to Tensor, since we want to work with tensors
data_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()]
)

dataset = CaptionDataset(captions_path, personalities_path, images_path, data_transform)
vocab = dataset.get_vocab()
Embedding.set(vocab)

window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, vocab=vocab)

train_loader = DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn=collate_fn)
iterator = iter(train_loader)

# `images` is of dimension (4, 3, 224, 224) where 4 is the batch size, 3 is the number of channels (RGB)
# and 224*224 is the spatial dimension
# `texts` is a dictionary with 3 keys: `personality`, `image_hash` and `comment`
# value for each key is of type `list` of size 4 (batch size)

images, padded_captions, lengths = next(iterator)

model = Net(vocab)
batch = (images, padded_captions, lengths)
model.forward(batch)

# Params/vars for training
n_epochs = 1
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def loss_fn(output, target):
    output = torch.permute(output, (1, 0, 2))
    embeds = Embedding.get(target)
    loss = criterion(output, embeds)
    return loss

# Train loop
def train(epoch):
    model.train()
    for batch_idx, (images, padded_captions, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model.forward((images, padded_captions, lengths))
        # output => (len of sentence * num of images in batch * vector size)
        
        loss = loss_fn(output, padded_captions)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch} Loss : {loss.item()}")

train(1)

# 
# for i in range(log_output.size()[0]):
#     idx = torch.argmax(log_output[i][0])
#     print(self.vocab.idx_to_word[idx.item()])
# %%
