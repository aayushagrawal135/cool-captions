from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CaptionDataset

# Hardcoded path variables for now
base_path = "../../ParlAI/data/"
captions_path = base_path + "personality_captions/train.json"
images_path = base_path + "yfcc_images/"

# We need to `Resize` since images are of varying spatial dimensions
# Before, `Resize`, we need to make it PIL image - since Resize does not work on numpy arrays
# Eventually, we convert it to Tensor, since we want to work with tensors
data_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()]
)

dataset = CaptionDataset(captions_path, images_path, data_transform)
data_loader = DataLoader(dataset, batch_size=4)
iterator = iter(data_loader)

# `images` is of dimension (4, 3, 224, 224) where 4 is the batch size, 3 is the number of channels (RGB)
# and 224*224 is the spatial dimension
# `texts` is a dictionary with 3 keys: `personality`, `image_hash` and `comment`
# value for each key is of type `list` of size 4 (batch size)
images, texts = next(iterator)
