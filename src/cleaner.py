# %%
import os
import matplotlib.pyplot as plt
from torchvision import transforms

base_path = "../../ParlAI/data/"
images_path = base_path + "yfcc_images/"
image_names = os.listdir(images_path)

data_transform = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()]
)

count = 0
transformed_image = None
for image_name in image_names:
    try:
        image = plt.imread(images_path + image_name)
        transformed_image = data_transform(image)
        if (transformed_image.size()[0] == 3 and transformed_image.size()[1] == 224 and transformed_image.size()[2] == 224):
            continue
        else:
            raise
    except:
        print(transformed_image.size())
        print(image_name)
        # os.remove(images_path + image_name)
        count = count + 1
print(count)
# %%
