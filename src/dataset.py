from torch.utils.data import Dataset, DataLoader
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import json

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CaptionDataset(Dataset):
    def __init__(self, captions_file, personalities_path, images_dir, data_transform = None):
        self.captions = pd.read_json(captions_file)
        with open(personalities_path, 'r') as file:
            self.personalities = json.loads(file.read())
        self.image_names = os.listdir(images_dir)
        self.images_dir = images_dir
        self.data_transform = data_transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = plt.imread(self.images_dir + image_name)
        # Similar to SELECT * FROM self.captions WHERE `image_hash` = image_name;
        # We did a split below since `image name = image_hash.jpg` in the dataset
        info = self.captions[self.captions['image_hash'] == image_name.split('.')[0]]

        if (self.data_transform != None):
            transformed_image = self.data_transform(image)
        # `info` is of the type `Dataframe`, pick first (and the only) row and convert it into dict
        # Did this conversion since there were issues using `iter` and `next` on batch of `Dataframe`
        # column name is the key and value is the value
        info_as_dict = info.iloc[0].to_dict()
        info_as_dict['personality'] = self.get_personality_type(info_as_dict['personality'])
        return transformed_image, info_as_dict

    def __len__(self):
        return len(self.image_names)

    def get_personality_type(self, personality):
        for personality_type, personality_list in self.personalities.items():
            if personality in personality_list:
                return personality_type
        return "neutral"
