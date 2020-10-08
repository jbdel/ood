from transforms import (
    RepeatGrayscaleChannels,
    Clahe,
    RandomSquareCrop,
    RandomHorizontalFlip,
    Transpose,
    ToTensor,
    ToFloat,
    ToLong,
)
import pandas as pd
import numpy as np
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import utils
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

test_input_transforms = Compose([
    RepeatGrayscaleChannels(3),
    Transpose(),
    ToTensor(),
    ToFloat(),
])

train_input_transforms = Compose([
    RepeatGrayscaleChannels(3),
    Clahe(),  # noop at the moment
    RandomSquareCrop((224, 224)),
    RandomHorizontalFlip(),
    Transpose(),
    ToTensor(),
    ToFloat(),
])

label_transform = Compose([
    ToTensor(),
    ToLong(),
])


class LarsonDataset(Dataset):

    def __init__(self,
                 name=None,
                 mode=None,
                 root_dir=None,
                 csv_file=None,
                 load_memory=False):
        self.name = name
        self.mode = mode
        assert self.mode in ['idd', 'ood']
        assert self.name in ['retina', 'skeletal-age', 'mura', 'mimic-crx']
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.load_memory = load_memory

        # Load into cpu ?
        if self.load_memory:
            self.image_arrs = []
            for index in range(len(self.df)):
                image_filename = f"{self.root_dir}/{self.df.iloc[index, 0]}.npy"
                try:
                    self.image_arrs.append(np.load(image_filename))
                except:
                    print(f"image_filename: {image_filename}")
                    raise Exception

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        if self.load_memory:
            image_arr = self.image_arrs[index]
        else:
            image_filename = f"{self.root_dir}/{self.df.iloc[index, 0]}.npy"
            image_arr = np.load(image_filename)

        if self.mode == "idd":
            inputs = train_input_transforms(image_arr)
        elif self.mode == "ood":
            inputs = test_input_transforms(image_arr)
        else:
            raise NotImplementedError

        metadata = utils.get_metadata(self.name, self.df, index)
        label = label_transform(utils.get_label(self.name, self.df, index))

        return inputs, label, metadata
