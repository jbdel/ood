import os
import pandas as pd
import numpy as np
import torch
from PIL import ImageFile, Image
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LarsonDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root_dir,
                 input_transforms,
                 label_transforms):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.input_transforms = input_transforms
        self.label_transforms = label_transforms

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
        if torch.is_tensor(index):
            index = index.tolist()

        image_arr = self.image_arrs[index]

        skeletal_age = self.df.iloc[index, 1]
        male = self.df.iloc[index, 2]
        try:
            real = self.df.iloc[index, 3]
        except:
            real = True

        inputs = [
            compose_obj({
                "image_arr": image_arr,
                "skeletal_age": skeletal_age,
                "male": male,
                "real": real,
            }) for compose_obj in self.input_transforms
        ]

        label = self.label_transforms({
            "image_arr": image_arr,
            "skeletal_age": skeletal_age,
            "male": male,
            "real": real,
        })

        metadata = {
            "id": self.df.iloc[index, 0],
            "index": index,
            "skeletal_age": skeletal_age,
            "sex": "M" if male else "F",
            "real": int(real),
        }
        return inputs, label, metadata
