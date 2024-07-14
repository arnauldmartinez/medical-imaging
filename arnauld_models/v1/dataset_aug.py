from glob import glob
import csv

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

condition_map = {"No Finding": 7,
                 "Enlarged Cardiomediastinum": 8, 
                 "Cardiomegaly": 9, 
                 "Lung Opacity": 10,
                 "Pneumonia": 11, 
                 "Pleural Effusion": 12,
                 "Pleural Other": 13, 
                 "Fracture": 14, 
                 "Support Devices": 15}

class CustomDataset(Dataset):
    def __init__( self, data_path, labels_path, condition=None, test=False, post_sigmoid=False ):
        self.data_path = data_path
        self.labels = self.get_labels(labels_path, condition)
        self.condition = condition
        self.test = test
        self.post_sigmoid = post_sigmoid

    def get_labels(self, labels_path, condition):
        df = pd.read_csv(labels_path)
        if not condition == None:
            df = df[['Path', condition]]
            df = df.dropna().reset_index(drop=True)
        return df

    def __len__(self):
        return 2000
        # return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.labels.iloc[idx]["Path"]

        if "solution" in img_path or "train" in img_path:
            img_path = (
                img_path.lstrip("solution") if self.test else img_path.lstrip("train")
            )
        if self.test:
            img_path = self.data_path + img_path
        else:
            img_path = self.data_path + "/" + img_path

        image = Image.open(img_path)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        transform = transforms.Compose([transforms.PILToTensor()])

        x = torch.tensor(transform(image), dtype=torch.float)
        x_flipped = torch.tensor(transform(flipped_image), dtype=torch.float)

        x = F.adaptive_avg_pool2d(x.unsqueeze(0), (256, 256)).squeeze(0)
        x_flipped = F.adaptive_avg_pool2d(x_flipped.unsqueeze(0), (256, 256)).squeeze(0)
        x_dull = x_flipped / 2

        curr_label = None
        if self.condition == None:
            curr_label = self.labels.iloc[idx, 7:]
            curr_label = [0.0 if pd.isna(n) else n for n in curr_label]
        else:
            curr_label = self.labels.iloc[idx][self.condition]
            if pd.isna(curr_label):
                curr_label = 0.0
            curr_label = [curr_label]
        y = torch.tensor(curr_label, dtype=torch.float)
        if self.post_sigmoid: y = (y + 1) / 2

        # load 1 hot encoded tensor into this variable
        return ([x, x_flipped, x_dull], y)

def get_normalized_dataset():
    ds = CustomDataset(
        data_path="/groups/CS156b/data/train",
        labels_path="/groups/CS156b/data/student_labels/train2023.csv",
        condition="Support Devices"
    )

    arr_pos = []

    for i in range(len(ds)):
        imgs, label = ds[i]
        




if __name__ == "__main__":
    ds = CustomDataset(
        data_path="/groups/CS156b/data/train",
        labels_path="/groups/CS156b/data/student_labels/train2023.csv",
        condition="Cardiomegaly"
    )
    # ds = CustomDataset(
    #     "/groups/CS156b/data/test",
    #     "/groups/CS156b/data/student_labels/solution_ids.csv",
    #     test=True,
    # )

    x, y = ds[6]

    print(x.shape)
    print(y)

    # plt.imshow(x[0,:,:], cmap='gray')
    # plt.axis('off')
    # plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # print(x.shape)
