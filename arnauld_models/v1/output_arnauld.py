import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
sys.path.append("/central/groups/CS156b/2024/clownmonkeys/")
# from arnauld_models import support_devices
from mdatasets import CustomDataset
import numpy as np
import pandas as pd
from torchsummary import summary
import math
import tqdm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            activation_fn,
        )

    def forward(self, x):
        return self.layers(x)

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.id = "cnn"
        self.input = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.block1 = Block(
            in_channels=16, out_channels=32, kernel_size=3, activation_fn=nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = Block(
            in_channels=32, out_channels=32, kernel_size=3, activation_fn=nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = Block(
            in_channels=32, out_channels=32, kernel_size=3, activation_fn=nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = Block(
            in_channels=32, out_channels=32, kernel_size=3, activation_fn=nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(32 * 20 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.act = nn.Tanh()

    def forward(self, x):
        out = self.input(x)

        out = self.block1(out)
        out = self.pool1(out)

        out = self.block2(out)
        out = self.pool2(out)

        out = self.block3(out)
        out = self.pool3(out)

        out = self.block4(out)
        # print("after last block", out.shape)
        out = self.flatten(out)
        # print("after flatten", out.shape)
        out = self.mlp(out)
        out = self.act(out)

        return out


model = CNN1().to("cuda")
model.load_state_dict(torch.load("/groups/CS156b/2024/clownmonkeys/checkpoints/05_29_2024_05:10:43/epoch_0-val_1.0004185438156128-train_0.997395787546985.pt", map_location=torch.device('cuda')))

test = CustomDataset(
    "/groups/CS156b/data/",
    "/groups/CS156b/data/student_labels/solution_ids.csv",
    test=True,
)

batch_size = 512
dataloader = torch.utils.data.DataLoader(
    test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=12
)

test_ids_df = pd.read_csv("/groups/CS156b/data/student_labels/solution_ids.csv")
all_preds = torch.zeros(0, 1).to("cuda")
model.eval()

print(torch.cuda.get_device_name())
print(os.cpu_count(), "cpus")

counter = 0
with torch.no_grad():
    for x, y in tqdm.tqdm(dataloader):
        # if counter == 3:
        #     break
        x, y = x.to("cuda"), y.to("cuda")
        
        preds = model(x)
        all_preds = torch.cat([all_preds, preds], dim=0)
        # counter += 1

torch.save(all_preds, "FINAL.pt")

preds = all_preds.cpu().detach().numpy().squeeze()
print(preds.shape)
preds_dict = {'Id': test_ids_df['Id'][:len(preds)],
              'Pneumonia': preds}

# preds_dict = {'Id': test_ids_df['Id'][:len(preds)], 'No Finding' : preds[:, 0], 'Enlarged Cardiomediastinum' : preds[:, 1],
#               'Cardiomegaly' : preds[:, 2], 'Lung Opacity' : preds[:, 3], 'Pneumonia' : preds[:, 4],
#               'Pleural Effusion' : preds[:, 5], 'Pleural Other' : preds[:, 6], 'Fracture' : preds[:, 7],
#               'Support Devices' : preds[:, 8]}

preds_df = pd.DataFrame(preds_dict)
preds_df.to_csv('arnauld_submission_pneumonia.csv', index=False)
print("done saved")