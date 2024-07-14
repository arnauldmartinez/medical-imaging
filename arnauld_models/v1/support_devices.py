#!/usr/bin/env python
# coding: utf-8


# # Imports

# In[1]:


import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
sys.path.append("/central/groups/CS156b/2024/clownmonkeys/")
from mdatasets import CustomDataset
import numpy as np
import pandas as pd
from torchsummary import summary
import tqdm
import math
# import wandb
from datetime import datetime

run_time = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

print("imports finished")

# In[2]:


ds = CustomDataset(
    "/groups/CS156b/data/train",
    "/groups/CS156b/data/student_labels/train2023.csv",
    condition="Pneumonia"
)

print("after dataset load")
# In[3]:

train_size = math.floor(0.9 * len(ds))
val_size = len(ds) - train_size
print("train/val:", train_size, val_size)
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_batch_size = 128
test_batch_size = 128

training_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=train_batch_size, shuffle=True, num_workers=12
)
validation_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=test_batch_size, shuffle=True, num_workers=12
)

# # NN

# In[4]:

print("before model init")


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


# In[5]:
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
else:
    print("cuda not available, using cpu")

model = CNN1().to(device)
print(summary(model, (1, 256, 256)))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# wandb.init(
#     project="cs156b",
#     entity="armeet-team",
#     config={
#         "model": model,
#         "loss_fn": loss_fn,
#         "optimizer": optimizer,
#         "batch_size": (train_batch_size, test_batch_size),
#         "dataset_split": (train_size, val_size),
#     },
# )

def train_one_epoch():
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(training_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # wandb.log({"train_loss": loss})
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(" batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0
    return running_loss / len(training_loader)


# In[8]:


EPOCHS = 100
epoch_number = 0
best_vloss = 1_000_000_000.0

print()
for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch()

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vinputs, vlabels in tqdm.tqdm(validation_loader):
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / len(validation_loader)
    # wandb.log({"avg_train_loss": avg_loss, "avg_val_loss": avg_vloss})
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    fpath = f"checkpoints/{run_time}"
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        model_path = f"./{fpath}/epoch_{epoch_number}-val_{avg_vloss}-train_{avg_loss}.pt"
        torch.save(model.state_dict(), model_path)
        print("saved to:", model_path)

    epoch_number += 1


# # Predict on Test

# wandb.finish()
