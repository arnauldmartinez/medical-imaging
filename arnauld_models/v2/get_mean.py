import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
sys.path.append("/central/groups/CS156b/2024/clownmonkeys/")
# from arnauld_models import support_devices
from mdatasets import CustomDataset
import numpy as np
import pandas as pd
from torchsummary import summary
import math
import tqdm
import time
from tempfile import TemporaryDirectory

print("imports finished")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

ds = CustomDataset(
    "/groups/CS156b/data/train",
    "/groups/CS156b/data/student_labels/train2023.csv",
    condition="Pleural Other",
    trans=transform
)

running_sum = 0
for i in range(len(ds)):
    running_sum += ds[i][1]

print(f"average for Pleural Other: {float(running_sum) / len(ds)}")