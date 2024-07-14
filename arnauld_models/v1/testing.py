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

test_ids_df = pd.read_csv("/groups/CS156b/data/student_labels/test_ids.csv")
all_preds = torch.load("preds.pt", map_location=torch.device('cuda'))

preds = all_preds.cpu().detach().numpy().squeeze()
print(preds.shape)
preds_dict = {'Id': test_ids_df['Id'][:len(preds)],
              'Support Devices': preds}

preds_df = pd.DataFrame(preds_dict)
preds_df.to_csv('new_arnauld_submission.csv', index=False)
print("done saved")