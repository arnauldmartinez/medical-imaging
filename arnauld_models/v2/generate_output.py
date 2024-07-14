import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
sys.path.append("/central/groups/CS156b/2024/clownmonkeys/")
# from arnauld_models import support_devices
from mdatasets import CustomDataset
import numpy as np
import pandas as pd
from torchsummary import summary
import math
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(weights='IMAGENET1K_V1')

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

# Step 2: Load the state dictionary from the file
model_path = "./pleural_effusion.pt"
model_ft.load_state_dict(torch.load(model_path, map_location=device))

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test = CustomDataset(
    "/groups/CS156b/data/",
    "/groups/CS156b/data/student_labels/solution_ids.csv",
    solution=True,
    trans=transform
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
model_ft.eval()

print(torch.cuda.get_device_name())
print(os.cpu_count(), "cpus")

counter = 0
with torch.no_grad():
    for x, y in tqdm.tqdm(dataloader):
        # if counter == 3:
        #     break
        x, y = x.to("cuda"), y.to("cuda")
        
        preds = model_ft(x)
        all_preds = torch.cat([all_preds, preds], dim=0)
        # counter += 1

# torch.save(all_preds, "preds_enlarged.pt")

preds = all_preds.cpu().detach().numpy().squeeze()
print(preds.shape)
preds_dict = {'Id': test_ids_df['Id'][:len(preds)],
              'Pleural Effusion': preds}

preds_df = pd.DataFrame(preds_dict)
preds_df.to_csv('pleural_effusion_final.csv', index=False)
print("done saved")