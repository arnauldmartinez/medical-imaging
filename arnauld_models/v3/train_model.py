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

print("imports complete")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

ds = CustomDataset(
    "/groups/CS156b/data/train",
    "/groups/CS156b/data/student_labels/train2023.csv",
    trans=transform
)

print("dataset loaded")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SPLIT_RATIO = 0.9
train_size = math.floor(SPLIT_RATIO * len(ds))
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

dataloaders = {'train':training_loader, 'val':validation_loader}
dataset_sizes = {'train':len(training_loader), 'val':len(validation_loader)}

print("training/validation split made")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = outputs
                        # _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(torch.abs(preds - labels.data) < 0.02)
                    # print(f"running_loss += {loss.item()} * {inputs.size(0)}")
                    # print(f'preds: {preds}\tlabels.data: {labels.data}')
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

model_ft = models.resnet18(weights='IMAGENET1K_V1')
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 9)

model_ft = model_ft.to(device)

criterion = nn.MSELoss()

optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=2)

print("After model train")

model_path = f"./FINAL.pt"
torch.save(model_ft.state_dict(), model_path)