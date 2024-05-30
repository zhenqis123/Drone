import gym
import ale_py
import torch
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.optim as optim

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ncps.torch import CfC
from ncps.datasets.torch import AtariCloningDataset
from offline_dataset import OfflineDataset
# import pathlib
import pathlib
from matplotlib import pyplot as plt

class DroneModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = ConvBlock()
        self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)
        return x, hx

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean((-1, -2))
        return x

def train(model, device, train_loader, val_loader, optimizer, criterion, epochs, sechduler):
    loss_all = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}, loss: {loss.item()/len(train_loader)}")
        scheduler.step()
        loss_all.append(running_loss/len(train_loader))
        # print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")
    plt.plot(loss_all)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    # model.eval()
def eval(model, device, val_loader, criterion):
    model.eval()
    A = np.empty((0, 4))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            a = outputs.view(outputs.size(0)*outputs.size(1), -1)
            b = labels.view(labels.size(0)*labels.size(1), -1)
            tem = (a - b)/b
            A = np.concatenate((A, tem.cpu().numpy()),axis=0)
        condition = np.abs(A) < 0.1
        new_array = np.where(condition, 1, 0)
        acc = np.array([0, 0, 0, 0])
        acc[0] = np.sum(new_array[:,0])/new_array.shape[0]
        acc[1] = np.sum(new_array[:,1])/new_array.shape[0]
        acc[2] = np.sum(new_array[:,2])/new_array.shape[0]
        acc[3] = np.sum(new_array[:,3])/new_array.shape[0]
        print('acc:', acc)
        print('mean:', np.mean(acc))
        
            

if __name__ == "__main__":
    # Define the directory path where the images are located
    image_dir = pathlib.Path('/data/xiziheng')

    # Define the file extension of the images to be deleted
    file_extension = '.png'

    # Iterate over all files in the directory
    for file in image_dir.iterdir():
        # Check if the file is a regular file and has the specified file extension
        if file.is_file() and file.suffix == file_extension:
            # Delete the file
            file.unlink()

    
    dataset = OfflineDataset('/data/xiziheng/drone_data')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)
    model = DroneModel(4)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = 300
    train(model, device, train_loader, val_loader, optimizer, criterion, epochs, scheduler)
    torch.save(model.state_dict(), 'drone_model.pth')
    eval(model, device, val_loader, criterion)