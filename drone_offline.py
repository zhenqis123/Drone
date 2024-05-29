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

def train(model, device, train_loader, val_loader, optimizer, criterion, epochs):
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
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")


if __name__ == "__main__":
    dataset = OfflineDataset('/data/xiziheng/drone_data')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    model = DroneModel(4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = 30
    train(model, device, train_loader, val_loader, optimizer, criterion, epochs)