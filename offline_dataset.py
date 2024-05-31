from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import torch
import random
class OfflineDataset(Dataset):
    def __init__(self, root_dir=".", seq_len=10, scaling_factor=1, transform=None):
        self.frames = np.load(f"{root_dir}/frames.npy")
        self.commands_horizon = np.load(f"{root_dir}/commands_horizon.npy")
        self.commands_all = np.load(f"{root_dir}/commands_all.npy")
        self.seq_len = seq_len
        self.total_frames = self.frames.shape[0]
        shape = self.frames.shape[1:3]
        self.new_shape = (int(shape[0]*scaling_factor), int(shape[1]*scaling_factor))
        self.root_dir = root_dir
        # self.save_dataset()
        
    def __len__(self):
        return int(self.total_frames/self.seq_len)
    
    # def __getitem__(self, idx):
    #     x = self.get_frames(idx)
    #     x = np.transpose(x, [0, 3, 1, 2])
    #     x = x.astype(np.float32) / 255
    #     x = torch.from_numpy(x)
    #     y = []
    #     for i in range(self.seq_len):
    #         y.append(self.y[idx*self.seq_len*self.data_interval+i*self.data_interval])
    #     y = np.stack(y, axis=0)
    #     y = torch.tensor(y)
        
    #     return x, y 
    
    def __getitem__(self, idx):
        x = self.frames[idx*self.seq_len:(idx+1)*self.seq_len]
        y = self.commands_horizon[idx*self.seq_len:(idx+1)*self.seq_len]
        x = np.transpose(x, [0, 3, 1, 2])
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y = y.unsqueeze(-1)
        return x, y
    
    
    def get_frames(self, idx):
        frames = []
        for i in range(self.seq_len):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, idx*self.seq_len+i)
            ret, frame = self.video.read()
            assert ret == True
            frame = cv2.resize(frame, self.new_shape)
            frames.append(frame)
        stacked_frames = np.stack(frames, axis=0)
        return stacked_frames

    def save_dataset(self):
        datax = []
        datay = []
        for i in range(len(self)):
            x, y = self[i]
            x = x.numpy()
            y = y.numpy()
            datax.append(x)
            datay.append(y)
        datax = np.stack(datax, axis=0)
        datay = np.stack(datay, axis=0)
        np.savez(f"{self.root_dir}/input.npz", datax)
        np.savez(f"{self.root_dir}/output.npz", datay)