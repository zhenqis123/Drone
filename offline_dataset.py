from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import torch
class OfflineDataset(Dataset):
    def __init__(self, root_dir=".", seq_len=10, new_shape=(64, 64)):
        path_action = Path(root_dir) / "01-水杉林1_3-晴天-T2.56.csv"
        path_input = Path(root_dir) / "01-水杉林1_3-晴天-T2.56.mp4"
        if not path_action.exists() or not path_input.exists():
            raise RuntimeError("Could not find data")
        video = cv2.VideoCapture(str(path_input))
        assert video.isOpened()
        self.root_dir = root_dir
        self.total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video = video
        self.seq_len = seq_len
        self.data_interval = 66
        self.new_shape = new_shape
        
        data_y = pd.read_csv(path_action, skiprows=range(0, 139))
        y1 = data_y['rcCommand[0]']
        y2 = data_y['rcCommand[1]']
        y3 = data_y['rcCommand[2]']
        y4 = data_y['rcCommand[3]']
        self.y = np.stack([y1, y2, y3, y4], axis=1)
        self.y = self.y.astype(np.float32)
        self.input = np.load(f"{self.root_dir}/input.npz")
        self.label = np.load(f"{self.root_dir}/output.npz")
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
        x = self.input['arr_0'][idx]
        x = torch.from_numpy(x)
        y = self.label['arr_0'][idx]
        y = torch.from_numpy(y)
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