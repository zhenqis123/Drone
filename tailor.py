import cv2
import pandas as pd
import numpy as np
periods = np.array([[10,30],[36,75],[80,120],[124,162],[169,189]])
period_frames = periods*30
video = cv2.VideoCapture("/data/xiziheng/drone_data/01-水杉林1_3-晴天-T2.56.mp4")
assert video.isOpened()
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 加载数据集
data = pd.read_csv("/data/xiziheng/drone_data/01-水杉林1_3-晴天-T2.56.csv", skiprows=range(0, 139))
y1 = data['rcCommand[0]']
y2 = data['rcCommand[1]']
y3 = data['rcCommand[2]']
y4 = data['rcCommand[3]']
y = np.stack([y1, y2, y3, y4], axis=1)
y = y.astype(np.float32)

# 对水平的command进行归一化到(-1, 1)
minval = np.min(y1, axis=0)
maxval = np.max(y1, axis=0)
rangeval = maxval - minval
y1 = (y1 - minval) / rangeval * 2 - 1
# print(y1)

interval = np.round(y.shape[0]/video.get(cv2.CAP_PROP_FRAME_COUNT)).astype(int)
frames = []
commands_horizon = []
commands_all = []

index = 0
for i, period_frame in enumerate(period_frames):
    start_frame, end_frame = period_frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for j in range(start_frame, end_frame):
        ret, frame = video.read() # [H, W, C]
        assert ret == True
        frame = cv2.resize(frame, (512, 288))
        # cv2.imwrite(f"/data/xiziheng/drone_data/images/frame_{index}.jpg", frame)
        # 转换成灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        index+=1
        frame = frame.astype(np.float32) / 255 # 归一化到0,1
        frames.append(frame)
        command_all = np.sum(y[j*interval:j*interval+interval], axis=0)/interval
        commands_all.append(command_all)
        commands_horizon.append(np.sum(y1[j*interval:j*interval+interval])/interval)
# 添加镜像数据
index = 0
for i, period_frame in enumerate(period_frames):
    start_frame, end_frame = period_frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for j in range(start_frame, end_frame):
        ret, frame = video.read() # [H, W, C]
        assert ret == True
        frame = cv2.resize(frame, (512, 288))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f"/data/xiziheng/drone_data/images/frame_{index}.jpg", frame)
        index+=1
        frame = frame.astype(np.float32) / 255 # 归一化到0,1
        frame_flipped = cv2.flip(frame, 1)
        frames.append(frame_flipped)
        command_all = np.sum(y[j*interval:j*interval+interval], axis=0)/interval
        command_all[0] = -command_all[0]
        commands_all.append(command_all)
        commands_horizon.append(-np.sum(y1[j*interval:j*interval+interval])/interval)
frames = np.stack(frames, axis=0)
frames = np.expand_dims(frames, axis=3)
commands_all = np.stack(commands_all, axis=0)
commands_horizon = np.stack(commands_horizon, axis=0)
np.save("/data/xiziheng/drone_data/frames.npy", frames) # [1440, 2560, 3]
np.save("/data/xiziheng/drone_data/commands_all.npy", commands_all)
np.save("/data/xiziheng/drone_data/commands_horizon.npy", commands_horizon)

        
