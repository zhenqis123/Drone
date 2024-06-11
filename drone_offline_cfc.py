import gym
import ale_py
import torch
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from torch.autograd import Variable
from ncps.torch import CfC
from ncps.datasets.torch import AtariCloningDataset
from offline_dataset import OfflineDataset
# import pathlib
import pathlib
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from ncps.torch import LTC
from ncps.wirings import AutoNCP, NCP
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=500)
argparser.add_argument('--batch_size', type=int, default=8)
argparser.add_argument('--seq_len', type=int, default=256)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--test_size', type=float, default=0.1)
argparser.add_argument('--train', action='store_false')
argparser.add_argument('--test', action='store_false')

args = argparser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DroneModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = ConvBlock()
        # self.conv_block = UNet(1, 256)
        self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions, mixed_memory=True)
        # wirings = AutoNCP(64, 1)
        # self.rnn = LTC(256, wirings, batch_first=True, mixed_memory=True)

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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, 256)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn4(self.conv5(x)))
        x = x.mean((-1, -2))
        x = F.relu(self.fc(x))
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x).mean((-1, -2))
        return logits
    
def train(model, train_loader, val_loader, optimizer, criterion, epochs, sechduler):
    loss_all = []
    accs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        acc = 0
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            acc_tem = test(outputs, labels)
            acc += acc_tem 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}, loss: {loss.item()/inputs.size(0)}, acc: {acc_tem}")
        scheduler.step()
        loss_all.append(running_loss/len(train_loader))
        accs.append(acc/len(train_loader))
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}, acc: {acc/len(train_loader)}")
        # print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")
    fig = plt.figure()
    
    ax1 = fig.add_subplot(211)
    plt.plot(loss_all)
    ax1.set_title('Loss')
    
    ax2 = fig.add_subplot(212)
    ax2.plot(accs)
    ax2.set_title('Acc')
    
    plt.tight_layout()
    plt.savefig(f'./saved_image/loss_acc_cfc_{epochs}.png')
    plt.cla()
    
def test(outputs, labels):
    outputs = outputs.detach()
    labels = labels.detach()
    a = outputs.view(outputs.size(0)*outputs.size(1), -1)
    b = labels.view(labels.size(0)*labels.size(1), -1)
    # import pdb;pdb.set_trace()
    tem = (a - b)/b
    tem = tem.cpu().numpy()
    condition = np.abs(tem) < 0.1
    new_array = np.where(condition, 1, 0)
    acc = np.sum(new_array, axis=0)/new_array.shape[0]
    return acc

def eval(model, val_loader, criterion):
    model.eval()
    labels_all = np.empty((0, 1))
    outputs_all = np.empty((0, 1))

    ## 用于保存视频的数据
    inputs_all = np.empty((0, 1, 288, 512))
    A = np.empty((0, 1))
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs, _ = model(inputs)
            a = outputs.view(outputs.size(0)*outputs.size(1), -1)
            b = labels.view(labels.size(0)*labels.size(1), -1)
            tem = (a - b)/b
            A = np.concatenate((A, tem.cpu().numpy()),axis=0)
            inputs_all = np.concatenate((inputs_all, inputs.cpu().numpy()), axis=0)
            labels_all = np.concatenate((labels_all, b.cpu().numpy()), axis=0)
            outputs_all = np.concatenate((outputs_all, a.cpu().numpy()), axis=0)
        saveVideo(f'./saved_image/output_test_{epochs}.mp4',
                  inputs_all,
                  outputs_all,
                  labels_all)
        generate_saliency_map(model, inputs_all)
        condition = np.abs(A) < 0.5
        new_array = np.where(condition, 1, 0)
        acc = np.sum(new_array, axis=0)/new_array.shape[0]
        print('test_acc:', acc)
        print('test_acc_mean:', np.mean(acc))
        plt.plot(labels_all[:500], label='labels', color='r')
        plt.plot(outputs_all[:500], label='outputs', color='b')
        plt.legend()
        plt.savefig(f'./saved_image/test_cfc_{epochs}.png')
        plt.clf()
        
def Saliency(model, device, val_loader):
    model.eval()
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        generate_saliency_map(model, inputs)
        break
    
def saveVideo(video_path, frames, output_commands, labels):
    tensor = frames
    tensor = tensor.view(-1, 1, 288, 512)
    output_commands = output_commands.view(-1, 1)
    labels = labels.view(-1, 1)
    # 确保张量是一个 4D 张量 (batch_size, channels, height, width)
    assert tensor.ndim == 4, "Tensor should be a 4D tensor (batch_size, channels, height, width)"
    
    # 将 PyTorch 张量从 CHW 转为 HWC 格式
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor*255
    # 将张量转换为 numpy 数组并确保类型为 uint8
    frames = tensor.cpu().numpy().astype(np.uint8)
    
    # 获取帧的尺寸
    height, width, channels = frames[0].shape
    
    # 确保图像是三通道的
    assert channels == 1, "Each image must have 3 channels (RGB)"
    
    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height), isColor=False)
    
    # 将每一帧写入视频
    for i in range(len(frames)):
        frame = frames[i].copy()
        cv2.putText(frame, f"Output: {output_commands[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {labels[i]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        video_writer.write(frame)
    
    # 释放 VideoWriter 对象
    video_writer.release()

def generate_saliency_map(model, input_image):
    # Set the model in evaluation mode
    model.eval()
    
    # initialize the video writer
    videowriter = cv2.VideoWriter('saliency_map.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2, (512, 288*2), isColor=False)
    saliencys = []
    for i in range(input_image.size(0)):
        for j in range(input_image.size(1)):
            image = input_image[i, j]
            image = image.unsqueeze(0).unsqueeze(0)
            input_var = Variable(image, requires_grad=True).to(device)
            # Forward pass
            output, _ = model(input_var)

            # Zero gradients
            model.zero_grad()

            # Backward pass
            output.backward()

            # Convert the input gradient from a tensor to a numpy array
            saliency = input_var.grad.data.cpu().numpy()

            saliency = np.abs(saliency)
            saliency = np.mean(saliency, axis=2)[0][0]
            ## normalize the saliency map
            saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-12)
            saliency = saliency**2
            saliency = np.expand_dims(saliency, axis=2) # cv2期望灰度图有通道维度
            saliency = (saliency * 255).astype(np.uint8)
            # saliency = cv2.equalizeHist(saliency)
            
            image = image.squeeze(0).squeeze(0)
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # image = np.expand_dims(image, axis=2)
            
            result = np.vstack((image, saliency))
            # cv2.imwrite('saliency.png', result)
            videowriter.write(result)
    videowriter.release()
    return saliency

if __name__ == "__main__":

    Filp_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0)
    ])
    dataset = OfflineDataset('/data/xiziheng/drone_data', transform=Filp_transform, seq_len=args.seq_len)
    val_size = int(args.test_size * len(dataset))
    train_size = len(dataset) - val_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset = Subset(dataset, range(0, train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    model = DroneModel(1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
    
    criterion = nn.MSELoss()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    epochs = args.epochs
    if args.train:
        train(model, train_loader, val_loader, optimizer, criterion, epochs, scheduler)
        torch.save(model.state_dict(), f'./checkpoints/drone_model_cfc_{epochs}.pth')
        eval(model, val_loader, criterion)
    if args.test:
        model.load_state_dict(torch.load(f'./checkpoints/drone_model_cfc_{epochs}.pth'))
        eval(model, val_loader, criterion)