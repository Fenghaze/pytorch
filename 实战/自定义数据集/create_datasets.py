import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, file_root, transform=None):
        fh = open(file_root, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, idx):
        image_path, label = self.imgs[idx]
        image = Image.open(image_path)  # 通过PIL.Image读取图片
        if self.transform:
            image = self.transform(image)
        return image,label

    def __len__(self):
        return len(self.imgs)

def data():

    transform = transforms.Compose([
        transforms.ToTensor(), # 归一化
        transforms.Normalize((0.1307,), (0.3081,)) # 标准化
    ])


    root = '../../data/cifar/'
    mydataset = {
        x: MyDataset(root + x + '.txt', transform=transform)
        for x in ['train', 'val']
    }

    datasize = {
        x: len(mydataset[x]) for x in ['train', 'val']
    }

    dataloader = {
        x: DataLoader(mydataset[x], batch_size=64, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    return datasize, dataloader


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 划分数据集
def split_dataset():
    pass

if __name__ == '__main__':
    datasize, dataloader = data()
    print(datasize['train'])
    dataiter = iter(dataloader['train'])
    images, labels = next(dataiter)
    print(images.size(), labels.size())
    for i, sample in enumerate(dataloader['train']):
        images, labels = sample
        print(images.size(), labels.size())
        imshow(torchvision.utils.make_grid(images[:4]))
        if i == 1:
            break