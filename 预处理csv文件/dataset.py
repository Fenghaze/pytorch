import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import utils
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from transformClass import Rescale, RandomCrop, ToTensor

class FaceLandmarkDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: .csv 文件
        :param root_dir: 图片存放路径
        :param transform: transform 选项
        """
        self.landmarks_frame = pd.read_csv(csv_file) # 数据对象
        self.root_dir = root_dir # 图片存储路径
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        img = io.imread(img_name) # 获取图片对象
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks]) # 转换为矩阵
        landmarks = landmarks.astype('float').reshape(-1, 2)
        item = {'image': img, 'landmarks': landmarks}

        if self.transform:
            item = self.transform(item)

        return item

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)

def show_landmarks_batch(sample_batched):
    """
    展示一组 batch 图像
    :param sample_batched: 图像对象  batch_size * channels * height * width
    :return:
    """
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    print(batch_size, im_size) # batch_size = 4, im_size = 224
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i*im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10,
                    marker='.',
                    c='r')
        plt.title('Batch from dataloader')

if __name__ == '__main__':
    # 读取原始图像
    face_dataset = FaceLandmarkDataset(csv_file='../data/faces/face_landmarks.csv',
                                       root_dir='../data/faces')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        item = face_dataset[i]

        print(i, item['image'].shape, item['landmarks'].shape)

        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i+1)) # 设置标题
        ax.axis('off') # 关闭坐标轴
        show_landmarks(**item)

        if i == 3:
            plt.show()
            break

    # 读取 transformed 图像
    transformed_dataset = FaceLandmarkDataset(csv_file='../data/faces/face_landmarks.csv',
                                       root_dir='../data/faces',
                                       transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),
                                           ToTensor()
                                       ]))
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['landmarks'].size())

        if  i == 3:
            break

    # 读取 dataloader 图像，保证图像完整

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        # sample_batched 是一个图像对象，但是增加了一维 batch_size
        print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())


        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.show()
            break