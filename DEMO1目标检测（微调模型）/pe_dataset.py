import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PennFudanDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # 生成列表，方便使用下标进行索引
        '''
        imgs = ['FudanPed00001.png', 'FudanPed00002.png', 'FudanPed00003.png', ...]
        '''
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx]) # 图的路径
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx]) # 图对应的 mask 路径
        img = Image.open(img_path).convert('RGB') # 将图片转化为 RGB 模式
        mask = Image.open(mask_path)
        mask = np.array(mask) # 将 masks 图片转化为向量形式， mask 图片是一张黑白图（二维）

        obj_ids = np.unique(mask) # 去掉列表中重复的数字
        obj_ids = obj_ids[1:] # 0 是背景，去掉

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, xmax, ymin, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1] * boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


if __name__ == '__main__':
    root = '../data/PennFudanPed/'
    datasets = PennFudanDataset(root)
    for i in range(len(datasets)):
        img,target = datasets[i]
        print(i, img.size,target)
        if  i == 2:
            break