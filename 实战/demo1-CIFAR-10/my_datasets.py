# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
import torchvision

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip() # 删掉一行字符串末尾的所有空格
            words = line.split() # 以空格符分割字符串
            imgs.append((words[0], int(words[1]))) # 前半部分是图片路径，后半部是标签

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    mydataset = MyDataset(os.path.join("../../data/cifar-10/","train.txt"), transform=transform)
    print(len(mydataset))
    dataloader = DataLoader(mydataset,batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        imgs, labels = data
        print(imgs.shape)
        if i == 1:
            break