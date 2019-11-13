import torch.utils.data
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tensorboardX import SummaryWriter

def ClassifyDataset(root):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    image_datasets = {
        x : datasets.ImageFolder(os.path.join(root, x), data_transforms[x])
        for x in ['train', 'val']
    }


    dataloaders = {
        x : torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                        shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {
        x : len(image_datasets[x]) for x in ['train', 'val']
    }
    classes_name = image_datasets['train'].classes

    return classes_name, dataset_sizes, dataloaders

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) # 归一化，将图片像素压缩在 0 1 之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    root = '../data/hymenoptera_data' # 不用再加一个 '/'
    classes_name, sizes, dataloaders = ClassifyDataset(root)
    print(classes_name, sizes)
    print(len(dataloaders['train']))
    print(len(dataloaders['val']))


    images, labels = next(iter(dataloaders['train']))
    print(images.size(), images.size(0),images.size()[0],labels.size(), sizes['train'], classes_name)

    out = torchvision.utils.make_grid(images[:4]) # 4 张图片

    imshow(out, title=[classes_name[labels[j]] for j in range(4)])


