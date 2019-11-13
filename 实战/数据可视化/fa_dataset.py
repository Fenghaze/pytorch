import torch
import torchvision
from torchvision import transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

def FaDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST('../../data', train=True,
                                                 transform=transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.FashionMNIST('../../data', train=False,
                                                transform=transform, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                              shuffle=True, num_workers=4)

    classes_name = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    return trainloader, testloader, classes_name

def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    trainloader,_,classes_name = FaDataset()
    print(len(trainloader), len(trainloader.dataset))
    for i, sample in enumerate(trainloader, 0):
        images, labels = sample
        print(images.shape)
        imshow(torchvision.utils.make_grid(images)) # 显示一个 batch 的所有图片
        print(' '.join('%5s' % classes_name[labels[j]] for j in range(4)))# 打印前 4 个图片的 labels
        if i == 0:
            break