import torch
from torchvision import transforms,datasets
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def data(root):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root, download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64, num_workers=4)

    testset = datasets.MNIST(root, download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=64, num_workers=4)

    classes = (0,1,2,3,4,5,6,7,8,9)

    return trainloader, testloader, classes

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



if __name__ == '__main__':
    root = '../data'
    trainloader, _, classes = data(root)
    print(len(trainloader),len(trainloader.dataset))
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    for i, sample in enumerate(trainloader, 0):
        images, labels = sample
        print(images.shape)
        imshow(torchvision.utils.make_grid(images)) # 显示一个 batch 的所有图片
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))# 打印前 4 个图片的 labels
        if i == 0:
            break