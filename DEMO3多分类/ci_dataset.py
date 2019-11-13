import torch.utils.data
from torchvision import transforms, models, datasets
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,testloader,classes

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if __name__ == '__main__':
    trainloader, _, classes = data()
    dataiter = iter(trainloader)
    images, labels = next(dataiter) # 取第一组图片的数据
    print(images[0].shape, labels[0].data.item()) # labels 是 1-10 的数字
    imshow(torchvision.utils.make_grid(images))

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print(len(trainloader), len(trainloader.dataset))