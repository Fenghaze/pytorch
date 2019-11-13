"""
txt 文件一般格式：

img_path label
"""

from torchvision import transforms, datasets
import torchvision.datasets.mnist as mnist
from skimage import io
import os
import numpy as np

def download_dataset():
    pass


def imagefolder():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.224])
    ])

    trainset = datasets.ImageFolder('../../data/hymenoptera_data/train', transform=transform)

    print(len(trainset), trainset[1][0].size(), trainset[1][1], trainset.classes)

# 官方 FashionMNIST 数据集分为 train， test 文件夹，以及生成 txt 文件
def create_fashionmnist_lst(train=True):
    root = '../../data/FashionMNIST/raw'

    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
    )

    if train:
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])): # zip(a, b) 将 a b 对应位置打包成元组列表
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy()) # 以 numpy 形式保存
            a = label
            a = label.item()
            f.write(img_path + str(a) + '\n')

        f.close()

    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            a = label
            a = label.item()
            f.write(img_path + ' ' + str(a) + '\n')

        f.close()


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)

# cifar-10-batch 数据集分为 train， test 文件夹，生成 txt 文件
def create_cifar_lst(train=False):
    path = '../../data/cifar/'
    if train:
        data_path = path + 'train/'  # 存放 train 数据集的路径
        my_mkdir(data_path)
        f = open(path + 'train.txt', 'w')
        for i in range(1, 6):
            root = '../../data/cifar-10-batches-py/data_batch_' + str(i)
            train_data = unpickle(root) # 'data', 'labels', 'label_names'
            print(train_data[b'data'].shape)
            print(root + " is loading...")
            for j in range(10000): # cifar 有 10000 个训练样本
                img = np.reshape(train_data[b'data'][j], (3,32,32))
                img = img.transpose(1,2,0) # 图片转换为 32*32*3
                label = train_data[b'labels'][j]

                img_path = data_path +  'data_batch_' + str(i) + '_' +  str(j) + '.jpg'
                io.imsave(img_path, img) # 保存图片  ../../data/cifar/train/data_batch_i_j.jpg

                f.write(img_path + ' ' + str(label) + '\n')
        f.close()

    else:
        data_path = path + 'val/'  # 存放 val 数据集的路径
        my_mkdir(data_path)
        f = open(path + 'val.txt', 'w')
        root = '../../data/cifar-10-batches-py/test_batch'
        train_data = unpickle(root)  # 'data', 'labels', 'label_names'
        print(train_data[b'data'].shape)
        print(root + " is loading...")
        for j in range(10000):  # cifar 有 10000 个训练样本
            img = np.reshape(train_data[b'data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)  # 图片转换为 32*32*3

            img_path = data_path + str(j) + '.jpg'
            io.imsave(img_path, img)  # 保存图片  ../../data/cifar/val/x.jpg

            label = train_data[b'labels'][j]

            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

# cifar-10-batch 数据集分为 train， val 文件夹，生成分类文件夹，生成 txt 文件
def create_category(train=False):
    path = '../../data/cifar-cateogry/'
    my_mkdir(path)
    if train:
        f = open(path + 'train.txt', 'w')
        for i in range(1, 6):
            root = '../../data/cifar-10-batches-py/data_batch_' + str(i)
            train_data = unpickle(root)

            data_path = path + 'train/'
            my_mkdir(data_path)

            print(root + 'is loading...')
            for j in range(10000):
                img = np.reshape(train_data[b'data'][j], (3,32,32))
                img = img.transpose(1,2,0)

                label = train_data[b'labels'][j]

                category_path = data_path + str(label) + '/'
                io.imsave(category_path, img)
                img_path = category_path + str(j) + '.jpg'

                f.write(img_path + ' ' + str(label) + '\n')
            f.close()
    else:
        f = open(path + 'val.txt', 'w')
        root = '../../data/cifar-10-batches-py/test_batch'
        test_data = unpickle(root) # 返回字典形式的数据信息，包含 b'data', b'labels'

        data_path = path + 'val/'
        my_mkdir(data_path)

        print(root + 'is loading...')
        for i in range(10000):
            img = np.reshape(test_data[b'data'][i], (3,32,32))
            img = img.transpose(1,2,0)

            label = test_data[b'labels'][i]

            category_path = data_path + str(label) + '/'
            my_mkdir(category_path)

            img_path = category_path + str(i) + '.jpg'
            io.imsave(img_path, img)

            f.write(img_path + ' ' +  str(label) + '\n')
        f.close()


create_cifar_lst(train=True)
create_cifar_lst(train=False)