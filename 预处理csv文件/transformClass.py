from skimage import transform
import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt




class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.

        e.g.
        r1 = Rescale(3)  # 输出 3x3
        r2 = Rescale((3, 4)) # 输出 3x4
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2] # 获取前两列的值，分别为长和宽
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h] # landmarks 也要等比例缩放

        return {'image': img, 'landmarks': landmarks}



class RandomCrop():
    """
    随机裁剪图像
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        image, landmarks = item['image'], item['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [left, top]

        return {'image':image, 'landmarks':landmarks}


class ToTensor():
    """
    将矩阵转换为 torch 的张量
    """
    def __call__(self, item):
        image, landmarks = item['image'], item['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}

if __name__ == '__main__':
    from dataset import FaceLandmarkDataset, show_landmarks
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
    face_dataset = FaceLandmarkDataset(csv_file='../data/faces/face_landmarks.csv',
                                       root_dir='../data/faces')
    fig = plt.figure()
    sample = face_dataset[65]

    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i+1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)
    plt.show()