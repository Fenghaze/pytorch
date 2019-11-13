from PIL import Image
import numpy as np
import skimage.io as io

img_object = Image.open('./data/cat.jpg')
img = np.array(img_object)
img2 = io.imread('./data/cat.jpg')

if __name__ == '__main__':
    print(img.shape)
    #img_object.show()
    io.imshow(img)