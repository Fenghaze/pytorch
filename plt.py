import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
img = mpimg.imread('./data/cat.jpg')
lum_img = img[:, :, 0]


if  __name__ == '__main__' :
    print(img)
    print(img.shape)
    plt.imshow(lum_img)
    pylab.show()
    plt.imshow(lum_img, cmap='hot')
    pylab.show()