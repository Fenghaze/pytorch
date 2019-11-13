import numpy as np
import cv2


img = cv2.imread('./data/cat.jpg', 0)

if __name__ == '__main__':
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    cv2.imwrite('./data/cat2.jpg', img)
    cv2.destroyAllWindows()
