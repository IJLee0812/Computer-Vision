import numpy as np
import cv2

def contrast1():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    s = 2.0

    dst = cv2.multiply(src, s) # 포화 연산 함께 수행됨.

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


import cv2

def contrast2():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    alpha = 1.0  # 기본 alpha 값

    dst = cv2.convertScaleAbs(src, alpha = alpha + 1, beta = -128 * alpha)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # contrast1()
    contrast2()