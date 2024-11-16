import numpy as np
import cv2


def blurring_mean():
    src = cv2.imread('src/rose.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('src', src)

    for ksize in range(3, 8, 2):
        dst = cv2.blur(src, (ksize, ksize)) # tuple로 넣어줌

        desc = "Mean: %dx%d" % (ksize, ksize)
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()

    cv2.destroyAllWindows()


def blurring_gaussian():
    src = cv2.imread('src/rose.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('src', src)

    arr = cv2.getGaussianKernel(9, 0) # (ksize, sigma, ktype) -> 커널 크기와 표준 편차를 전달하면 가우시안 필터를 반환합니다.

    print(arr)

    for sigma in range(1, 6):
        dst = cv2.GaussianBlur(src, (0, 0), sigma)

        desc = "Gaussian: sigma = %d" % (sigma)
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, 255, 1, cv2.LINE_AA)
        
        cv2.imshow('dst', dst)
        cv2.waitKey()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    # blurring_mean()
    blurring_gaussian()