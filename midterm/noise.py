import numpy as np
import cv2
import random

def noise_gaussian():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('src', src)

    for stddev in [10, 20, 30, 40, 50]:
        
        
        
        noise = np.zeros(src.shape, np.int32) # 초기화 진행해줘야 함에 유의!

        cv2.randn(noise, 0, stddev)

        dst = cv2.add(src, noise, dtype = cv2.CV_8UC1) # OpenCV 덧셈 연산으로 원본 이미지에 노이즈 추가함.

        desc = 'stddev = %d' % stddev
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, 255, 1, cv2.LINE_AA)
        
        cv2.imshow('dst', dst)
        cv2.waitKey()
    
    cv2.destroyAllWindows()


def filter_bilateral():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('src', src)

    noise = np.zeros(src.shape, np.int32)

    cv2.randn(noise, 0, 5) # 표준편차가 5인 노이즈 추가하기 위해 노이즈 생성하여 noise 변수에 추가
    src = cv2.add(src, noise, src, dtype = cv2.CV_8UC1) # 원본 영상에 노이즈 추가

    dst1 = cv2.GaussianBlur(src, (0, 0), 5)
    dst2 = cv2.bilateralFilter(src, -1, 10, 5) # (src, d, sigmaColor, sigmaSpace) -> 노이즈 사라짐 + "에지 보존!"

    cv2.imshow('noised_src', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def filter_median():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    for i in range(0, int(src.size / 10)): # 전체 크기의 10 퍼센트에 랜덤으로 소금/후추 잡음 삽입.
        x = random.randint(0, src.shape[1] - 1) 
        y = random.randint(0, src.shape[0] - 1)
        src[x, y] = (i % 2) * 255 # 나머지 0 : 후추 / 나머지 1 : 소금 잡음 추가
    
    dst1 = cv2.GaussianBlur(src, (0, 0), 1)
    dst2 = cv2.medianBlur(src, 3)
    cv2.imshow('src', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    # noise_gaussian()
    # filter_bilateral()
    filter_median()