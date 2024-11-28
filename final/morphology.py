import numpy as np
import cv2
from matplotlib import pyplot as plt

def erode_dilate():
    src = cv2.imread('src/milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 먼저 이진화를 거쳐야 침식 / 팽창 연산이 가능해짐

    dst1 = cv2.erode(src_bin, None) # kernel = None : 기본값 구조 요소 행렬인 3 x 3 사이즈 사용
    dst2 = cv2.dilate(src_bin, None)
    
    # 범용적인 모폴로지 연산을 제공하는 morphologyEx 함수 이용해도 됨
    # dst1 = cv2.morphologyEx(src_bin, cv2.MORPH_ERODE, None) 
    # dst2 = cv2.morphologyEx(src_bin, cv2.MORPH_DILATE, None) 

    plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('erode')
    plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('dilate')
    plt.show()


def open_close():
    src = cv2.imread('src/milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst1 = cv2.morphologyEx(src_bin, cv2.MORPH_OPEN, None)
    dst2 = cv2.morphologyEx(src_bin, cv2.MORPH_CLOSE, None)
    dst3 = cv2.morphologyEx(src_bin, cv2.MORPH_GRADIENT, None)

    plt.subplot(231), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(232), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(233), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('open')
    plt.subplot(234), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('close')
    plt.subplot(235), plt.axis('off'), plt.imshow(dst3, 'gray'), plt.title('gradient') # 윤곽선 검출 시각화 추가
    plt.show()




if __name__ == "__main__":
    erode_dilate()
    open_close()