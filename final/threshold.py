import sys
import numpy as np
import cv2

def on_threshold(pos):
    _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('dst', dst)

if __name__ == "__main__":
    src = cv2.imread('src/neutrophils.png', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('src', src)

    cv2.namedWindow('dst')

    cv2.createTrackbar('Threshold', 'dst', 0, 255, on_threshold) # 콜백함수 등록
    cv2.setTrackbarPos('Threshold', 'dst', 128) # 임계점 초깃값을 128로 설정

    cv2.waitKey()
    cv2.destroyAllWindows()
