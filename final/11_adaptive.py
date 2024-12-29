# 사용자가 설정한 트랙바 위치 값을 블록의 크기로 사용하는 적응형 이진화를 수행하는 소스코드
import sys
import numpy as np
import cv2


def on_trackbar(pos):
    bsize = pos
    if bsize % 2 == 0: # 짝수이면 홀수로 변경해주기 위함
        bsize = bsize - 1  
    if bsize < 3: # 3보다 작으면 3으로 설정
        bsize = 3

    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, bsize, 5) # bsize : 3보다 같거나 큰 홀수, C = 5(임계값 조정 상수)

    cv2.imshow('dst', dst)


src = cv2.imread('src/sudoku.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)
cv2.namedWindow('dst')
cv2.createTrackbar('Block Size', 'dst', 0, 200, on_trackbar)
cv2.setTrackbarPos('Block Size', 'dst', 11) # bsize의 초깃값을 11로 설정함

cv2.waitKey()
cv2.destroyAllWindows()
