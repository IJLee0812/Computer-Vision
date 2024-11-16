# 그레이스케일 레벨을 16단계로 보여주는 예제. 트랙바 위치에 16을 곱한 결과를 전체 픽셀 값으로 설정함.

import numpy as np
import cv2

def saturated(value): # 포화 연산 정의
    if value > 255:
        value = 255

    elif value < 0:
        value = 0

    return value

def on_lev_change(pos):
    img[:] = saturated(pos * 16) # 트랙바 위치에 16을 곱함
    cv2.imshow('image', img)

img = np.zeros((400, 400), np.uint8)

cv2.namedWindow('image')
cv2.createTrackbar('level', 'image', 0, 16, on_lev_change)

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()
