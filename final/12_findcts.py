import numpy as np
import cv2
import random

def contours_basic():
    src = cv2.imread('src/contours.bmp', cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # 중요 -> RETR_LIST : 모든 컨투어 찾기(계층 구조 만들지 않음), CHAIN_APPROX_NONE : 모든 외곽선 점들의 좌표를 저장

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for i in range(len(contours)): 
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        cv2.drawContours(dst, contours, i, c, thickness = 3) # 3은 컨투어 라인의 두께를 의미함.

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


def contours_hierarchical():
    """
        컨투어들을 계층 구조에 따라 순회하며, 각 컨투어를 랜덤한 색상으로 채워서 이미지에 표시
    """
    src = cv2.imread('src/contours.bmp', cv2.IMREAD_GRAYSCALE)

    contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # 중요 -> RETR_CCOMP : 컨투어를 두 계층으로 구성함, CHAIN_APPROX_SIMPLE : 직선을 구성하는 점들만 저장하여 컨투어 정보를 간략화함

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    idx = 0
    while idx >= 0:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        cv2.drawContours(dst, contours, idx, c, -1, cv2.LINE_8, hierarchy) # 중요 -> -1 : 컨투어 라인 두께 = -1 이면 컨투어 내부를 채움, cv2.LINE_8 : 8-connectivity, hierarchy : 컨투어 계층 정보
        
        idx = hierarchy[0, idx, 0] # 다음 컨투어의 인덱스를 가져와 idx를 업데이트

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # contours_basic()
    contours_hierarchical()



