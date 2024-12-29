import sys
import cv2
import numpy as np


def keyPoints():
    src = cv2.imread('src/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create() # ORB 객체 생성 먼저 수행해야 함

    keypoints = orb.detect(src)
    keypoints, desc = orb.compute(src, keypoints) # 리턴값은 특징점(들), 특징점 기술자 행렬
    # keypoints, desc = orb.detectAndCompute(src, None)과 동치

    print('len(keypoints) : ', len(keypoints))    
    print('desc.shape : ', desc.shape) # 기술자 행렬은 500 x 32로 구성됨, CV_8UC1(float)이기 때문에 각 기술자의 크기는 32바이트

    # (-1, -1, -1) : BGR공간에서 랜덤하게 색상 뽑아줌
    dst = cv2.drawKeypoints(src, keypoints, None, (-1, -1, -1), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS) # 맨 마지막 열거형 상수 : Keypoint의 위치, 크기 및 방향 정보를 함께 표시해주는 옵션

    cv2.imshow('src', src)
    cv2.imshow('dst', dst) # 각 특징점 위치를 중심으로 하는 다수의 원이 그려짐. 원 크기는 특징점 검출 시 고려한 "이웃 영역의 크기", 원의 중심에서 뻗어 나간 직선은 특징점 근방에서 추출된 "주된 방향"

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    keyPoints()