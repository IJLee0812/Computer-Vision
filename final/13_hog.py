# HOGDescriptor 클래스가 제공하는 보행자 검출 HOG 정보를 이용하여 동영상 매 프레임에서 보행자를 검출하고, 그 결과를 화면에 표시하는 예제

import sys
import numpy as np
import cv2
import random

cap = cv2.VideoCapture('src/vtest.avi')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # 보행자 / 보행자X 판별을 위한 SVM 모델 설정, HOG 기술자와 SVM 분류기를 연결하여 객체를 효과적으로 탐지할 수 있도록 구성

while True:
    ret, frame = cap.read()

    detected, _ = hog.detectMultiScale(frame)

    for (x, y, w, h) in detected:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # 무작위 색상 부여를 위함

        cv2.rectangle(frame, (x, y), (x + w, y + h), c, 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()

