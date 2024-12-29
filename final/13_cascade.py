import cv2
import numpy as np

def detect_face():

    src = cv2.imread('src/kids.png')

    classifier = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')

    faces = classifier.detectMultiScale(src) # 이미지 스케일링 피라미드. 다양한 이미지 사이즈를 고려함. 

    for (x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detect_eyes():
    src = cv2.imread('src/kids.png')

    face_classifier = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(src)

    eye_classifier = cv2.CascadeClassifier('src/haarcascade_eye.xml')

    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(src, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2) # 얼굴 영역을 보라색 사각형으로 검출
        faceROI = src[y1:y1 + h1, x1:x1 + w1] # (y, x)임에 유의
        
        eyes = eye_classifier.detectMultiScale(faceROI) # "얼굴 ROI 안에서만" 눈을 검출해서 효율성을 올림

        for (x2, y2, w2, h2) in eyes:
            center = (int(x2 + w2 / 2), int(y2 + h2 / 2)) # 중점이므로, (w2 / 2 및 h2 / 2)로 구해야 함
            cv2.circle(faceROI, center, int(w2 / 2), (255, 0, 0), 2, cv2.LINE_AA) # 중점 center, 반지름 길이 int(w2 / 2)로 눈 위치를 파란색 원으로 표시 (눈의 가로가 더 기므로, 가로 폭을 2로 나눈 값을 반지름으로 설정하였음)

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face()
    detect_eyes()
    