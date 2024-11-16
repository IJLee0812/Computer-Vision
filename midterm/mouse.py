import sys
import numpy as np
import cv2

def on_mouse(event, x, y, flags, param):
    global oldx, oldy # 기존 이벤트 발생 좌표 보존을 위함

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y
        print('EVENT_LBUTTONDOWN : %d, %d' % (x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        print('EVENT_LBUTTONUP : %d, %d' % (x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (0, 255, 255), 2) # (x, y) : 업데이트된 현재 커서 위치까지 선을 그려줌

            cv2.imshow('img', img)

            oldx, oldy = x, y

img = cv2.imread('src/lenna.bmp')

cv2.namedWindow('img')
cv2.setMouseCallback('img', on_mouse) # 마우스 콜백함수 등록하여, 이벤트를 처리하는 소스코드 추가. 기 정의한 on_mouse 함수가 실행됨.

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

    