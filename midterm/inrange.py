import numpy as np
import cv2


def on_hue_changed(_ = None):
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'mask')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'mask')

    lowerb = (lower_hue, 100, 0) # 색상의 하한 값을 트랙바로 조절할 수 있게 함.
    upperb = (upper_hue, 255, 255) # 색상의 상한 값을 트랙바로 조절할 수 있게 함.
    mask = cv2.inRange(src_hsv, lowerb, upperb)

    cv2.imshow('mask', mask)


def main():
    global src_hsv

    src = cv2.imread('src/candies.png', cv2.IMREAD_COLOR)

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    cv2.imshow('src', src)

    cv2.namedWindow('mask')
    cv2.createTrackbar('Lower Hue', 'mask', 40, 179, on_hue_changed)
    cv2.createTrackbar('Upper Hue', 'mask', 80, 179, on_hue_changed)

    on_hue_changed(0) # 시작

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

