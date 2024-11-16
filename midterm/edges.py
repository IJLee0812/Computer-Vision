import numpy as np
import cv2

def sobel_derivative():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]).astype(np.float32)

    my = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]]).astype(np.float32)
    
    dx = cv2.filter2D(src, -1, mx, delta = 128) # 결과치에 128을 더하여 saturation 방지 및 회색 배경을 디폴트로 설정 후 에지 검출 확인(검정색 or 흰색 부분이 에지 부분임)
    dy = cv2.filter2D(src, -1, my, delta = 128)

    cv2.imshow('src', src)
    cv2.imshow('dx', dx)
    cv2.imshow('dy', dy)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sobel_edge():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dx = cv2.Sobel(src, cv2.CV_32F, 1, 0) # x방향으로 1차 미분
    dy = cv2.Sobel(src, cv2.CV_32F, 0, 1) # y방향으로 1차 미분

    fmag = cv2.magnitude(dx, dy)

    mag = np.uint8(np.clip(fmag, 0, 255)) # 그래디언트 크기를 그레이스케일 영상 형식으로 나타냄, 상 / 하한 설정 및 수치 조정됨(saturate).

    _, edge = cv2.threshold(mag, 150, 255, cv2.THRESH_BINARY) # > 150(THRESHOLD)인 픽셀들은 255로(흰색), 그렇지 않은 픽셀은 0으로(검정색) 표현된 이진 영상. 임계값을 너무 낮추면 잡음(노이즈)의 영향도 에지로 검출될 수 있음.

    cv2.imshow('src', src)
    cv2.imshow('mag', mag)
    cv2.imshow('edge', edge)
    cv2.waitKey()
    cv2.destroyAllWindows()


def canny_edge():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dst1 = cv2.Canny(src, 50, 100) # 비교적 관대하게 검사를 진행함. 범위가 좁을수록 관대하다는 의미.
    dst2 = cv2.Canny(src, 50, 150) # 비교적 엄격하게 검사를 진행함.


    cv2.imshow('src', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # sobel_derivative()
    # sobel_edge()
    canny_edge()