import numpy as np
import cv2

def brightness1():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dst = cv2.add(src, 100)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def brightness2(): # 모든 픽셀을 방문하면서 픽셀 값에 일정한 상수를 더하거나 빼도 밝기 조절이 적용됨. 권장되는 방법은 아님(연산속도 차이 때문), 하지만 사용할 때가 있다.
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dst = np.empty(src.shape, src.dtype) # 결과 영상

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = src[y, x] + 100
            
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def saturated(value):
    if value < 0:
        value = 0
    elif value > 255:
        value = 255
    return value


def brightness3():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)
    
    dst = np.empty(src.shape, src.dtype) # 결과 영상

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = saturated(src[y, x] + 100)
            
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
def brightness4():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)
    
    def update(pos): # inner 콜백함수 정의.
        dst = cv2.add(src, pos)
        cv2.imshow('dst', dst)

    cv2.namedWindow('dst') # namedWindow() 함수로 먼저 영상 이름을 명명해야 createTrackbar 함수가 해당 이름으로 영상을 인식할 수 있음.
    cv2.createTrackbar('Brightness', 'dst', 0, 100, update)
    
    update(0) # 최초 실행 시, 트랙바로 조절되지 않은 원본 영상( + 0 ) 먼저 띄움.

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # brightness1()
    # brightness2()
    # brightness3()
    brightness4()