import numpy as np
import cv2

def color_op():
    src = cv2.imread('src/butterfly.jpg', cv2.IMREAD_COLOR)

    print('The pixel value [B, G, R] at (0, 0) is', src[0, 0])


def color_inverse():
    src = cv2.imread('src/butterfly.jpg', cv2.IMREAD_COLOR)

    dst = np.zeros(src.shape, src.dtype) # shape, dtype -> 결과 이미지를 저장할 공간을 우선 영행렬로 선언하고, shape 및 dtype을 설정하여 src 행렬과 동일하게 만든다.

    print(src.shape[0], src.shape[1])

    tm = cv2.TickMeter()
    
    tm.start()
    # elementwise for-loop inverse
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            p1 = src[i, j]
            p2 = dst[i, j]
            p2[0] = 255 - p1[0]
            p2[1] = 255 - p1[1]
            p2[2] = 255 - p1[2]
    
    # dst = ~src
    tm.stop()
   
    
    print(f"{tm.getAvgTimeMilli()} ms.\n")

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
def color_grayscale(): # 중요 : cv2.IMREAD_GRAYSCALE로 불러오는 것과 완전히 동치임!!    
    src = cv2.imread('butterfly.jpg', cv2.IMREAD_COLOR)

    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # 그레이스케일 영상으로 변환되더라도, 원본 color 정보는 유지가 가능하다!

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def color_split():
    src = cv2.imread('src/candies.png', cv2.IMREAD_COLOR)

    bgr_planes = cv2.split(src)

    cv2.imshow('src', src)
    cv2.imshow('B_plane', bgr_planes[0])
    cv2.imshow('G_plane', bgr_planes[1])
    cv2.imshow('R_plane', bgr_planes[2])
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # color_op()
    # color_inverse()
    # color_grayscale()
    color_split()