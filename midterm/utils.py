import numpy as np
import cv2

def mask_setTo():
    src = cv2.imread('./src/lenna.bmp', cv2.IMREAD_COLOR)
    mask = cv2.imread('./src/mask_smile.bmp', cv2.IMREAD_GRAYSCALE) # 마스킹되는 부분 : 흰색(255) 부분. 이외 검정색(0) 부분은 0이므로 마스킹이 일어나지 않음에 유의.

    src[mask > 0] = (0, 255, 255) # 빨 + 초 = 노

    cv2.imshow('src', src)
    cv2.imshow('mask', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mask_copyTo():
    src = cv2.imread('src/airplane.bmp', cv2.IMREAD_COLOR)
    mask = cv2.imread('src/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread('src/field.bmp', cv2.IMREAD_COLOR)

    dst[mask > 0] = src[mask > 0]
    # cv2.copyTo(src, mask, dst)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def time_inverse():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dst = np.empty(src.shape, dtype = src.dtype) # 최종 출력 NumPy 배열 초기화(생성)

    tm = cv2.TickMeter()
    
    tm.start()

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = 255 - src[y, x]

    tm.stop()
    print('Image inverse implementation took %4.3f ms.\n' % tm.getAvgTimeMilli())


def time_inverse_pixel_NumPy():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dst = np.empty(src.shape, dtype = src.dtype) # 최종 출력 NumPy 배열 초기화(생성)

    tm = cv2.TickMeter()
    
    tm.start()

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            dst[y, x] = ~src[y, x] # 'pixelwise' ~ 연산 적용 시 결과

    tm.stop()
    print('Image inverse implementation took %4.3f ms.\n' % tm.getAvgTimeMilli())


def time_inverse_NumPy():
    src = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)
    
    dst = np.empty(src.shape, dtype = src.dtype) # 최종 출력 NumPy 배열 초기화(생성)

    tm = cv2.TickMeter()

    tm.start()

    dst = ~src

    tm.stop()
    print('Image inverse implementation took %4.3f ms.\n' % tm.getAvgTimeMilli())


def useful_func():
    img = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    # print(type(img)) # np.ndarray

    sum_img = np.sum(img)
    mean_img = np.mean(img)
    print('Sum:', sum_img)
    print('Mean:', mean_img)

    minVal, maxVal, minPos, maxPos = cv2.minMaxLoc(img)
    print('minVal is', minVal, 'at', minPos)
    print('maxVal is', maxVal, 'at', maxPos)


if __name__ == "__main__":
    # mask_setTo()
    # mask_copyTo()
    # time_inverse()
    # time_inverse_pixel_NumPy()
    # time_inverse_NumPy()
    useful_func()