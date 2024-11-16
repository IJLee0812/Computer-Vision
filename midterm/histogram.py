import numpy as np
import cv2

def calcGrayHist(img):
    channels = [0]
    histSize = [256]
    histRange = [0, 256]

    hist = cv2.calcHist([img], channels, None, histSize, histRange) # 3번째 파라미터 = 마스크 영상

    return hist # 256 x 1 크기의 CV_32FC1 타입 히스토그램 행렬.


# 히스토그램 그래프에서 최대 빈도 수를 표현하는 막대그래프 길이(y축)가 100픽셀이 되도록 그래프를 그림.
def getGrayHistImage(hist):
    histMax = np.max(hist)
    
    imgHist = np.full((100, 256), 255, dtype = np.uint8) # 흰색으로 초기화된 256 x 100 크기의 영상 imgHist 생성

    for x in range(256): # 원점이 좌상단임을 고려한 pt1, pt2 설정
        pt1 = (x, 100) # 바닥부터 시작
        
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax)) 
        
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist


# 히스토그램 스트레칭 함수(OpenCV에서 따로 제공해주지 않아, 손수 구현해야 함.)
def histogram_stretching():
    src = cv2.imread('src/hawkes.bmp', cv2.IMREAD_GRAYSCALE)

    gmin, gmax, _, _ = cv2.minMaxLoc(src)

    dst = cv2.convertScaleAbs(src, alpha = 255.0 / (gmax - gmin), beta = -gmin * 255.0 / (gmax - gmin))

    cv2.imshow('src', src)
    cv2.imshow('srcHist', getGrayHistImage(calcGrayHist(src)))

    cv2.imshow('dst', dst)
    cv2.imshow('dstHist', getGrayHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows()


# 히스토그램 평탄화 함수
def histogram_equalization():
    src = cv2.imread('src/hawkes.bmp', cv2.IMREAD_GRAYSCALE)

    dst = cv2.equalizeHist(src)

    cv2.imshow('src', src)
    cv2.imshow('srcHist', getGrayHistImage(calcGrayHist(src)))

    cv2.imshow('dst', dst)
    cv2.imshow('dstHist', getGrayHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # img = cv2.imread('src/lenna.bmp', cv2.IMREAD_GRAYSCALE)
    # hist = calcGrayHist(img)
    # imgHist = getGrayHistImage(hist)
    # cv2.imshow('src', img)
    # cv2.imshow('histogram', imgHist) # <=> cv2.imshow('histogram', getGrayHistImage(calcGrayHist(img)))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # histogram_stretching()

    histogram_equalization()



