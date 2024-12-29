import numpy as np
import cv2


def corner_harris():
    src = cv2.imread('src/building.jpg', cv2.IMREAD_GRAYSCALE)

    harris = cv2.cornerHarris(src, 3, 3, 0.04) # blockSize, ksize, k -> 모든 픽셀의 R값을 저장함
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # R값을 그레이스케일 범위로 정규화함 -> GRAYSCALE 공간에서 코너가 밝은 회색으로 검출됨

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # 코너를 빨간색 점으로 표현하기 위해, 색공간 변환이 요구됨

    for y in range(harris_norm.shape[0]):
        for x in range(harris_norm.shape[1]):
            if harris_norm[y, x] > 120: # threshold 값을 휴리스틱하게 120보다 큰 경우 코너로 간주, 낮출 경우 더 많은 건물 모서리를 코너로 검출하나 나뭇잎 또는 풀밭에서도 코너로 검출되는 부분이 생김(노이즈)
                # 비최대 억제
                if ((harris[y, x] > harris[y - 1, x]) and
                    (harris[y, x] > harris[y + 1, x]) and
                    (harris[y, x] > harris[y, x - 1]) and
                    (harris[y, x] > harris[y, x + 1])): # 4-way connectivity일 때, 기준점 (x, y)를 기준으로 인접한 네 개의 픽셀에 대해 비최대 억제를 만족하는 (x, y)만 코너로 검출함
                    cv2.circle(dst, (x, y), 5, (0, 0, 255), 2)

    cv2.imshow('src', src)
    cv2.imshow('harris_norm', harris_norm)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def corner_fast():
    src = cv2.imread('src/building.jpg', cv2.IMREAD_GRAYSCALE)

    fast = cv2.FastFeatureDetector_create(60) # FAST 코너 검출 객체 생성, 밝기 차이가 60보다 큰 경우 코너로 간주. 즉 인자로 threshold 값을 대입함, 기본값으로 비최대 억제를 수행함(nonmaxSuppression = False로 비최대 억제를 비활성화할 수 있음)
    keypoints = fast.detect(src) # FAST 알고리즘을 사용하여 코너를 검출하고, 이를 keypoints로 리턴. 검출된 코너들의 리스트(위치 관련 정보 포함)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) # 코너를 빨간색 점으로 표현하기 위해, 색공간 변환이 요구됨 

    for kp in keypoints:
        pt = (int(kp.pt[0])), (int(kp.pt[1])) # y, x좌표 추출
        cv2.circle(dst, pt, 5, (0, 0, 255), 2)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # corner_harris()
    corner_fast()
