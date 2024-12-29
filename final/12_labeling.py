import numpy as np
import cv2

def labeling_basic():
    src = np.array([[0, 0, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 1, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.uint8)
    
    # 브로드캐스팅으로 객체 영역의 원소값을 255로 재설정
    src = src * 255

    cnt, labels = cv2.connectedComponents(src) # 리턴값 : 객체의 개수(배경 0 포함), 레이블 맵

    print('src:'), 
    print(src)
    print('labels:')
    print(labels)
    print('number of labels:', cnt)


def labeling_stats():
    src = cv2.imread('src/keyboard.bmp', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i] # 레이블링된 각 객체를 행 단위로 불러옴(stats 행렬에 통계 정보가 저장되어 있음)

        if area < 20: # 휴리스틱하게 설정
            continue

        pt1 = (x, y) # 왼쪽 위(바운딩 박스 기준점)
        pt2 = (x + w, y + h) # 오른쪽 아래

        cv2.rectangle(dst, pt1, pt2, (0, 255, 255)) # 각 객체의 바운딩 박스를 노란색으로 표시함

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # labeling_basic()
    labeling_stats()