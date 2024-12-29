import numpy as np
import cv2

def keypoint_matching_knn_added():
    # 입력 이미지 로드
    src1 = cv2.imread('src/box.png', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('src/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    # ORB 기술자 생성
    orb = cv2.ORB_create()

    # 키포인트와 기술자 추출
    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)

    # 매처 생성 (해밍 거리 기준) 및 단일 매칭 수행
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)

    # KNN 매칭 수행 (각 기술자에 대해 가장 가까운 5개의 매칭), 좋은 매칭만 필터링 (Ratio Test)
    knn_matches = matcher.knnMatch(desc1, desc2, k = 2)
    ratio_thresh = 0.75
    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]

    # 매칭 결과 시각화
    dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # 맨 마지막 열거형 상수 의미 : 매칭되지 않은 특징점은 그리지 않음
    knn_dst = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, [good_matches], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 결과 출력
    cv2.imshow('dst', dst)  # 단일 매칭 결과
    cv2.imshow('knn_dst', knn_dst)  # KNN 매칭 결과
    cv2.waitKey()
    cv2.destroyAllWindows()


def good_matching():
    src1 = cv2.imread('src/box.png', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('src/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)
    
    # - - - 검출 완료 - - -

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)
    # print(matches[0].distance)
    
    matches = sorted(matches, key = lambda x : x.distance) # 거리 기준 오름차순 정렬(거리 값이 작은 것이 더 좋으므로)
    
    good_matches = matches[:50] # 오름차순 정렬된 것 상위 50개 추출

    # - - - 추출 완료 - - -

    dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, good_matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def find_homography():
    # 입력 이미지 로드 (그레이스케일로 변환하여 읽음)
    src1 = cv2.imread('src/box.png', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread('src/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    # ORB 특징점 검출기 및 기술자 생성
    orb = cv2.ORB_create()

    # 키포인트와 기술자 추출
    keypoints1, desc1 = orb.detectAndCompute(src1, None)
    keypoints2, desc2 = orb.detectAndCompute(src2, None)

    # BFMatcher 생성 - 해밍 거리 사용
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # 단일 매칭 수행
    matches = matcher.match(desc1, desc2)

    # 매칭 결과를 거리 기준으로 오름차순 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 거리 기준 상위 50개의 매칭 추출
    good_matches = matches[:50]

    # 좋은 매칭 결과를 시각화
    dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, good_matches, None, 
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 좋은 매칭 결과의 특징점 좌표를 추출
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

    # RANSAC 알고리즘을 사용해 호모그래피 행렬(H) 계산
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # 첫 번째 이미지(src1)의 네 모서리 좌표를 생성
    (h, w) = src1.shape[:2]
    corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)

    # 호모그래피 행렬을 이용해 두 번째 이미지(src2)에서의 모서리 좌표를 계산
    corners2 = cv2.perspectiveTransform(corners1, H)

    # 좌표를 두 번째 이미지의 위치로 이동 (시각화를 위해)
    corners2 = corners2 + np.float32([w, 0])

    # 두 번째 이미지에서 계산된 모서리 위치를 녹색 선으로 표시
    cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # keypoint_matching_knn_added()
    # good_matching()
    find_homography()
