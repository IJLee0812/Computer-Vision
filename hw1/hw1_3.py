import cv2
import numpy as np
import sys

# - - - 블로그 활용, 출처 : https://ks-jun.tistory.com/m/198 - - -
def order_points(pts):
    rect = np.zeros((4, 2), dtype = np.float32)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) ; widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) ; heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
# - - - 블로그 활용, 출처 : https://ks-jun.tistory.com/m/198 - - -

if __name__ == "__main__":
    image_path = sys.argv[1]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    edges = cv2.Canny(img, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations = 1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 찾기 (체커보드의 외곽선으로 가정)
    largest_contour = max(contours, key = cv2.contourArea)

    # 윤곽선을 근사화하여 꼭짓점 찾기
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 꼭짓점이 4개인 경우 (사각형)
    if len(approx) == 4:
        # 투영 변환 적용
        warped = four_point_transform(img, approx.reshape(4, 2))

        # 변환된 이미지 크기 조정
        warped = cv2.resize(warped, (500, 500))

        # 원본 이미지 및 투영 변환 결과 이미지 표시
        cv2.imshow('Original Image', img)
        cv2.imshow("Warped Checkerboard", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("투영 변환 결과에서 체커보드가 감지되지 않습니다.")