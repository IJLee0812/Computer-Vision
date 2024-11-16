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


def detect_squares(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations = 2) # 윤곽선 구분을 용이하게 하기 위해 팽창 연산 적용
    dilated = ~dilated # 체커보드 각 칸의 윤곽선을 하얀색에서 검정색으로 바꾸기 위해 색 반전 수행
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dilated_with_contours = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dilated_with_contours, contours, -1, (255, 0, 0), 2)
    cv2.imshow("Binary with All Contours", dilated_with_contours)

    # 사각형으로 판별된 contour의 면적을 저장하기 위한 리스트
    areas = []
    squares = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 사각형 모양을 만족하는 윤곽선만 일차적으로 선택
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            areas.append(area)

    # 사각형으로 판정된 윤곽선 면적의 평균과 표준편차 계산하여 필터링 수행
    if len(areas) > 0:
        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # 평균 면적에 근접한 사각형만 선택
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h

                # 평균치와 차이가 적은 윤곽선만 추가
                if abs(area - mean_area) < std_area:
                    squares.append(approx)

    return squares

def determine_board_size(num_squares):
    if 10 <= num_squares <= 64:
        return "8 x 8"
    elif 65 <= num_squares <= 100:
        return "10 x 10"
    else:
        return "현 알고리즘으로는 사이즈 측정 불가능한 이미지"

def main():
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations = 1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 찾기 (체커보드의 외곽선으로 가정)
    largest_contour = max(contours, key = cv2.contourArea)

    # 윤곽선을 근사화하여 꼭짓점 찾기
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) != 4:
        print("체커보드의 4개 꼭짓점을 찾을 수 없습니다.")
        return

    warped = four_point_transform(image, approx.reshape(4, 2))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    squares = detect_squares(gray)  # detect_squares 함수 호출하여 squares 검출
    
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, squares, -1, (0, 0, 255), 2) # 체커보드의 각 칸으로 검출된 윤곽선을 표시함
    
    board_size = determine_board_size(len(squares))
    print(f"검출된 사각형 개수: {len(squares)}")
    print(f"추정되는 체커보드 크기: {board_size}")
    
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Squares", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()