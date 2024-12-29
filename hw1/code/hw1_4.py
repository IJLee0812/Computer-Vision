import sys
import cv2
import numpy as np

def count_checker_pieces(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 원 검출을 위한 Hough 변환, 파라미터 참조 : https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv 
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp = 1, minDist = 20,
                               param1 = 50, param2 = 30, minRadius = 10, maxRadius = 50)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        white_count = 0
        black_count = 0
        
        for i in circles[0, :]:
            # 원의 중심점 좌표
            x, y = i[0], i[1]
            
            # 원의 중심점 픽셀 밝기 값
            brightness = gray[y, x]
            
            # 밝기에 따라 흰색 또는 검은색 말로 분류
            if brightness > 128:  # threshold를 128로 설정(임의의 이미지에 대해 범용으로 처리)
                white_count += 1
            else:
                black_count += 1
    
        return white_count, black_count
    else:
        return 0, 0

if __name__ == "__main__":
    image_path = sys.argv[1]
    # 말 개수 세기
    white, black = count_checker_pieces(image_path)
    # 결과 출력
    print(f'w:{white} b:{black}')