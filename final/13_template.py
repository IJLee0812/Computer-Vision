import sys
import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread('src/circuit.bmp', cv2.IMREAD_COLOR)
    templ = cv2.imread('src/crystal.bmp', cv2.IMREAD_COLOR)

    img = img + (50, 50, 50) # 밝기 값을 전체적으로 향상시켜, 원본 영상이 밝기에 강건한지 확인하기 위한 작업

    noise = np.zeros(img.shape, np.int32) 
    cv2.randn(noise, 0, 10) # 표준편차 10인 가우시안 노이즈 생성

    img = cv2.add(img, noise, dtype = cv2.CV_8UC3) # 원본 영상이 노이즈에 강건한지 확인하기 위한 작업

    res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED) # 결괏값 범위 : -1 ~ 1
    res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # 결괏값 행렬을 영상으로 표현하기 위해 MINMAX Scaling 적용하여 템플릿 매칭 결과를 시각적으로 확인할 수 있음 + 그레이스케일 영상으로 변환되었음(타입을 CV_8U로 변경 및 픽셀을 0 ~ 255로 변경)

    _, maxV, _, maxLoc = cv2.minMaxLoc(res) # 최댓값 및 최댓값 가지는 위치 산출을 위해 해당 함수 이용

    print('Maximum Value :', maxV) # 유사도 값(상관계수 값) 산출

    (th, tw) = templ.shape[:2]

    print(f"th : {th}, tw : {tw}")

    cv2.rectangle(img, maxLoc, (maxLoc[0] + tw, maxLoc[1] + th), (255, 0, 0), 2)

    cv2.imshow('templ', templ)
    cv2.imshow('res_norm', res_norm) # 크기 : (img.shape[0] - tw + 1, img.shape[1] - th + 1)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


