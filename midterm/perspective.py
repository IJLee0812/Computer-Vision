import sys
import numpy as np
import cv2

def on_mouse(event, x, y, flags, param):

    global cnt, src_pts # 클릭한 좌표 4개를 저장할 배열, 0으로 초기화(전역변수)

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 클릭 이벤트 발생 시

        if cnt < 4: # 투시 변환을 적용할 카드 선택하기 위한 로직
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)

            cnt += 1

            cv2.circle(src, (x, y), 5, (0, 0, 255), -1) # 점 찍은 위치 빨간색 원으로 표시
            cv2.imshow('src', src)

        if cnt == 4: # 꼭짓점 4개 모두 찍었을 경우
            w = 200
            h = 300

            dst_pts = np.array([[0, 0],
                                [w - 1, 0],
                                [w - 1, h - 1],
                                [0, h - 1]]).astype(np.float32)
            
            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts) # 투시 변환 행렬 (2 x 3) 리턴

            dst = cv2.warpPerspective(src, pers_mat, (w, h)) # 투시 변환 결과 영상 생성

            cv2.imshow('dst', dst)

if __name__ == "__main__":
    
    cnt = 0
    src_pts = np.zeros([4, 2], dtype = np.float32)
    src = cv2.imread('src/card.bmp')

    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse) # 콜백함수 등록하기

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()




        
