import sys
import numpy as np
import cv2

def on_mouse(event, x, y, flags, param):
    global cnt, src_pts

    if event == cv2.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)
            cnt += 1

            cv2.circle(src, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('src', src)

        if cnt == 4:
            w = 500
            h = 500

            dst_pts = np.array([[0, 0], 
                               [w - 1, 0], 
                               [w - 1, h - 1], 
                               [0, h - 1]]).astype(np.float32)
        
            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            dst = cv2.warpPerspective(src, pers_mat, (w, h))

            cv2.imshow('dst', dst)


if __name__ == "__main__":
    # 변수 및 환경 초기화
    cnt = 0
    img_path = sys.argv[1]
    src = cv2.imread(img_path)
    src_pts = np.zeros([4, 2], dtype = np.float32)
    
    # 마우스 콜백 함수 설정
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse)

    # 원본 이미지 표시
    cv2.imshow('src', src)
    
    # 꼭짓점 4개 모두를 유저가 선택하면, 투시 변환 결과 창을 띄워준다. 아무 키나 누르면 종료된다.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

