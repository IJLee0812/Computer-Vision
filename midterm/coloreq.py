import sys
import numpy as np
import cv2

src = cv2.imread('src/pepper.bmp', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed!')
    sys.exit()

src_YCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

ycrcb_planes = list(cv2.split(src_YCrCb))  # 튜플을 리스트로 변환해줘야 함!!!

ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])  # Y 채널에 대해 히스토그램 평활화 적용

dst_ycrcb = cv2.merge(ycrcb_planes)

dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)  # 주의: ycrcb_planes가 아니라 dst_ycrcb를 사용해야 함

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()