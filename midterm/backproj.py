import numpy as np
import cv2

# - - - 1. 기준 영상(ref_ycrcb)에서, 마스크 이미지(기 정의) 영역에 대해, CrCb 2차원 히스토그램(자주 사용되는 색상 정보가 담기게 됨)을 계산하여 hist에 저장함. - - -

ref = cv2.imread('src/ref.png', cv2.IMREAD_COLOR)

mask = cv2.imread('src/mask.bmp', cv2.IMREAD_GRAYSCALE) # 기 정의된 마스크 영상. 피부색 영역에 대한 히스토그램을 기존 영상에서 추출한 것. 

ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

channels = [1, 2] # YCrCb 공간의 1, 2 = 색상 채널 두 개를 사용함을 의미

# 0 ~ 255 범위 중 절반만 사용하겠다는 의미.
cr_bins = 128 
cb_bins = 128

histSize = [cr_bins, cb_bins]

cr_range = [0, 256]
cb_range = [0, 256]

ranges = cr_range + cb_range

hist = cv2.calcHist([ref_ycrcb], channels, mask, histSize, ranges) # 기준 영상에서 마스크 영역에서만(흰색 영역) CrCb 2차원 히스토그램을 계산함.

# - - - - - - - - - - - - - - - - 
# Apply histogram backprojection to an input image!

src = cv2.imread('src/kids.png', cv2.IMREAD_COLOR)
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1) # calcBackProject : 히스토그램 역투영. 입력 영상으로부터 찾고자 하는 객체의 "컬러 히스토그램"을 미리 구해 둔 hist 변수를 이용하여, 히스토그램 역투영을 수행함. 주어진 입력 영상인 src_ycrcb 영상에서 hist에 부합하는 영역을 찾아낸다.

cv2.imshow('ref', ref)
cv2.imshow('src', src)
cv2.imshow('mask', mask)
cv2.imshow('hist', hist)
cv2.imshow('backproj', backproj)
cv2.waitKey()
cv2.destroyAllWindows()



