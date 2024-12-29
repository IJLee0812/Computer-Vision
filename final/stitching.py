import sys
import numpy as np
import cv2

def stitching_image():
    argc = len(sys.argv)
    imgs = []

    for i in range(1, argc):
        img = cv2.imread(sys.argv[1])
        imgs.append(img)

    stitcher = cv2.Stitcher_create()
    status, dst = stitcher.stitch(imgs) # 리스트 내 이미지들 자동으로 스티칭 수행

    if status != cv2.Stitcher_OK:
        print('Error on Stitching Process. Program Terminates.')
        sys.exit()

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stitching_image()