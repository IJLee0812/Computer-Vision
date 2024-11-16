import numpy as np
import cv2

src = cv2.imread('src/rose.bmp', cv2.IMREAD_GRAYSCALE)

cv2.imshow('src', src)

for sigma in range(1, 6):
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)

    alpha = 1.0 # alpha를 1로 지정 시, 날카로운 성분을 그대로 한 번 더하는 것. 1보다 작게 설정 시 조금 덜 날카로운 영상을 만들 수 있음.

    dst = cv2.addWeighted(src, 1 + alpha, blurred, -alpha, 0.0) # src * (1 + alpha) - dst * alpha + 0.0(gamma)

    desc = "sigma: %d" % sigma
    cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, 255, 1, cv2.LINE_AA)

    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
