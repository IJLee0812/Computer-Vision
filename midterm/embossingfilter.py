import numpy as np
import cv2

src = cv2.imread('src/rose.bmp', cv2.IMREAD_GRAYSCALE)

emboss = np.array([[-1, -1, 0],
                   [-1, 0, 1], 
                   [0, 1, 1]], dtype = np.float32)

dst = cv2.filter2D(src, -1, emboss, delta = 128) # -1 : ddepth. 출력 이미지의 깊이 지정하는 파라미터. 입력 이미지와 동일한 깊이(비트 수)를 유지하라는 의미. / emboss : 적용할 엠보싱 필터 / delta : 음수 값이 모두 포화연산에 의해 0이 되면 입체감이 크게 감소하므로, 결과 영상에 128을 더하는 방법을 사용함.

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

