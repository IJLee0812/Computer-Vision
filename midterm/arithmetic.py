import numpy as np
import cv2
from matplotlib import pyplot as plt

src1 = cv2.imread('src/lenna256.bmp', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('src/circle.bmp', cv2.IMREAD_GRAYSCALE)

# src2를 src1의 크기로 리사이징
src2 = cv2.resize(src2, (src1.shape[1], src1.shape[0])) # shape[0] : 세로 길이 / shape[1] : 가로 길이

dst1 = cv2.add(src1, src2)
dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0) # np.float64이므로 0.0으로 표기해야 함
dst3 = cv2.subtract(src1, src2)
dst4 = cv2.absdiff(src1, src2)

plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('add')
plt.subplot(234), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('addWeighted')
plt.subplot(235), plt.axis('off'), plt.imshow(dst3, 'gray'), plt.title('subtract')
plt.subplot(236), plt.axis('off'), plt.imshow(dst4, 'gray'), plt.title('absdiff')
plt.show()