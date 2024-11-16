import cv2 as cv
img = cv.imread('src/lenna.bmp', cv.IMREAD_GRAYSCALE)

img = ~img[:, img.shape[1] // 2: ]

cv.imshow('result', img)
cv.waitKey()
cv.destroyAllWindows()