import numpy as np
import cv2
import math

def setLabel(img, pts, label):
    
    (x, y, w, h) = cv2.boundingRect(pts)

    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


def main():
    img = cv2.imread("src/polygon.bmp", cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print(type(contours)) # 튜플

    for pts in contours:
        if cv2.contourArea(pts) < 400: # 휴리스틱하게 설정
            continue
            
        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True) # True / False : 폐곡선 여부
        
        vtc = len(approx) # 꼭짓점 개수

        if vtc == 3:
            setLabel(img, pts, 'TRIANGLE')

        elif vtc == 4:
            setLabel(img, pts, 'RECTANGLE')
        
        else:
            length = cv2.arcLength(pts, True) # 원의 둘레의 길이
            area = cv2.contourArea(pts) # 원의 면적
            
            ratio = 4. * math.pi * area / (length * length)
            if ratio > 0.85: # 휴리스틱하게 설정(원으로 검출할 지 말지 여부)
                setLabel(img, pts, 'CIRCLE') 

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()