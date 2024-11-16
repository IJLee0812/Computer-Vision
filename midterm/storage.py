import numpy as np
import cv2

filename = 'mydata.json'
# filename = 'mydata.xml'
# filename = 'mydata.yml'

def writeData():
    name = 'LeeIJ'
    age = 24
    pt1 = (100, 200)
    scores = (100, 100, 95)
    mat1 = np.array([[1.0, 1.5], [2.0, 3.3]], dtype = np.float32)

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    fs.write('name', name)
    fs.write('age', age)
    fs.write('point', pt1)
    fs.write('scores', scores)
    fs.write('data', mat1)

    fs.release() # 중요! 반드시 실행해줘야 파일에 정상적으로 써짐!


# 읽기 모드로 파일을 열면, 파일 전체를 분석하여 '계층적 구조'를 갖는 노드 집합을 구성함.
def readData():
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    name = fs.getNode('name').string() # 특정 이름으로 저장되어 있는 Node에 접근하기 위한 getNode() 함수.
    age = int(fs.getNode('age').real())
    pt1 = tuple(fs.getNode('point').mat().astype(np.int32).flatten())
    scores = tuple(fs.getNode('scores').mat().flatten())
    mat1 = fs.getNode('data').mat()

    fs.release() # 마찬가지로 release 함수를 실행해줘야 파일을 정상적으로 읽음!

    print('name:', name)
    print('age:', age)
    print('point:', pt1)
    print('scores:', scores)
    print('data:')
    print(mat1)


if __name__ == '__main__':
    writeData()
    readData()