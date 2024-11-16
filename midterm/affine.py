import numpy as np
import cv2

def affine_transform():

    src = cv2.imread('src/tekapo.bmp')

    rows = src.shape[0] # 행 : 세로 길이
    cols = src.shape[1] # 열 : 가로 길이

    print(f'rows(세로길이) : {rows}, cols(가로길이) : {cols}')

    src_pts = np.array([[0, 0], # 좌상단
                       [cols - 1, 0], # 우상단
                       [cols - 1, rows - 1]] # 우하단
                       ).astype(np.float32)

    # 어파인 변환 적용될 세 점의 좌표
    dst_pts = np.array([[50, 50], # 좌상단
                       [cols - 100, 100], # 우상단
                       [cols - 50, rows - 50]] # 우하단. 좌하단 좌표는 세 점이 결정되었으므로, 알아서 결정되게 된다.
                       ).astype(np.float32)
    
    affine_mat = cv2.getAffineTransform(src_pts, dst_pts)

    print("affine_mat\n\n", affine_mat)

    dst = cv2.warpAffine(src, affine_mat, (0, 0))

    # 원본 점들을 변환 적용
    transformed_pts = cv2.transform(np.array([src_pts]), affine_mat)
    print("Original points:\n", src_pts)
    print("Transformed points:\n", transformed_pts)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_translation():
    src = cv2.imread('src/tekapo.bmp')

    # 이동변환 2x3 행렬을 직접 정의.
    affine_mat = np.array([[1, 0, 150], # a = 150
                           [0, 1, 100] # b = 100
                           ]).astype(np.float32)

    # warpAffine() 함수에 파라미터로 전달
    dst = cv2.warpAffine(src, affine_mat, (0, 0))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_shear_horizontal():
    src = cv2.imread('src/tekapo.bmp')

    rows = src.shape[0] # 세로
    cols = src.shape[1] # 가로

    m_x = 0.3

    affine_mat = np.array([[1, m_x, 0],
                           [0, 1, 0]]).astype(np.float32) # 가로 방향 어파인 변환 행렬
    
    dst = cv2.warpAffine(src, affine_mat, (int(cols + rows * m_x), rows))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_shear_vertical():
    src = cv2.imread('src/tekapo.bmp', cv2.IMREAD_COLOR)

    rows = src.shape[0] # 변경
    cols = src.shape[1] # 고정

    m_y = 0.2

    affine_mat = np.array([[1, 0, 0],
                           [m_y, 1, 0]]).astype(np.float32)

    dst = cv2.warpAffine(src, affine_mat, (cols, int(m_y * cols + rows)))

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_scale():
    src = cv2.imread('src/rose.bmp')

    dst1 = cv2.resize(src, (0, 0), fx = 4, fy = 4, interpolation = cv2.INTER_NEAREST) # fx 및 fy 명시(각 축 기준으로 4배씩 확대하겠다는 의미)

    dst2 = cv2.resize(src, (1920, 1280)) # interpolation 기법 = cv2.INTER_LINEAR -> 기본값!

    dst3 = cv2.resize(src, (1920, 1280), interpolation = cv2.INTER_CUBIC)

    dst4 = cv2.resize(src, (1920, 1280), interpolation = cv2.INTER_LANCZOS4)

    cv2.imshow('src', src)
    cv2.imshow('nearest', dst1)
    cv2.imshow('linear', dst2)
    cv2.imshow('cubic', dst3)
    cv2.imshow('lanczos4', dst4)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_rotation():
    src = cv2.imread('src/tekapo.bmp')

    center_point = (src.shape[1] / 2, src.shape[0] / 2)

    affine_mat = cv2.getRotationMatrix2D(center_point, 20, 0.5) # (회전 중심 좌표, 각도, 회전 후 추가적으로 확대 / 축소할 비율)

    print('affine matrix : \n\n', affine_mat)

    dst1 = cv2.warpAffine(src, affine_mat, (0, 0))

    dst2 = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('src', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def affine_flip():
    src = cv2.imread('src/eastsea.bmp')

    cv2.imshow('src', src)

    for flip_code in [1, 0, -1]:
        dst = cv2.flip(src, flip_code)

        desc = 'flipCode: %d' % flip_code
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()

    cv2.destroyAllWindows()




if __name__ == "__main__":
    # affine_transform()
    # affine_translation()
    # affine_shear_horizontal()
    # affine_shear_vertical()
    # affine_scale()
    # affine_rotation()
    affine_flip()
