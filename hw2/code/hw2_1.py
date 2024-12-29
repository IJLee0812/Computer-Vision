import numpy as np
import cv2
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

# HOG 디스크립터 파라미터
hog = cv2.HOGDescriptor(
    _winSize = (20, 20),
    _blockSize = (10, 10),
    _blockStride = (5, 5),
    _cellSize = (5, 5),
    _nbins = 9)

# SIFT 디스크립터 파라미터
sift = cv2.SIFT_create()

def preprocess_image(img):
    # 지역화: 비영역(0이 아닌 값이 있는 영역)을 잘라냄
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y : y + h, x : x + w]

    # 스케일링: 비율을 유지하면서 20x20으로 크기 조정
    scaled = cv2.resize(cropped, (20, 20), interpolation = cv2.INTER_AREA)

    return scaled


def extract_hog_features(images):
    # HOG 특징 추출
    features = [hog.compute(preprocess_image(img)).flatten() for img in images]
    return np.array(features)


def extract_sift_features(images, vector_size = 128):
    # SIFT 특징 추출
    features = []
    for img in images:
        gray = img
        kp, des = sift.detectAndCompute(gray, None)

        if des is not None:
            des = des.flatten()
            if len(des) > vector_size:
                des = des[:vector_size]
            else:
                des = np.pad(des, (0, vector_size - len(des)), 'constant')
        else:
            des = np.zeros(vector_size, dtype = np.float32)

        features.append(des)
    
    return np.array(features)


def prepare_mnist_data(feature_type = 'hog'):
    # MNIST 데이터셋 로드
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 데이터셋 크기 출력
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of test images: {len(test_images)}")

    # 이미지를 [0, 255] 범위로 정규화
    train_images = (train_images / 255.0 * 255).astype(np.uint8)
    test_images = (test_images / 255.0 * 255).astype(np.uint8)

    # 특징 유형에 따라 특징 추출
    print(f"Extracting {feature_type.upper()} features for training set...")
    start_time = time.time()
    if feature_type == 'hog':
        train_features = extract_hog_features(train_images)
    elif feature_type == 'sift':
        train_features = extract_sift_features(train_images)
    else:
        raise ValueError("Invalid feature type. Choose 'hog' or 'sift'.")
    print(f"Training feature extraction time: {time.time() - start_time:.2f}s")

    # 테스트 세트를 위한 특징 추출
    print(f"Extracting {feature_type.upper()} features for test set...")
    start_time = time.time()
    if feature_type == 'hog':
        test_features = extract_hog_features(test_images)
    elif feature_type == 'sift':
        test_features = extract_sift_features(test_images)
    print(f"Test feature extraction time: {time.time() - start_time:.2f}s")

    return train_features, test_features, train_labels, test_labels, train_images, test_images


def train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, train_images, test_images, classifier_type = 'knn', feature_type = 'hog'):
    # 분류기 훈련
    if classifier_type == 'knn':
        classifier = cv2.ml.KNearest_create()
        print("Training KNN...")
        start_time = time.time()
        classifier.train(train_features.astype(np.float32), cv2.ml.ROW_SAMPLE, train_labels.astype(np.int32))
        print(f"Training time: {time.time() - start_time:.2f}s")

        # KNN 테스트
        print("Testing KNN...")
        start_time = time.time()
        _, results, _, _ = classifier.findNearest(test_features.astype(np.float32), k = 5)
        print(f"Test inference time: {time.time() - start_time:.2f}s")

    elif classifier_type == 'svm':
        classifier = cv2.ml.SVM_create()
        classifier.setKernel(cv2.ml.SVM_RBF)
        classifier.setC(2.5)
        classifier.setGamma(0.5)
        print("Training SVM...")
        start_time = time.time()
        classifier.train(train_features.astype(np.float32), cv2.ml.ROW_SAMPLE, train_labels.astype(np.int32))
        print(f"Training time: {time.time() - start_time:.2f}s")

        # SVM 테스트
        print("Testing SVM...")
        start_time = time.time()
        _, results = classifier.predict(test_features.astype(np.float32))
        print(f"Test inference time: {time.time() - start_time:.2f}s")

    # 정확도 평가
    accuracy = accuracy_score(test_labels, results.flatten())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 10개의 랜덤 테스트 샘플에 대한 예측 시각화
    print("Visualizing predictions for 10 random test samples...")
    random_indices = random.sample(range(len(test_images)), 10)
    random_test_images = test_images[random_indices]
    random_test_labels = test_labels[random_indices]
    random_test_features = test_features[random_indices]

    # 선택된 샘플에 대한 레이블 예측
    if classifier_type == 'knn':
        _, random_predictions, _, _ = classifier.findNearest(random_test_features.astype(np.float32), k = 5)
    else:
        _, random_predictions = classifier.predict(random_test_features.astype(np.float32))
    random_predictions = random_predictions.flatten().astype(int)

    # 예측과 함께 샘플 시각화
    plt.figure(figsize = (10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(random_test_images[i], cmap = "gray")
        plt.title(f"True: {random_test_labels[i]}\nPred: {random_predictions[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def kNN_with_HOG():
    # HOG 특징을 사용한 KNN
    train_features, test_features, train_labels, test_labels, train_images, test_images = prepare_mnist_data(feature_type = 'hog')
    train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, train_images, test_images, classifier_type = 'knn', feature_type = 'hog')

def SVM_with_HOG():
    # HOG 특징을 사용한 SVM
    train_features, test_features, train_labels, test_labels, train_images, test_images = prepare_mnist_data(feature_type = 'hog')
    train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, train_images, test_images, classifier_type = 'svm', feature_type = 'hog')

def SVM_with_SIFT():
    # SIFT 특징을 사용한 SVM
    train_features, test_features, train_labels, test_labels, train_images, test_images = prepare_mnist_data(feature_type = 'sift')
    train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, train_images, test_images, classifier_type = 'svm', feature_type = 'sift')

if __name__ == "__main__":
    kNN_with_HOG()    # Case 1
    # SVM_with_HOG()    # Case 2
    # SVM_with_SIFT()   # Case 3