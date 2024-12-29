import argparse
import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
def load_model(weights_path):
    model = YOLO(weights_path)  # Ultralytics의 YOLOv8 API로 모델 로드
    return model

# 이미지에서 엠파이어 스테이트 빌딩 검출
def detect_building(model, image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # 모델을 사용하여 객체 검출
    results = model.predict(source = img, save = False, conf = 0.60)  # 예측 실행, confidence threshold 설정
    detections = results[0].boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2, conf, cls], ...]
    confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도 값
    classes = results[0].boxes.cls.cpu().numpy()  # 클래스 번호
    labels = model.names  # 클래스 이름 리스트
    target_class = "empire-state-building"  # 엠파이어 스테이트 빌딩 클래스 이름

    # 각 객체를 순회하며 처리
    for i, box in enumerate(detections):
        cls = int(classes[i])
        conf = confidences[i]
        if labels[cls] == target_class:  # 지정된 클래스와 신뢰도 조건에 맞는 경우 True 출력 후 리턴
            print("\n\nEmpire State Building in Image : True\n")
            return
    
    print("\n\nEmpire State Building in Image : False\n")
    return

# 커맨드라인 인자 처리
def parse_args():
    parser = argparse.ArgumentParser(description = "Detect Empire State Building in an image")
    parser.add_argument('image', type = str, help = "Path to the image to detect Empire State Building")
    parser.add_argument('--weights', type = str, default = './best.pt', help = "Path to the trained YOLOv8 weights")
    return parser.parse_args()

def main():
    # 커맨드라인 인자 파싱
    args = parse_args()

    # 모델 로드
    model = load_model(args.weights)

    # 이미지에서 엠파이어 스테이트 빌딩 감지
    detect_building(model, args.image)

if __name__ == '__main__':
    main()