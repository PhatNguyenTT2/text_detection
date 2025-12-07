# src/detect.py
from ultralytics import YOLO
import cv2
import os

MODEL_PATH = os.path.join("model", "best.pt")
IMAGE_DIR = os.path.join("dataset", "images")
RESULT_DETECT_DIR = os.path.join("results_detect")
RESULT_CROP_DIR = os.path.join("results_crop")

def detect_image(image_name: str):
    os.makedirs(RESULT_DETECT_DIR, exist_ok=True)
    os.makedirs(RESULT_CROP_DIR, exist_ok=True)

    image_path = os.path.join(IMAGE_DIR, image_name)

    model = YOLO(MODEL_PATH)
    img = cv2.imread(image_path)

    results = model(img)

    plates = []   # lưu bbox biển số

    for r in results:
        plotted = r.plot()

        # Lấy bbox
        boxes = r.boxes
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box.data[0].tolist()
            if conf < 0.5:
                continue
            if int(cls) != 0:   # giả sử class 0 = biển số
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img[y1:y2, x1:x2]

            crop_name = f"{os.path.splitext(image_name)[0]}_plate_{i}.jpg"
            crop_path = os.path.join(RESULT_CROP_DIR, crop_name)
            cv2.imwrite(crop_path, crop)

            plates.append(crop_path)

        out_path = os.path.join(RESULT_DETECT_DIR, f"detect_{image_name}")
        cv2.imwrite(out_path, plotted)
        cv2.imshow("YOLOv8 detection", plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plates  # list path ảnh crop

if __name__ == "__main__":
    detect_image("043.jpg")
