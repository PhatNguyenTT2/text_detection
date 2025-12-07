from ultralytics import YOLO
import cv2
import os

MODEL_PATH = os.path.join("model", "best.pt")
IMAGE_DIR = os.path.join("dataset", "images")
RESULT_DIR = os.path.join("results_detect")

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    image_name = "test.jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)

    # 1. Load model và ảnh
    model = YOLO(MODEL_PATH)
    img = cv2.imread(image_path)

    # 2. Chạy detect
    results = model(img)

    # 3. Vẽ bounding box + label bằng plot()
    for r in results:
        plotted = r.plot()  # trả về ảnh BGR đã vẽ box + label [web:23][web:70]

        # Hiển thị
        cv2.imshow("YOLOv8 detection", plotted)
        cv2.waitKey(0)

        # Lưu ra file
        out_path = os.path.join(RESULT_DIR, f"detect_{image_name}")
        cv2.imwrite(out_path, plotted)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
