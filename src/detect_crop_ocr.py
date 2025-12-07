from ultralytics import YOLO
import cv2
import os
from phase2_ocr_easyocr import init_ocr, read_plate


MODEL_PATH = os.path.join("model", "best.pt")
IMAGE_DIR = os.path.join("dataset", "images")


def main():
    image_name = "test.jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)


    model = YOLO(MODEL_PATH)
    img = cv2.imread(image_path)


    results = model(img)


    reader = init_ocr(['en'], gpu=True)


    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            conf_det = float(box.conf[0].item())


            if cls != 0:
                continue


            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop_img = img[y1:y2, x1:x2]


            text, conf_ocr = read_plate(crop_img, reader)
            print(f"DET_CONF={conf_det:.2f}  OCR_CONF={conf_ocr:.2f}  PLATE={text}")


if __name__ == "__main__":
    main()
