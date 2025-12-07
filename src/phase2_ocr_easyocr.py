import cv2
import easyocr
import numpy as np
import re

def init_ocr(langs=None, gpu=True):
    if langs is None:
        langs = ['en']
    reader = easyocr.Reader(langs, gpu=gpu)
    return reader

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)          # giảm nhiễu, giữ biên [web:84]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    h, w = gray.shape[:2]
    if max(h, w) < 80:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # adaptive threshold (bật nếu thấy hữu ích)
    # gray = cv2.adaptiveThreshold(gray, 255,
    #                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                              cv2.THRESH_BINARY, 31, 5)

    return gray

def postprocess_vn_plate(text_raw):
    """
    Chuẩn hoá chuỗi theo format phổ biến: 2 số + 1 chữ + 1 số + '-' + 5 số.
    Sửa một số lỗi OCR hay gặp giữa chữ/số. [web:44][web:108]
    """
    text = text_raw.replace(" ", "").upper()

    # map một số nhầm lẫn phổ biến
    mapping_digits = {
        "O": "0", "D": "0",        # O/D -> 0
        "I": "1", "L": "1",        # I/L -> 1
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8"
    }

    fixed = []
    for ch in text:
        if ch in mapping_digits:
            fixed.append(mapping_digits[ch])
        else:
            fixed.append(ch)
    fixed = "".join(fixed)

    # chèn '-' nếu thiếu mà độ dài tương ứng 2+1+1+5
    if "-" not in fixed and len(fixed) == 9:
        fixed = fixed[:4] + "-" + fixed[4:]

    # regex đơn giản cho dạng 90B2-45230
    pattern = r"^\d{2}[A-Z]\d-\d{5}$"
    if not re.match(pattern, fixed):
        return fixed  # nếu không khớp vẫn trả về để debug

    return fixed

def read_plate(plate_img, reader):
    processed = preprocess_plate(plate_img)
    result = reader.readtext(processed)

    if not result:
        return "", 0.0

    texts = [r[1] for r in result]
    confs = [float(r[2]) for r in result]
    text_joined = "".join(texts)
    conf_avg = sum(confs) / len(confs)

    final_text = postprocess_vn_plate(text_joined)
    return final_text, conf_avg
