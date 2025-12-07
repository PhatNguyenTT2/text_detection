# src/preprocess.py
import cv2
import numpy as np

def preprocess_plate(plate_path: str):
    """
    Tiền xử lý ảnh biển số tối ưu cho OCR
    - Cải thiện chất lượng ảnh
    - Tăng contrast
    - Denoise
    - Thử nhiều phương pháp threshold
    """
    img = cv2.imread(plate_path)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize về chiều cao chuẩn (120px cho chữ rõ hơn)
    h, w = gray.shape[:2]
    target_h = 120
    target_w = int(w * target_h / h)
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Denoise trước khi tăng contrast
    denoised = cv2.fastNlMeansDenoising(resized, h=7, templateWindowSize=7, searchWindowSize=21)
    
    # Tăng contrast bằng CLAHE với tham số mạnh hơn
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Sharpen để làm rõ cạnh chữ
    kernel_sharpen = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # Otsu threshold - thường tốt nhất cho biển số
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Kiểm tra nếu background sáng hơn text -> đảo ngược
    mean_val = np.mean(thresh)
    if mean_val > 127:  # Background trắng
        thresh = cv2.bitwise_not(thresh)
    
    # Morphology nhẹ để làm sạch noise nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Scale lên 2.5x cho OCR đọc tốt hơn (tổng ~3x)
    final = cv2.resize(cleaned, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    return final
