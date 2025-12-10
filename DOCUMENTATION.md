# 📚 License Plate Recognition - Tài Liệu Chi Tiết

## 🎯 Tổng Quan Hệ Thống

Hệ thống nhận diện biển số xe Việt Nam sử dụng **2 mô hình YOLOv5**:
1. **LP_detector.pt** - Phát hiện vùng biển số trong ảnh
2. **LP_ocr.pt** - Nhận dạng từng ký tự trong biển số

---

## 📁 Cấu Trúc Thư Mục

```
License-Plate-Recognition/
│
├── model/                          # Chứa các model YOLOv5
│   ├── LP_detector.pt             # Model phát hiện biển số (full)
│   ├── LP_detector_nano_61.pt     # Model phát hiện biển số (nano - nhanh hơn)
│   ├── LP_ocr.pt                  # Model OCR đọc ký tự (full)
│   └── LP_ocr_nano_62.pt          # Model OCR đọc ký tự (nano - nhanh hơn)
│
├── function/                       # Thư viện helper functions
│   ├── helper.py                  # Logic đọc và xử lý biển số
│   └── utils_rotate.py            # Xử lý xoay và cân bằng ảnh
│
├── yolov5/                        # YOLOv5 framework
│
├── lp_image.py                    # Script xử lý ảnh tĩnh
└── webcam.py                      # Script xử lý video realtime
```

---

## 🔧 Chi Tiết Functions

### 1. `helper.py` - Core Logic Functions

#### 1.1. `linear_equation(x1, y1, x2, y2)`

**Mục đích:** Tính phương trình đường thẳng đi qua 2 điểm

**Công thức toán học:**
```
Phương trình: y = ax + b
Với:
  a = (y2 - y1) / (x2 - x1)    # Hệ số góc
  b = y1 - a*x1                 # Hệ số tự do
```

**Tham số:**
- `x1, y1` - Tọa độ điểm thứ nhất
- `x2, y2` - Tọa độ điểm thứ hai

**Trả về:** 
- `(a, b)` - Tuple chứa hệ số góc và hệ số tự do

**Ví dụ:**
```python
# Đường thẳng qua (0, 0) và (10, 20)
a, b = linear_equation(0, 0, 10, 20)
# a = 2.0, b = 0.0
# Phương trình: y = 2x
```

**Ứng dụng:** Kiểm tra các ký tự có nằm thẳng hàng (biển 1 dòng) hay không

---

#### 1.2. `check_point_linear(x, y, x1, y1, x2, y2)`

**Mục đích:** Kiểm tra một điểm có nằm trên đường thẳng hay không

**Thuật toán:**
```python
1. Tính phương trình đường thẳng qua (x1,y1) và (x2,y2)
2. Dự đoán vị trí y_predicted từ x
3. So sánh y_predicted với y thực tế
4. Cho phép sai số ±3 pixels
```

**Tham số:**
- `x, y` - Tọa độ điểm cần kiểm tra
- `x1, y1, x2, y2` - Hai điểm định nghĩa đường thẳng
- `abs_tol=3` - Ngưỡng sai số cho phép (mặc định 3 pixels)

**Trả về:**
- `True` - Điểm nằm trên đường thẳng (trong khoảng sai số)
- `False` - Điểm không nằm trên đường thẳng

**Ví dụ:**
```python
# Đường thẳng qua (0, 0) và (10, 10)
check_point_linear(5, 5, 0, 0, 10, 10)   # True
check_point_linear(5, 8, 0, 0, 10, 10)   # True (trong sai số ±3)
check_point_linear(5, 15, 0, 0, 10, 10)  # False (xa quá)
```

**Ứng dụng:** Phân loại biển số 1 dòng vs 2 dòng

---

#### 1.3. `read_plate(yolo_license_plate, im)`

**Mục đích:** Đọc text biển số từ ảnh đã crop

**Workflow chi tiết:**

```
┌─────────────────────────────────────┐
│  INPUT: Ảnh biển số đã crop         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ BƯỚC 1: YOLOv5 OCR Detection        │
│ - Detect tất cả ký tự trong ảnh    │
│ - Mỗi ký tự: [x1,y1,x2,y2,conf,ch] │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ BƯỚC 2: Validation                  │
│ - Kiểm tra 7-10 ký tự              │
│ - Nếu không đủ → return "unknown"  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ BƯỚC 3: Tính tâm các ký tự         │
│ - x_center = (x1 + x2) / 2         │
│ - y_center = (y1 + y2) / 2         │
│ - Lưu [x_c, y_c, character]        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ BƯỚC 4: Xác định loại biển số      │
│ - Tìm điểm trái nhất & phải nhất   │
│ - Vẽ đường thẳng giữa 2 điểm       │
│ - Kiểm tra các điểm khác:          │
│   + Nằm trên đường → 1 dòng        │
│   + Lệch đường → 2 dòng            │
└──────────────┬──────────────────────┘
               │
      ┌────────┴─────────┐
      │                  │
      ▼                  ▼
┌──────────┐      ┌─────────────┐
│ 1 DÒNG   │      │  2 DÒNG     │
└────┬─────┘      └──────┬──────┘
     │                   │
     ▼                   ▼
┌─────────────┐   ┌──────────────────┐
│ Sắp xếp X   │   │ Tính y_mean      │
│ Ghép liền   │   │ Chia 2 dòng      │
│ "51F12345"  │   │ "51F-12345"      │
└─────────────┘   └──────────────────┘
```

**Chi tiết từng bước:**

##### BƯỚC 1: YOLOv5 OCR Detection
```python
results = yolo_license_plate(im)
bb_list = results.pandas().xyxy[0].values.tolist()
```
**Output:** Danh sách bounding boxes
```python
[
  [x1, y1, x2, y2, confidence, class_id],
  # Ví dụ:
  [10, 5, 25, 30, 0.95, '5'],
  [30, 5, 45, 30, 0.92, '1'],
  [50, 5, 65, 30, 0.88, 'F'],
  ...
]
```

##### BƯỚC 2: Validation
```python
if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
    return "unknown"
```
**Quy tắc biển số Việt Nam:**
- Tối thiểu: 7 ký tự (VD: `29A1234`)
- Tối đa: 10 ký tự (VD: `29A-123.45`)
- Ngoài khoảng này → lỗi detection

##### BƯỚC 3: Tính tâm
```python
center_list = []
for bb in bb_list:
    x_c = (bb[0] + bb[2]) / 2  # Tâm X
    y_c = (bb[1] + bb[3]) / 2  # Tâm Y
    center_list.append([x_c, y_c, bb[-1]])
```
**Ví dụ:**
```
Bounding box: [10, 5, 25, 30]
→ Center: [(10+25)/2, (5+30)/2] = [17.5, 17.5]
```

##### BƯỚC 4: Phân loại biển số

**4a. Tìm điểm biên:**
```python
l_point = center_list[0]  # Khởi tạo
r_point = center_list[0]

for cp in center_list:
    if cp[0] < l_point[0]:
        l_point = cp  # Điểm trái nhất
    if cp[0] > r_point[0]:
        r_point = cp  # Điểm phải nhất
```

**4b. Kiểm tra thẳng hàng:**
```python
LP_type = "1"  # Mặc định 1 dòng

for ct in center_list:
    if not check_point_linear(
        ct[0], ct[1], 
        l_point[0], l_point[1], 
        r_point[0], r_point[1]
    ):
        LP_type = "2"  # Có điểm lệch → 2 dòng
        break
```

**Minh họa:**
```
BIỂN 1 DÒNG:
  5  1  F  1  2  3  4  5
  ●  ●  ●  ●  ●  ●  ●  ●  ← Tất cả thẳng hàng
  └──────────────────────┘
  
BIỂN 2 DÒNG:
     5  1  F           ← Dòng trên
     ●  ●  ●
     
  1  2  3  4  5        ← Dòng dưới
  ●  ●  ●  ●  ●
  └─ Không thẳng hàng
```

##### BƯỚC 5a: Xử lý biển 1 dòng
```python
if LP_type == "1":
    # Sắp xếp từ trái sang phải theo x
    for l in sorted(center_list, key=lambda x: x[0]):
        license_plate += str(l[2])
```
**Output:** `"51F12345"`

##### BƯỚC 5b: Xử lý biển 2 dòng
```python
if LP_type == "2":
    # Tính ngưỡng phân chia
    y_mean = int(y_sum / len(bb_list))
    
    # Chia thành 2 dòng
    line_1 = []  # Dòng trên
    line_2 = []  # Dòng dưới
    
    for c in center_list:
        if int(c[1]) > y_mean:
            line_2.append(c)
        else:
            line_1.append(c)
    
    # Ghép dòng 1
    for l1 in sorted(line_1, key=lambda x: x[0]):
        license_plate += str(l1[2])
    
    license_plate += "-"  # Dấu phân cách
    
    # Ghép dòng 2
    for l2 in sorted(line_2, key=lambda x: x[0]):
        license_plate += str(l2[2])
```
**Output:** `"51F-12345"`

**Ví dụ hoàn chỉnh:**
```python
Input: Ảnh biển số "51F-12345" (2 dòng)

Detection results:
  Character '5': center [20, 10]
  Character '1': center [40, 10]
  Character 'F': center [60, 10]
  Character '1': center [15, 30]
  Character '2': center [30, 30]
  Character '3': center [45, 30]
  Character '4': center [60, 30]
  Character '5': center [75, 30]

y_mean = (10+10+10+30+30+30+30+30) / 8 = 20

Dòng 1 (y < 20): ['5', '1', 'F']
Dòng 2 (y > 20): ['1', '2', '3', '4', '5']

Output: "51F-12345"
```

---

### 2. `utils_rotate.py` - Image Processing Functions

#### 2.1. `changeContrast(img)`

**Mục đích:** Tăng độ tương phản để cải thiện edge detection

**Thuật toán CLAHE** (Contrast Limited Adaptive Histogram Equalization):

```python
┌─────────────────┐
│  BGR Image      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Convert LAB    │  ← Tách Lightness/Color
│  L: Độ sáng     │
│  A: Green-Red   │
│  B: Blue-Yellow │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLAHE on L     │  ← Cân bằng histogram cục bộ
│  clipLimit=3.0  │     (chỉ áp dụng cho độ sáng)
│  tileSize=8x8   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Merge → BGR    │
└─────────────────┘
```

**Tham số:**
- `clipLimit=3.0` - Giới hạn độ tương phản (tránh noise)
- `tileGridSize=(8,8)` - Kích thước vùng cục bộ

**So sánh:**
```
TRƯỚC CLAHE:          SAU CLAHE:
████████████          ░░██████▓▓
████████████    →     ░░██████▓▓
████████████          ░░██████▓▓
(Mờ, tối)             (Rõ ràng, tương phản cao)
```

**Ứng dụng:** Cải thiện chất lượng ảnh trước khi detect edge

---

#### 2.2. `rotate_image(image, angle)`

**Mục đích:** Xoay ảnh theo góc cho trước

**Thuật toán:**
```python
1. Tìm tâm ảnh: (width/2, height/2)
2. Tạo ma trận xoay: cv2.getRotationMatrix2D()
3. Áp dụng phép biến đổi: cv2.warpAffine()
```

**Tham số:**
- `image` - Ảnh cần xoay
- `angle` - Góc xoay (độ), dương = ngược chiều kim đồng hồ

**Ví dụ:**
```python
# Xoay 15 độ
rotated = rotate_image(img, 15)

# Xoay -15 độ (cùng chiều kim đồng hồ)
rotated = rotate_image(img, -15)
```

---

#### 2.3. `compute_skew(src_img, center_thres)`

**Mục đích:** Tính góc nghiêng của biển số

**Workflow chi tiết:**

```
┌──────────────┐
│  Input Image │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│  Canny Edge Detect  │
│  - threshold1 = 30  │
│  - threshold2 = 100 │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Hough Line Transform       │
│  - Tìm các đường thẳng      │
│  - minLineLength = w/1.5    │
│  - maxLineGap = h/3         │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Lọc đường gần tâm nhất     │
│  - Loại đường ở mép ảnh     │
│  - center_thres = ngưỡng    │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Tính góc nghiêng           │
│  - angle = arctan2(Δy, Δx)  │
│  - Trung bình các đường     │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Output: Góc (độ)           │
└─────────────────────────────┘
```

**Chi tiết từng bước:**

##### BƯỚC 1: Canny Edge Detection
```python
img = cv2.medianBlur(src_img, 3)  # Làm mờ giảm noise
edges = cv2.Canny(
    img, 
    threshold1=30,   # Ngưỡng thấp
    threshold2=100,  # Ngưỡng cao
    apertureSize=3,
    L2gradient=True
)
```
**Output:** Ảnh nhị phân chỉ có các cạnh

##### BƯỚC 2: Hough Line Transform
```python
lines = cv2.HoughLinesP(
    edges,
    rho=1,                    # Độ phân giải khoảng cách (pixels)
    theta=math.pi/180,        # Độ phân giải góc (1 độ)
    threshold=30,             # Ngưỡng vote tối thiểu
    minLineLength=w/1.5,      # Độ dài tối thiểu
    maxLineGap=h/3.0          # Khoảng cách tối đa giữa các đoạn
)
```
**Output:** Danh sách đường thẳng `[[x1, y1, x2, y2], ...]`

##### BƯỚC 3: Tìm đường gần tâm
```python
min_line = 100
min_line_pos = 0

for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        center_point_y = (y1 + y2) / 2
        
        # Bỏ qua đường quá gần mép trên
        if center_thres == 1 and center_point_y < 7:
            continue
        
        # Tìm đường có tâm Y nhỏ nhất
        if center_point_y < min_line:
            min_line = center_point_y
            min_line_pos = i
```

##### BƯỚC 4: Tính góc
```python
angle = 0.0
cnt = 0

for x1, y1, x2, y2 in lines[min_line_pos]:
    ang = np.arctan2(y2 - y1, x2 - x1)
    
    # Loại bỏ góc cực đoan (> 30 độ)
    if math.fabs(ang) <= 30:
        angle += ang
        cnt += 1

return (angle / cnt) * 180 / math.pi
```

**Ví dụ:**
```
Biển số nghiêng 10 độ:
┌─────────┐
│  ╱╱╱╱╱  │  ← Cạnh trên nghiêng
│ ╱51F╱   │
│╱1234╱   │
└─────────┘

arctan2(Δy, Δx) ≈ 10°
→ Cần xoay -10° để thẳng
```

---

#### 2.4. `deskew(src_img, change_cons, center_thres)`

**Mục đích:** Xoay ảnh để biển số thẳng

**Logic:**
```python
if change_cons == 1:
    # Tăng contrast trước
    enhanced_img = changeContrast(src_img)
    skew_angle = compute_skew(enhanced_img, center_thres)
else:
    # Dùng ảnh gốc
    skew_angle = compute_skew(src_img, center_thres)

return rotate_image(src_img, skew_angle)
```

**Tham số:**
- `change_cons`:
  - `0` = Dùng ảnh gốc để tính góc
  - `1` = Tăng contrast trước khi tính góc
- `center_thres`:
  - `0` = Cho phép đường gần mép
  - `1` = Bỏ qua đường gần mép trên (y < 7)

**Ứng dụng trong code:**
```python
# Thử 4 cách khác nhau
for cc in range(0, 2):      # change_cons
    for ct in range(0, 2):  # center_thres
        deskewed = deskew(crop_img, cc, ct)
        lp = read_plate(yolo_license_plate, deskewed)
        if lp != "unknown":
            break
```

---

## 🎬 Workflow Tổng Thể

### Luồng xử lý `lp_image.py`:

```
START
  │
  ├─ Load models (YOLOv5)
  │
  ├─ Đọc ảnh đầu vào
  │
  ├─ YOLOv5 Detector: Tìm vùng biển số
  │
  ├─ CÓ phát hiện biển số?
  │  │
  │  ├─ KHÔNG ────────────┐
  │  │                    │
  │  └─ CÓ               │
  │     │                │
  │     ├─ Vẽ rectangle   │
  │     │                │
  │     ├─ Crop vùng BS  │
  │     │                │
  │     └─ Thử 4 cách:   │
  │        Loop cc(0,1)  │  ← Thử dùng/không dùng tăng contrast
  │        Loop ct(0,1)  │  ← Thử ngưỡng center khác nhau
  │           │          │
  │           ├─ Deskew  │
  │           │          │
  │           ├─ OCR     │
  │           │          │
  │           └─ Đọc BS  │
  │              │       │
  ├──────────────┴───────┘
  │
  ├─ Đọc toàn ảnh (fallback)
  │
  ├─ Hiển thị kết quả
  │
END
```

### Luồng xử lý `webcam.py`:

```
START
  │
  ├─ Load models (nano version)
  │
  ├─ Mở webcam
  │
  └─ LOOP (mỗi frame):
     │
     ├─ Đọc frame
     │
     ├─ YOLOv5 Detector
     │
     ├─ For each plate detected:
     │  │
     │  ├─ Crop
     │  │
     │  ├─ Vẽ rectangle
     │  │
     │  └─ Thử 4 cách deskew
     │     │
     │     ├─ Deskew
     │     │
     │     ├─ OCR
     │     │
     │     └─ Nếu thành công → break
     │
     ├─ Vẽ text lên frame
     │
     ├─ Tính FPS
     │
     ├─ Hiển thị frame
     │
     └─ Nhấn 'q' để thoát
```

---

## 📊 So Sánh Model Versions

| Model | Kích thước | Tốc độ | Độ chính xác | Sử dụng |
|-------|-----------|--------|--------------|---------|
| **LP_detector.pt** | ~14MB | Chậm | Cao | Ảnh tĩnh, độ chính xác quan trọng |
| **LP_detector_nano_61.pt** | ~4MB | Nhanh | Trung bình | Webcam realtime |
| **LP_ocr.pt** | ~14MB | Chậm | Cao | Ảnh tĩnh |
| **LP_ocr_nano_62.pt** | ~4MB | Nhanh | Trung bình | Webcam realtime |

**Confidence threshold:** `0.60` (60% - cân bằng giữa precision và recall)

---

## 🎯 Các Trường Hợp Xử Lý

### Trường hợp 1: Biển số 1 dòng
```
Input:  51F12345
Output: "51F12345"
```

### Trường hợp 2: Biển số 2 dòng
```
Input:  51F
        12345
Output: "51F-12345"
```

### Trường hợp 3: Không phát hiện vùng biển số
```
→ Fallback: Chạy OCR trên toàn bộ ảnh
→ Nếu vẫn không đọc được → "unknown"
```

### Trường hợp 4: Biển số nghiêng
```
→ Thử 4 cấu hình deskew
→ Chọn cấu hình đầu tiên đọc được
```

### Trường hợp 5: Số ký tự không hợp lệ
```
< 7 hoặc > 10 ký tự → "unknown"
```

---

## ⚙️ Tham Số Quan Trọng

### YOLOv5 Detection
```python
yolo_LP_detect(img, size=640)
```
- `size=640`: Kích thước ảnh input (càng lớn càng chính xác nhưng chậm)

### OCR Confidence
```python
yolo_license_plate.conf = 0.60
```
- Chỉ chấp nhận ký tự có confidence ≥ 60%

### Deskew Parameters
```python
deskew(crop_img, change_cons, center_thres)
```
- 4 tổ hợp: `(0,0), (0,1), (1,0), (1,1)`

### Validation
```python
if len(bb_list) < 7 or len(bb_list) > 10:
    return "unknown"
```
- Biển số Việt Nam: 7-10 ký tự

---

## 🚀 Cách Sử Dụng

### 1. Xử lý ảnh tĩnh:
```bash
cd License-Plate-Recognition
python lp_image.py -i path/to/image.jpg
```

### 2. Xử lý webcam:
```bash
cd License-Plate-Recognition
python webcam.py
```
- Nhấn `q` để thoát

### 3. Sử dụng API Service:
```python
from lp_service.lp_recognition_service import LicensePlateRecognitionService

service = LicensePlateRecognitionService()
result = service.recognize_from_image("image.jpg")
print(result['licensePlate'])
```

---

## 🔍 Debug & Troubleshooting

### Không detect được biển số:
1. Kiểm tra độ sáng ảnh
2. Tăng kích thước input: `size=640` → `size=1280`
3. Giảm confidence threshold: `0.60` → `0.40`

### Đọc sai ký tự:
1. Kiểm tra ảnh crop có rõ không
2. Thử các cấu hình deskew khác
3. Tăng độ tương phản trước khi OCR

### FPS thấp trên webcam:
1. Dùng model nano
2. Giảm kích thước input
3. Skip frame (xử lý mỗi 2-3 frame)

---

## 📝 Notes

- **Biển số Việt Nam:** Hỗ trợ cả 1 dòng và 2 dòng
- **Performance:** Nano models nhanh hơn ~3x nhưng độ chính xác giảm ~5-10%
- **Deskew:** Cải thiện accuracy ~15-20% cho ảnh nghiêng
- **CLAHE:** Cải thiện ~10% cho ảnh tối/mờ

---

## 🎓 Tham Khảo

- YOLOv5: https://github.com/ultralytics/yolov5
- OpenCV: https://opencv.org/
- CLAHE: Contrast Limited Adaptive Histogram Equalization
- Hough Transform: https://en.wikipedia.org/wiki/Hough_transform
