import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('036_plate_0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hImg, wImg, _ = img.shape
cong = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img, config=cong)
print(boxes)

for x, b in enumerate(boxes.splitlines()):
  if x != 0:
    b = b.split()
    print(b)
    if len(b) == 12:
      x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
      cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
      cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

cv2.waitKey(0)