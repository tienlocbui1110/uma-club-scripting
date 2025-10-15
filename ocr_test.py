import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en',
    ocr_version='PP-OCRv5',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    return_word_box=False,
)

image_path = "frame.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

result = ocr.ocr(image)
print("\n=== RAW RESULT ===")
print(result)

texts = []
for page in result:
    if "rec_texts" in page:
        texts.extend(page["rec_texts"])

print("\n=== DETECTED TEXTS ===")
for t in texts:
    print(" -", t)
