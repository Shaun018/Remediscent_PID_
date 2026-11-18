
import easyocr
import cv2
import sys

reader = easyocr.Reader(['en'], gpu=False)


def extract_imprint(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Unable to load image")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enlarge image
    enlarged = cv2.resize(
        gray, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    processed = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(enlarged)

    # Adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    results = reader.readtext(
        threshold,
        detail=1,
        paragraph=False
    )

    print("\nIMPROVED OCR RESULTS")
    print("-------------------------")

    for box, text, confidence in results:

        if confidence >= 0.30:
            print(f"Text: {text}")
            print(f"Confidence: {confidence:.4f}")

    if not any(result[2] >= 0.30 for result in results):
        print("No reliable text detected.")

    return results


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
    else:
        extract_imprint(sys.argv[1])
