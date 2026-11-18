
import tensorflow as tf
import easyocr
import numpy as np
import cv2
import json
import sys

# Load classification model
model = tf.keras.models.load_model(
    "models/best_pill_model.keras"
)

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Initialize OCR
reader = easyocr.Reader(["en"], gpu=False)


def identify_pill(image_path):

    # Image classification
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    visual_results = [
        {
            "medicine": class_names[i],
            "score": round(float(predictions[i]) * 100, 2)
        }
        for i in top_indices
    ]

    # OCR extraction
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ocr_results = reader.readtext(gray)

    imprint_results = [
        {
            "text": text,
            "confidence": round(float(confidence), 4)
        }
        for box, text, confidence in ocr_results
    ]

    return {
        "visual_predictions": visual_results,
        "extracted_imprints": imprint_results,
        "notice": "Candidate identification only. Verify with a pharmacist."
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python combined_prediction.py <image_path>")
    else:
        result = identify_pill(sys.argv[1])

        print("\nCOMBINED PILL IDENTIFICATION")
        print("-------------------------")

        print("\nVisual predictions:")
        for item in result["visual_predictions"]:
            print(item)

        print("\nOCR results:")
        for item in result["extracted_imprints"]:
            print(item)

        print("\n", result["notice"])
