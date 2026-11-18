
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
import tensorflow as tf
import easyocr
import numpy as np
import cv2
import json

app = Flask(__name__)
CORS(app)

# Load classification model
model = tf.keras.models.load_model(
    "models/best_pill_model.keras"
)

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Initialize OCR
reader = easyocr.Reader(["en"], gpu=False)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MedRemind Combined Identification API"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_bytes = request.files["image"].read()

        # Visual classification
        image = tf.keras.utils.load_img(
            BytesIO(image_bytes),
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

        # OCR
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray)

        imprint_results = [
            {
                "text": text,
                "confidence": round(float(confidence), 4)
            }
            for box, text, confidence in ocr_results
        ]

        return jsonify({
            "visual_predictions": visual_results,
            "extracted_imprints": imprint_results,
            "notice": "Candidate identification only. Verify with a pharmacist."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
