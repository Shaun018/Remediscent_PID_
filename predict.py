
import tensorflow as tf
import numpy as np
import json
import sys

from pathlib import Path

MODEL_PATH = "models/best_pill_model.keras"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


def predict_pill(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    print("\nTOP 3 PREDICTIONS")
    print("-------------------------")

    for index in top_indices:
        print(
            f"{class_names[index]}: "
            f"{predictions[index] * 100:.2f}%"
        )

    print("\nNote: These are model scores, not verified medicine identities.")
    print("Confirm medicine identity with a pharmacist before use.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_pill(sys.argv[1])
