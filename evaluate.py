
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TEST_DIR = "dataset/split/test"
MODEL_PATH = "models/best_pill_model.keras"

# Load saved class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

if test_ds.class_names != class_names:
    raise ValueError("Test class names do not match training class names")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate test loss and accuracy
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTEST RESULTS")
print("-------------------------")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print("-------------------------")
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(13, 11))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    xticks_rotation=90,
    values_format="d"
)

plt.title("Pill Identification - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# Save individual predictions
results = pd.DataFrame({
    "Actual": [class_names[i] for i in y_true],
    "Predicted": [class_names[i] for i in y_pred],
    "Top_1_Score": np.max(y_prob, axis=1),
    "Correct": y_true == y_pred
})

results.to_csv("outputs/test_predictions.csv", index=False)

print("\nEvaluation completed!")
print("Classification report: outputs/classification_report.txt")
print("Confusion matrix: outputs/confusion_matrix.png")
print("Predictions: outputs/test_predictions.csv")
