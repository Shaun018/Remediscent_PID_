
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

TRAIN_DIR = "dataset/split/train"
VAL_DIR = "dataset/split/validation"

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names

if len(class_names) != 10:
    raise ValueError(f"Expected 10 classes, found {len(class_names)}")

if val_ds.class_names != class_names:
    raise ValueError("Training and validation class names do not match")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

# Save class names
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# Optimize input pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomContrast(0.1)
])

# Pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Training callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_pill_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save("models/pill_model.keras")

# Save training history
with open("outputs/training_history.json", "w") as f:
    json.dump(
        {k: [float(v) for v in values]
         for k, values in history.history.items()},
        f,
        indent=4
    )

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training Accuracy")
plt.legend()
plt.savefig("outputs/accuracy.png")
plt.close()

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Training Loss")
plt.legend()
plt.savefig("outputs/loss.png")
plt.close()

print("\nTraining completed!")
print("Best model: models/best_pill_model.keras")
print("Final model: models/pill_model.keras")
print("Graphs saved in outputs/")
