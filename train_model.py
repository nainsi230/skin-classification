import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================

TRAIN_DIR = r"D:\Skin_Classification\dataset\binary_split\train"
VAL_DIR = r"D:\Skin_Classification\dataset\binary_split\val"

MODEL_DIR = Path(r"D:\Skin_Classification\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["Healthy", "Unhealthy"],
    shuffle=True,
    seed=42
)

# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["Healthy", "Unhealthy"],
    shuffle=False
)

print("\nClass names:", train_data.class_names)

# ============================================================
# 5. DATA PREFETCHING
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

# Healthy = 0
# Unhealthy = 1

healthy_count = 247
unhealthy_count = 1931

classes = np.array([0, 1])
labels = np.array(
    [0] * healthy_count +
    [1] * unhealthy_count
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("\nClass weights:")
print("Healthy:", class_weights[0])
print("Unhealthy:", class_weights[1])

# ============================================================
# 7. MOBILE NET V2
# ============================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ============================================================
# 8. BUILD MODEL
# ============================================================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # MobileNetV2 expects pixels in [-1, 1]
    layers.Rescaling(1.0 / 127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ============================================================
# 10. DISPLAY MODEL
# ============================================================

model.summary()

# ============================================================
# 11. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_DIR / "skin_binary_mobilenetv2.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ============================================================
# 12. TRAIN MODEL
# ============================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ============================================================
# 13. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_DIR / "skin_binary_mobilenetv2_final.keras"
)

print("\n===================================")
print("Training completed successfully!")
print("Model saved inside:")
print(MODEL_DIR)
print("===================================")