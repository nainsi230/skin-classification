import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# PATHS
# ==============================

MODEL_PATH = r"D:\Skin_Classification\models\skin_binary_mobilenetv2.keras"
VAL_DIR = r"D:\Skin_Classification\dataset\binary_split\val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ==============================
# LOAD MODEL
# ==============================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# LOAD VALIDATION DATASET
# ==============================

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["Healthy", "Unhealthy"],
    shuffle=False
)

print("\nClass names:", val_data.class_names)

# ==============================
# MODEL EVALUATION
# ==============================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

results = model.evaluate(val_data, verbose=1)

print("\nEvaluation Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ==============================
# GET PREDICTIONS
# ==============================

print("\nGenerating predictions...")

predictions = model.predict(val_data, verbose=1).ravel()

# Convert probabilities to classes
predicted_classes = (predictions >= 0.5).astype(int)

# Get actual labels
true_classes = np.concatenate([
    y.numpy().ravel()
    for x, y in val_data
]).astype(int)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(true_classes, predicted_classes)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=["Healthy", "Unhealthy"],
        digits=4
    )
)

# ==============================
# PREDICTION STATISTICS
# ==============================

print("\n==============================")
print("PREDICTION STATISTICS")
print("==============================")

print("Minimum prediction:", predictions.min())
print("Maximum prediction:", predictions.max())
print("Average prediction:", predictions.mean())
print("Median prediction:", np.median(predictions))

# ==============================
# SEPARATE HEALTHY / UNHEALTHY
# ==============================

healthy_predictions = predictions[true_classes == 0]
unhealthy_predictions = predictions[true_classes == 1]

print("\n==============================")
print("HEALTHY IMAGE PREDICTIONS")
print("==============================")

print("Count:", len(healthy_predictions))
print("Minimum:", healthy_predictions.min())
print("Maximum:", healthy_predictions.max())
print("Average:", healthy_predictions.mean())
print("Median:", np.median(healthy_predictions))

print("\n==============================")
print("UNHEALTHY IMAGE PREDICTIONS")
print("==============================")

print("Count:", len(unhealthy_predictions))
print("Minimum:", unhealthy_predictions.min())
print("Maximum:", unhealthy_predictions.max())
print("Average:", unhealthy_predictions.mean())
print("Median:", np.median(unhealthy_predictions))

# ==============================
# SAMPLE PREDICTIONS
# ==============================

print("\n==============================")
print("FIRST 20 PREDICTIONS")
print("==============================")

for i in range(min(20, len(predictions))):
    actual = "Healthy" if true_classes[i] == 0 else "Unhealthy"
    predicted = "Healthy" if predicted_classes[i] == 0 else "Unhealthy"

    print(
        f"{i+1}. Actual: {actual:10} | "
        f"Score: {predictions[i]:.4f} | "
        f"Predicted: {predicted}"
    )

print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")