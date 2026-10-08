import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model(
    "models/skin_binary_mobilenetv2.keras"
)

# Image path
img_path = input("Enter image path: ")

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array.astype("float32")

# Prediction
prediction = model.predict(img_array)[0][0]

print("\nPrediction Score:", prediction)

if prediction >= 0.5:
    print("Result: UNHEALTHY / ABNORMAL SKIN")
else:
    print("Result: HEALTHY / NORMAL SKIN")