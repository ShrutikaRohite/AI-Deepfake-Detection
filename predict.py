import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("models/deepfake_model.h5")

# Path to new image
img_path = "data/sample/fake1.png"

# Preprocess image
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Predict
pred = model.predict(img_array)[0][0]

if pred > 0.5:
    print("Prediction: Fake")
else:
    print("Prediction: Real")
