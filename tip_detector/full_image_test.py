import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from helpers import preprocess_image

image_path = "full_scan_examples/example_5.png"


# Load the pre-trained model
print("loading model")
model = load_model("model.h5")

# Load the image file
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

roi = preprocess_image(gray)

prediction = model.predict(roi)[0][0]

print("Prediction:", prediction)
