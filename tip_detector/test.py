import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from helpers import preprocess_image

image_path = "full_scan_examples/example_0.png"

# Images 1, 2 use (6,6) with 0.01
CONTOUR_MIN_SIZE = (6, 6)  # Minimum size of the contour to pass (width, height)
SHARP_PREDICTION_THRESHOLD = 0.01  # Prediction threshold for sharpness - Greater than or equal to this value is sharp, otherwise blunt

RED = (50, 50, 255)
GREEN = (0, 255, 0)
BLUE = (255, 200, 0)

CLASS_NAMES = {
    0: "Sharp",
    1: "Dull",
}

# Handle arguments
if len(sys.argv) > 1:
    image_path = sys.argv[1]

# Load the pre-trained model
print("loading model")
# model = load_model("stm_model.h5")
model = load_model("model.h5")

# Load the image file
img = cv2.imread(image_path)
img2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Process the image with different contrast levels

roi = preprocess_image(gray)

prediction = model.predict(roi)[0][0]

print("Prediction:", prediction)
