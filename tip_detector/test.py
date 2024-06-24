import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

image_path = "full_scan_examples/sharp_single.png"

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
model = load_model("stm_model.h5")
# model = load_model("model.h5")


# Define a function to preprocess the image for the model
def preprocess_image(roi):
    roi = cv2.resize(roi, (75, 75), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi


# Load the image file
img = cv2.imread(image_path)
img2 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def process_image(img, alpha):
    global img_contrast, gray_contrast, blurred_contrast, edged_contrast, contours

    # Lowering contrast avoids impurities and prevents the need of size thresholding
    img_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    gray_contrast = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)
    blurred_contrast = cv2.GaussianBlur(gray_contrast, (5, 5), 0)
    edged_contrast = cv2.Canny(blurred_contrast, 50, 150)
    contours, _ = cv2.findContours(
        edged_contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )


# Process the image with different contrast levels

roi = preprocess_image(gray)

prediction = model.predict(roi)[0][0]

print("Prediction:", prediction)
