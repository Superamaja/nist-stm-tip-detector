import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

image_path = "full_scan_examples/full_scan_example1.png"

# Images 1, 2 use (6,6) with 0.01
CONTOUR_MIN_SIZE = (6, 6)  # Minimum size of the contour to pass (width, height)
SHARP_PREDICTION_THRESHOLD = 0.01  # Prediction threshold for sharpness - Greater than or equal to this value is sharp, otherwise blunt

RED = (50, 50, 255)
GREEN = (0, 255, 0)
BLUE = (255, 200, 0)

CLASS_NAMES = {
    0: "Dull",
    1: "Sharp",
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
process_image(img, 0.6)
if len(contours) == 0:
    process_image(img, 1.0)

total_bonds = 0
total_cls = {0: 0, 1: 0}
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if (
        w >= CONTOUR_MIN_SIZE[0] and h >= CONTOUR_MIN_SIZE[1]
    ):  # filter out small contours
        roi = gray[y : y + h, x : x + w]
        roi_preprocessed = preprocess_image(roi)
        prediction = model.predict(roi_preprocessed)[0][0]
        # cls = np.argmax(prediction)
        cls = 1 if prediction >= SHARP_PREDICTION_THRESHOLD else 0
        print(f"Class: {CLASS_NAMES[cls]}, Prediction: {prediction}")

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN if cls == 1 else RED, 0)

        # Object details
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.35
        thickness = 1

        # if prediction >= 0.005:
        #     cv2.putText(
        #         img,
        #         f"{prediction:.2f}",
        #         (x, y - 5),
        #         font,
        #         fontScale,
        #         BLUE,
        #         thickness,
        #     )

        # Count the number of contours
        total_bonds += 1
        total_cls[cls] += 1

decision = 0 if total_cls[0] > total_cls[1] else 1
percent = total_cls[decision] / total_bonds * 100

print(f"Total bonds: {total_bonds}")
print(f"Total sharp: {total_cls[0]}")
print(f"Total blunt: {total_cls[1]}")
print(f"Overall Decision: {CLASS_NAMES[decision]}")
print(f"{CLASS_NAMES[decision]} Percentage: {percent:.4f}%")

cv2.imshow("Scan", img)
cv2.imshow("Contrast", img_contrast)
cv2.imshow("Edges", edged_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
