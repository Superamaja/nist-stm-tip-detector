import json
import sys

import cv2
from tensorflow.keras.models import load_model  # type: ignore

from helpers import extract_roi, find_contours, preprocess_image, resize_roi

with open("../config.json") as f:
    config = json.load(f)

SQUARE_NM_SIZE = config["SQUARE_NM_SIZE"]

image_path = "full_scan_examples/full_scan_example1.png"

# Images 1, 2 use (6,6) with 0.01
CONTOUR_MIN_SIZE = (6, 6)  # Minimum size of the contour to pass (width, height)
SHARP_PREDICTION_THRESHOLD = 0.5  # Prediction threshold for sharpness - Greater than or equal to this value is sharp, otherwise dull
DEBUG = False

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
# model = load_model("stm_model.h5")
model = load_model("model.h5")


# Load the image file
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Process the image with different contrast levels
contours, img_contrast, edged_contrast = find_contours(img, 0.6)
if len(contours) == 0:
    contours, img_contrast, edged_contrast = find_contours(img, 1)

total_bonds = 0
total_cls = {0: 0, 1: 0}

# TODO: Calculate nm/pixel
nm_p_pixel = 45 / img.shape[1]

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w >= CONTOUR_MIN_SIZE[0] and h >= CONTOUR_MIN_SIZE[1]:
        roi, x, y, square_size = extract_roi(gray, x, y, x + w, y + h)
        roi, x, y, new_size = resize_roi(
            gray, x, y, square_size, int(SQUARE_NM_SIZE / nm_p_pixel)
        )
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue
        roi_preprocessed = preprocess_image(roi)
        prediction = model.predict(roi_preprocessed)[0][0]
        # cls = np.argmax(prediction)
        cls = 1 if prediction >= SHARP_PREDICTION_THRESHOLD else 0
        print(f"Class: {CLASS_NAMES[cls]}, Prediction: {prediction}")

        # Draw bounding box
        cv2.rectangle(
            img,
            (x, y),
            (x + new_size, y + new_size),
            GREEN if cls else RED,
            0,
        )

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

        if DEBUG:
            cv2.imshow("ROI2", roi_preprocessed[0])
            cv2.moveWindow("ROI2", 75, 400)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

decision = 0 if total_cls[0] > total_cls[1] else 1
percent = total_cls[decision] / total_bonds * 100

print(f"Total bonds: {total_bonds}")
print(f"Total sharp: {total_cls[1]}")
print(f"Total dull: {total_cls[0]}")
print(f"Overall Decision: {CLASS_NAMES[decision]}")
print(f"{CLASS_NAMES[decision]} Percentage: {percent:.4f}%")

cv2.imshow("Scan", img)
cv2.imshow("Contrast", img_contrast)
cv2.imshow("Edges", edged_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
