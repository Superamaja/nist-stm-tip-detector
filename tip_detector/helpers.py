import cv2
import numpy as np


def preprocess_image(roi):
    """
    Preprocess the image before feeding it to the model.
    """
    if roi.shape[0] != 75 or roi.shape[1] != 75:
        roi = cv2.resize(roi, (75, 75), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi


def find_contours(img, alpha):
    """
    Processes the image and finds the contours.
    """
    img_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    gray = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, img_contrast, edged


def extract_roi(img, x, y, w, h):
    """
    Extracts the region of interest from the image.
    """
    # Calculate the size of the square
    square_size = max(w, h)

    # Calculate the coordinates for the square
    x = x + (w - square_size) // 2
    y = y + (h - square_size) // 2
    # Extract the square region of interest
    return img[y : y + square_size, x : x + square_size], x, y, square_size
