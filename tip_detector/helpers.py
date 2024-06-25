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


def merge_contours(contours):
    """
    Merges the contours into one by combining their furthest points.
    """
    left_x, top_y, w, h = cv2.boundingRect(contours[0])
    right_x = left_x + w
    bottom_y = top_y + h
    for cnt in contours[1:]:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < left_x:
            left_x = x
        if y < top_y:
            top_y = y
        if x + w > right_x:
            right_x = x + w
        if y + h > bottom_y:
            bottom_y = y + h

    return left_x, top_y, right_x, bottom_y


def extract_roi(img, x1, y1, x2, y2):
    """
    Extracts the region of interest from the image.
    """
    # Calculate the size of the square
    square_size = max(x2 - x1, y2 - y1)

    # Calculate the coordinates for the square
    x = x1 + (x2 - x1) // 2 - square_size // 2
    y = y1 + (y2 - y1) // 2 - square_size // 2

    # Extract the square region of interest
    return img[y : y + square_size, x : x + square_size], x, y, square_size


def resize_roi(img, x, y, square_size, new_size):
    """
    Expands the region of interest to the new size.
    """
    x_new = x - (new_size - square_size) // 2
    y_new = y - (new_size - square_size) // 2
    if x_new < 0:
        x_new = 0
    if y_new < 0:
        y_new = 0
    return (
        img[y_new : y_new + new_size, x_new : x_new + new_size],
        x_new,
        y_new,
        new_size,
    )


def locate_brighthest_pixel(img):
    """
    Locates the brightest pixel in the image.
    Finds the average location if there are multiple pixels with the same brightness.
    """
    max_value = np.max(img)
    y, x = np.where(img == max_value)
    return int(np.mean(x)), int(np.mean(y))
