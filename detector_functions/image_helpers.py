import json

import cv2
import numpy as np

# Load config file
with open("config.json") as f:
    config = json.load(f)

SQUARE_PIXEL_SIZE = config["SQUARE_PIXEL_SIZE"]


def preprocess_image(roi):
    """
    Preprocess the image before feeding it to the model.
    """
    if roi.shape[0] != SQUARE_PIXEL_SIZE or roi.shape[1] != SQUARE_PIXEL_SIZE:
        roi = cv2.resize(
            roi, (SQUARE_PIXEL_SIZE, SQUARE_PIXEL_SIZE), interpolation=cv2.INTER_LINEAR
        )
    roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
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


def extract_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Extracts the region of interest from the image.

    Parameters:
        img (ndarray): Image to extract the region of interest from
        x1 (int): X coordinate of the top left corner of the square
        y1 (int): Y coordinate of the top left corner of the square
        x2 (int): X coordinate of the bottom right corner of the square
        y2 (int): Y coordinate of the bottom right corner of the square
    """
    # Calculate the size of the square
    square_size = max(x2 - x1, y2 - y1)

    # Calculate the coordinates for the square
    x = x1 + (x2 - x1) // 2 - square_size // 2
    y = y1 + (y2 - y1) // 2 - square_size // 2
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # Extract the square region of interest
    return img[y : y + square_size, x : x + square_size], x, y, square_size


def resize_roi(img: np.ndarray, x: int, y: int, square_size: int, new_size: int):
    """
    Expands the region of interest to the new size.

    Parameters:
        img (ndarray): Image to extract the region of interest from
        x (int): X coordinate of the top left corner of the original square
        y (int): Y coordinate of the top left corner of the original square
        square_size (int): Current size of the square
        new_size (int): New size of the square
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


def locate_brightest_pixel(img):
    """
    Locates the brightest pixel in the image.
    Finds the average location if there are multiple pixels with the same brightness.
    """
    max_value = np.max(img)
    y, x = np.where(img == max_value)
    return int(np.mean(x)), int(np.mean(y))


def rotate_image(img, angle):
    """
    Rotates the image by the specified angle.
    """
    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Get the average color of just the edges
    if len(img.shape) == 3:
        border_color = np.average(
            [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], axis=0
        )
    else:
        border_color = np.average([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])

    # Apply the rotation
    return cv2.warpAffine(
        img,
        M,
        (cols, rows),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
