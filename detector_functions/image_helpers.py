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
        # borderValue=border_color,
        borderValue=(255, 255, 255),  # ! TEMP CHANGE
    )


def merge_overlapping_contours(contours, overlap_threshold=0.5):
    """
    Merge contours that overlap by more than a certain threshold.
    Code produced by GPT-4o.
    """
    merged_contours = []
    contours = list(contours)  # Convert tuple to list
    while contours:
        base = contours.pop(0)
        base_rect = cv2.boundingRect(base)
        base_x, base_y, base_w, base_h = base_rect
        merged = False
        i = 0
        while i < len(contours):
            cnt = contours[i]
            cnt_rect = cv2.boundingRect(cnt)
            cnt_x, cnt_y, cnt_w, cnt_h = cnt_rect
            if max(base_x, cnt_x) < min(base_x + base_w, cnt_x + cnt_w) and max(
                base_y, cnt_y
            ) < min(base_y + base_h, cnt_y + cnt_h):
                if (min(base_x + base_w, cnt_x + cnt_w) - max(base_x, cnt_x)) * (
                    min(base_y + base_h, cnt_y + cnt_h) - max(base_y, cnt_y)
                ) >= overlap_threshold * base_w * base_h:
                    merged_contour = np.vstack((base, cnt))
                    contours.pop(i)
                    contours.insert(0, merged_contour)
                    merged = True
                    break
            i += 1
        if not merged:
            merged_contours.append(base)
    return merged_contours


def calculate_mode_color_ratio(img, pt1, pt2):
    """
    Calculate the ratio of the mode color to the total number of pixels in the region.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    mode_color = np.bincount(img[y1:y2, x1:x2].flatten()).argmax()
    num_mode_color = np.sum(img[y1:y2, x1:x2] == mode_color)

    # Divide by 3 if the image is in color
    if len(img.shape) == 3:
        num_mode_color /= 3

    return num_mode_color / ((x2 - x1) * (y2 - y1))
