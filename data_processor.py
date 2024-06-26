import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

from helpers import get_ordered_fnames
from tip_detector.helpers import (
    extract_roi,
    find_contours,
    locate_brighthest_pixel,
    merge_contours,
    preprocess_image,
    resize_roi,
)

img_directory = "training_data"

DEBUG = False

# Load the configuration file
with open("config.json") as f:
    config = json.load(f)
SQUARE_NM_SIZE = config["SQUARE_NM_SIZE"]

# Check for system arguments
start_index = 0
if len(sys.argv) > 1:
    if "-d" in sys.argv:
        DEBUG = True
    if "-s" in sys.argv:
        start_index = int(sys.argv[sys.argv.index("-s") + 1])


fnames = get_ordered_fnames(img_directory)
features = pd.read_csv(os.path.join(img_directory, "features.csv"), sep=",")

for i, fname in enumerate(fnames):
    if i < start_index:
        continue
    # if not (
    #     features.iloc[i]["defectType"] == "DB" and features.iloc[i]["sampleBias"] > 0
    # ):
    #     print(f"Skipping {fname}")
    #     continue
    img = cv2.imread(fname)
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, img_contrast, edged_contrast = find_contours(img, 2)

    x1, y1, x2, y2 = 0, 0, 0, 0
    if len(contours) > 1:
        x1, y1, x2, y2 = merge_contours(contours)

    elif len(contours) == 1:
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2 = x1 + w
        y2 = y1 + h
    else:
        print(f"No contours found for {fname}")
        cv2.imshow("Scan", img)
        cv2.imshow("Contrast", img_contrast)
        cv2.imshow("Edges", edged_contrast)
        cv2.waitKey(0)
        continue

    # Extract the region of interest
    roi, x_roi, y_roi, square_size = extract_roi(gray, x1, y1, x2, y2)
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue

    # Find the brightest pixel in the ROI
    x_b, y_b = locate_brighthest_pixel(roi)

    # Calculator the size of the square in nm
    nm_p_pixel = [
        features.iloc[i]["scaleX"] / img.shape[1],
        features.iloc[i]["scaleY"] / img.shape[0],
    ]

    # Adjust the size of the square to be SQUARE_NM_SIZE nm
    # ! Assumes that the image is square
    if nm_p_pixel[0] * roi.shape[0] < SQUARE_NM_SIZE:
        roi, x_exp, y_exp, new_size = resize_roi(
            gray, x_roi, y_roi, square_size, int(SQUARE_NM_SIZE / nm_p_pixel[0])
        )
    else:
        # Creates a new roi and use the brightest pixel as the center of the new roi
        roi, x_exp, y_exp, new_size = extract_roi(
            gray,
            x_b + x_roi - int(SQUARE_NM_SIZE / nm_p_pixel[0] / 2),
            y_b + y_roi - int(SQUARE_NM_SIZE / nm_p_pixel[0] / 2),
            x_b + x_roi + int(SQUARE_NM_SIZE / nm_p_pixel[0] / 2),
            y_b + y_roi + int(SQUARE_NM_SIZE / nm_p_pixel[0] / 2),
        )

    roi_preprocessed = preprocess_image(roi)

    # Save the preprocessed image
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    cv2.imwrite(f"processed_data/{fname.split('\\')[-1]}", roi_preprocessed[0] * 255)

    # Clone the csv over
    if not os.path.exists("processed_data/features.csv"):
        features.to_csv("processed_data/features.csv", index=False)

    if DEBUG:
        cv2.rectangle(
            img,
            (x_roi, y_roi),
            (x_roi + square_size, y_roi + square_size),
            (255, 255, 0),
            0,
        )
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            0,
        )
        cv2.rectangle(
            img,
            (x_exp, y_exp),
            (x_exp + new_size, y_exp + new_size),
            (0, 0, 255),
            0,
        )
        cv2.circle(
            img,
            (x_roi + x_b, y_roi + y_b),
            1,
            (0, 0, 255),
            -1,
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                img2,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                0,
            )

        print()
        print(f"Image: {fname}")
        print(f"Tip Quality: {features.iloc[i]['tipQuality']}")
        print(f"Resolution: {img.shape[0]} x {img.shape[1]}")
        print(f"Scale: {features.iloc[i]['scaleX']} x {features.iloc[i]['scaleY']}")
        print(f"ROI Resolution: {roi.shape[0]} x {roi.shape[1]}")
        cv2.imshow("Scan", img)
        cv2.imshow("All Contours", img2)
        cv2.imshow("Contrast", img_contrast)
        cv2.imshow("Edges", edged_contrast)
        cv2.imshow("ROI", roi_preprocessed[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
