import os

import cv2
import numpy as np
import pandas as pd

from tip_detector.helpers import extract_roi, find_contours, preprocess_image

img_directory = "training_data"

example_fnames0 = []
for file in os.listdir(img_directory):
    if file.endswith(".png"):
        example_fnames0.append(os.path.join(img_directory, file))

# re-order according to the index specified at the end of each file name: *_[index].png
# (e.g. example_5.png should correspond to an index of 5)
# this is important because the .csv file assumes this order
fname_order = []
for fname in example_fnames0:
    end = fname.split("_")[-1]
    num_str = end.split(".")[0]
    fname_order.append(int(num_str))

# example_fnames will hold the correctly ordered set of file names
example_fnames = [None] * len(example_fnames0)
old_idx = 0
for idx in fname_order:
    example_fnames[idx] = example_fnames0[old_idx]
    old_idx += 1

# read in the features/labels which are ordered the same as img_data
features = pd.read_csv(os.path.join(img_directory, "features.csv"), sep=",")


for i, fname in enumerate(example_fnames):
    if not (
        features.iloc[i]["defectType"] == "DB" and features.iloc[i]["sampleBias"] > 0
    ):
        print(f"Skipping {fname}")
        continue
    img = cv2.imread(fname)
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, img_contrast, edged_contrast = find_contours(img, 2)

    x, y, w, h = 0, 0, 0, 0
    if len(contours) > 1:
        # Merge multiple contours into one by combining their furthest points
        least_x, least_y, largest_w, largest_h = cv2.boundingRect(contours[0])
        for cnt in contours[1:]:
            x, y, w, h = cv2.boundingRect(cnt)
            if x < least_x:
                largest_w += least_x - x
                least_x = x
            if y < least_y:
                largest_h += least_y - y
                least_y = y
            if x + w > least_x + largest_w:
                largest_w = x + w - least_x
            if y + h > least_y + largest_h:
                largest_h = y + h - least_y

        x, y, w, h = least_x, least_y, largest_w, largest_h

    elif len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        print(f"No contours found for {fname}")
        cv2.imshow("Scan", img)
        cv2.imshow("Contrast", img_contrast)
        cv2.imshow("Edges", edged_contrast)
        cv2.waitKey(0)
        continue

    # Extract the region of interest
    roi, x, y, square_size = extract_roi(gray, x, y, w, h)
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        continue
    roi_preprocessed = preprocess_image(roi)
    cv2.rectangle(
        img,
        (x, y),
        (x + square_size, y + square_size),
        (0, 255, 0),
        0,
    )
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(
            img2,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            0,
        )

    print()
    print(f"Image: {fname}")
    print(f"Tip Quality: {features.iloc[i]['tipQuality']}")
    print(f"Scale: {features.iloc[i]['scaleX']} x {features.iloc[i]['scaleY']}")
    cv2.imshow("Scan", img)
    cv2.imshow("All Contours", img2)
    cv2.imshow("Contrast", img_contrast)
    cv2.imshow("Edges", edged_contrast)
    cv2.imshow("ROI", roi_preprocessed[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
