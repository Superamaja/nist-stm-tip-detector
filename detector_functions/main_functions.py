import cv2
import numpy as np

from detector_functions.image_helpers import (
    extract_roi,
    find_contours,
    locate_brighthest_pixel,
    preprocess_image,
    resize_roi,
    rotate_image,
)

# Constants
CONTOUR_MIN_SIZE = (6, 6)  # Minimum size of the contour to pass (width, height)
SHARP_PREDICTION_THRESHOLD = 0.5  # Prediction threshold for sharpness - Greater than or equal to this value is sharp, otherwise dull
CLASS_NAMES = {
    0: "Dull",
    1: "Sharp",
}

# Colors
RED = (50, 50, 255)
GREEN = (0, 255, 0)
BLUE = (255, 200, 0)


def detect_tip(
    img,
    scan_nm,
    model,
    square_nm_size=2,
    cross_size=0,
    contrast=1,
    rotation=45,
    display_results=False,
    debug=False,
):
    if rotation != 0:
        img = rotate_image(img, rotation)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, img_contrast, edged_contrast = find_contours(img, contrast)

    total_bonds = 0
    total_cls = {0: 0, 1: 0}
    nm_p_pixel = scan_nm / img.shape[1]
    brightest_locations = set()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= CONTOUR_MIN_SIZE[0] and h >= CONTOUR_MIN_SIZE[1]:
            # Extract the ROI and resize it to a square
            roi, x, y, _ = extract_roi(gray, x, y, x + w, y + h)
            new_size = int(square_nm_size / nm_p_pixel)

            # Remove duplicates from brightness centering
            x_b, y_b = locate_brighthest_pixel(roi)
            if (x_b + x, y_b + y) in brightest_locations:
                continue
            brightest_locations.add((x_b + x, y_b + y))

            # Perform the cross check
            cross_predictions = []
            new_x = x + x_b - new_size // 2
            new_y = y + y_b - new_size // 2
            for direction in range(2):
                for shift in range(-cross_size, cross_size + 1):
                    roi, x, y, new_size = resize_roi(
                        gray,
                        new_x + shift * direction,
                        new_y + shift * (1 - direction),
                        new_size,
                        new_size,
                    )
                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue
                    roi_preprocessed = preprocess_image(roi)
                    cross_predictions.append(model.predict(roi_preprocessed)[0][0])
            prediction = np.max(cross_predictions)
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

            if debug:
                cv2.imshow("Scan", img)
                cv2.imshow("ROI2", roi_preprocessed[0])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if display_results:
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
