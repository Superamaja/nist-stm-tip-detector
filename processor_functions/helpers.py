import cv2


def box_all_image_contours(contours):
    """
    Creates a bounding box around all the contours.
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
