import access2thematrix
import cv2
import numpy as np


def matrix_to_img_array(matrix_path: str, trace: int) -> np.ndarray:
    """Extracts the image from a matrix file and returns the image as a numpy array.

    Parameters:
        matrix_path (str): Path to the matrix file.
        trace (int): Trace number to extract the image from.

    Returns:
        ndarray: Image as a numpy array.
    """
    mtrx_data = access2thematrix.MtrxData()
    traces, _ = mtrx_data.open(matrix_path)

    # Check if the dictionary is empty
    if not traces:
        return None

    # Select the first image
    im, _ = mtrx_data.select_image(traces[trace])

    # Normalize the data to 0-255 and reflect over the x-axis (not sure why it's like this)
    img = (im.data - np.min(im.data)) / (np.max(im.data) - np.min(im.data)) * 255
    img = np.flipud(img)

    # Convert the image into a cv2 image
    img = np.array(img, dtype=np.uint8)

    # Convert the image to a 3 channel image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def get_nm_from_matrix(matrix_path: str) -> float:
    """Extracts the nm from the matrix path.

    Parameters:
        matrix_path (str): Path of the matrix file.

    Returns:
        float: The height of the scan in nm.
    """
    mtrx_data = access2thematrix.MtrxData()
    traces, _ = mtrx_data.open(matrix_path)

    # Check if the dictionary is empty
    if not traces:
        return None

    # Select the first image
    im, _ = mtrx_data.select_image(traces[0])

    # Extract the nm from the matrix path
    return im.height * 1e9
