"""
Example input JSON:
{
    "scan_path": "path/to/scan.Z_mtrx", # The path to the scan image. Can be a .Z_mtrx file or a regular image file.
    "detector_options": {
        "scan_nm": 10, # DO NOT USE FOR MATRIX FILES UNLESS OVERRIDING. The scan size in nanometers.
        "contrast": 1.0, # The contrast of the image. Default is 1.0.
        "rotation": 0.0 # The rotation of the image. Default is 0.0.
    }
}

Example success JSON:
{
    "sharp": 10, # The number of sharp tips detected.
    "dull": 5, # The number of dull tips detected.
    "total": 15 # The total number of tips detected.
    "roi_data": [
        {
            "x": 10, # The top left x-coordinate of the ROI.
            "y": 20, # The top left y-coordinate of the ROI.
            "size": 30, # The size of the ROI.
            "prediction": 0.8, # The prediction of the ROI.
        }
    ]
}

Example error JSON:
{
    "error": "Error opening the matrix file: {Exception message}"
}
"""

import json
import select
import socket
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

from detector_functions.main_functions import detect_tip
from detector_functions.matrix_helpers import get_nm_from_matrix, matrix_to_img_array

# Load config file
with open("config.json") as f:
    config = json.load(f)

# Load the model
model = load_model("model.h5")


def convert_to_serializable(obj):
    # Dictionary has numpy types scattered throughout. This function recursively converts them to Python types.
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def process_image(data: dict) -> dict:
    if "scan_path" not in data or "detector_options" not in data:
        raise ValueError("Invalid input data")

    scan_path, detector_options = data.values()

    if scan_path.endswith(".Z_mtrx"):
        img = matrix_to_img_array(scan_path)
        scan_nm = (
            get_nm_from_matrix(scan_path)
            if "scan_nm" not in detector_options
            else detector_options["scan_nm"]
        )
    else:
        img = cv2.imread(scan_path)
        scan_nm = detector_options.get("scan_nm", 0)
        if scan_nm == 0:
            raise ValueError("Scan size in nanometers is required for regular images")

    contrast = detector_options.get("contrast", 1.0)
    rotation = detector_options.get("rotation", 0.0)

    tip_data = detect_tip(
        img,
        scan_nm=scan_nm,
        roi_nm_size=config["ROI_NM_SIZE"],
        model=model,
        cross_size=config["DETECTOR_CROSS_SIZE"],
        contrast=contrast,
        rotation=rotation,
        display_results=False,
        debug=False,
    )

    # Convert numpy types to Python types
    serializable_data = convert_to_serializable(tip_data)
    return serializable_data


def receive_json(client_socket: socket.socket, timeout=5) -> dict:
    data = ""
    start_time = time.time()
    while True:
        ready = select.select([client_socket], [], [], timeout)
        if ready[0]:
            chunk = client_socket.recv(1024).decode("utf-8")
            if not chunk:
                raise ConnectionError("Connection closed by client")
            data += chunk
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout while waiting for complete JSON")
                continue
        else:
            raise TimeoutError("Timeout while waiting for data")


def handle_client(client_socket: socket.socket) -> None:
    try:
        # Receive data from the client
        input_data = receive_json(client_socket)

        # Process the image
        result = process_image(input_data)

        # Send the result back to the client
        client_socket.send(json.dumps(result).encode("utf-8"))

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Invalid JSON data: {str(e)}"
        client_socket.send(json.dumps({"error": error_message}).encode("utf-8"))
    except TimeoutError as e:
        error_message = f"Timeout error: {str(e)}"
        client_socket.send(json.dumps({"error": error_message}).encode("utf-8"))
    except ConnectionError as e:
        error_message = f"Connection error: {str(e)}"
        print(error_message)  # Log to server console
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        client_socket.send(json.dumps({"error": error_message}).encode("utf-8"))

    finally:
        # Close the connection
        client_socket.close()


def start_server(host="localhost", port=5050) -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Accepted connection from {addr}")
        handle_client(client_socket)


if __name__ == "__main__":
    start_server()
