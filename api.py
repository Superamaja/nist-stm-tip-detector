import json
import os
import select
import socket
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

from detector_functions.main_functions import detect_tip

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
    img_path, scan_nm, contrast, rotation = data.values()
    img = cv2.imread(img_path)

    tip_data = detect_tip(
        img,
        scan_nm=scan_nm,
        roi_nm_size=config["ROI_NM_SIZE"],
        model=model,
        cross_size=config["DETECTOR_CROSS_SIZE"],
        contrast=contrast,
        rotation=rotation,
        display_results=config["DETECTOR_SCAN_DEBUG"],
        debug=config["DETECTOR_ROI_DEBUG"],
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
        client_socket.send(error_message.encode("utf-8"))
    except TimeoutError as e:
        error_message = f"Timeout error: {str(e)}"
        client_socket.send(error_message.encode("utf-8"))
    except ConnectionError as e:
        error_message = f"Connection error: {str(e)}"
        print(error_message)  # Log to server console
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        client_socket.send(error_message.encode("utf-8"))

    finally:
        # Close the connection
        client_socket.close()


def start_server(host="localhost", port=12345) -> None:
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
