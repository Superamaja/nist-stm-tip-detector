import json
import socket


def send_request(data, host="localhost", port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Send the data
    client_socket.sendall(json.dumps(data).encode("utf-8"))

    # Receive the result
    result = client_socket.recv(4096).decode("utf-8")

    client_socket.close()

    return result  # Note: This might be JSON or an error message


data = {
    "scan_path": "full_scan_examples/20230131-180445_20211029 W31 P14--STM_AtomManipulation--128_2.Z_mtrx",  # The path to the scan image. Can be a .Z_mtrx file or a regular image file.
    "detector_options": {
        "contrast": 0.6,  # The contrast of the image. Default is 1.0.
        "rotation": 0.0,  # The rotation of the image. Default is 0.0.
    },
}

result = send_request(data)
print(result)

# Save the results
with open("send_results.json", "w") as f:
    f.write(json.dumps(json.loads(result), indent=4))
