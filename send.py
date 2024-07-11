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
    "img_path": "full_scan_examples/full_scan_example1_45nmx45nm.png",
    "scan_nm": 45,
    "contrast": 0.6,
    "rotation": 0,
}

result = send_request(data)
print(result)

# Save the results
with open("result.json", "w") as f:
    f.write(result)
