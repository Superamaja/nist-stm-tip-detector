"""
Unit tests for the send_request function.

This module contains a series of unit tests for the send_request function, which
connects to a specified API via a socket, sends a JSON payload, and validates the response.
The tests are designed to check the following:
- The response is not None.
- The response is valid JSON.
- The response contains the expected keys: "sharp", "dull", "total", and "roi_data".
- The counts of "total" are as expected.

Note:
The expected count for "total" is hard-coded to 31.
These counts may change if any modifications are made to the detector logic in the API.
If the test for "total" counts fail, review the changes made to the detector logic and update the expected count accordingly.
There should be no affect from changing the model.

Reminder:
Ensure that api.py is running before executing these tests.

Usage:
python test_send_request.py
"""

import json
import socket
import unittest


def send_request(data, host="localhost", port=5050):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    # Send the data
    client_socket.sendall(json.dumps(data).encode("utf-8"))
    # Receive the result
    result = client_socket.recv(4096).decode("utf-8")
    client_socket.close()
    return result  # Note: This might be JSON or an error message


class TestSendRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {
            "scan_path": "full_scan_examples/20230131-180445_20211029 W31 P14--STM_AtomManipulation--128_2.Z_mtrx",
            "detector_options": {
                "contrast": 0.6,
                "rotation": 0.0,
            },
            "matrix_options": {
                "direction": 0,
            },
        }
        cls.result = send_request(cls.data)
        try:
            cls.result_dict = json.loads(cls.result)
        except json.JSONDecodeError:
            cls.result_dict = None

    def test_result_not_none(self):
        self.assertIsNotNone(self.result, "Result should not be None")

    def test_result_is_valid_json(self):
        self.assertIsNotNone(self.result_dict, "Result should be valid JSON")

    def test_result_contains_expected_keys(self):
        if self.result_dict is None:
            self.fail("Result dict is None, cannot check for expected keys")
        expected_keys = ["sharp", "dull", "total", "roi_data"]
        for key in expected_keys:
            self.assertIn(
                key, self.result_dict, f"Result should contain the key: {key}"
            )

    def test_total_count(self):
        if self.result_dict is None:
            self.fail("Result dict is None, cannot check sharp count")
        self.assertEqual(self.result_dict["total"], 31, "Expected total count")


if __name__ == "__main__":
    unittest.main()
