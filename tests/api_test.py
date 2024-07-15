import json
import logging
import os
import socket
import subprocess
import time
import unittest

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
HOST = "localhost"
PORT = 5050
API_SCRIPT = "api.py"


def send_request(data, host=HOST, port=PORT):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        client_socket.sendall(json.dumps(data).encode("utf-8"))
        result = client_socket.recv(4096).decode("utf-8")
        return result
    except Exception as e:
        logger.error(f"Error in send_request: {e}")
        return None
    finally:
        client_socket.close()


class SendRequestTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Construct the command to run api.py
        api_script_path = API_SCRIPT
        if not os.path.exists(api_script_path):
            raise FileNotFoundError(f"API script not found at {api_script_path}")

        cmd = ["python", api_script_path]

        logger.info(f"Starting API server with command: {' '.join(cmd)}")

        # Start the api.py server
        try:
            cls.api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise

        # Wait for the server to start
        time.sleep(2)

        # Check if the process is still running
        if cls.api_process.poll() is not None:
            stdout, stderr = cls.api_process.communicate()
            logger.error(
                f"API server failed to start. Stdout: {stdout}, Stderr: {stderr}"
            )
            raise RuntimeError("API server failed to start")

        logger.info("API server started successfully")

    @classmethod
    def tearDownClass(cls):
        logger.info("Stopping API server")
        if hasattr(cls, "api_process"):
            cls.api_process.terminate()
            cls.api_process.wait(timeout=5)
            logger.info("API server stopped")
        else:
            logger.warning("API process not found during teardown")

    def test_send_request(self):
        data = {
            "scan_path": "full_scan_examples/20230131-180445_20211029 W31 P14--STM_AtomManipulation--128_2.Z_mtrx",
            "detector_options": {
                "contrast": 0.6,
                "rotation": 0.0,
            },
        }

        result = send_request(data)

        # Check if the result is valid
        self.assertIsNotNone(result, "Result should not be None")

        # Check if the result is a valid JSON string
        try:
            result_dict = json.loads(result)
        except json.JSONDecodeError:
            self.fail("Result is not a valid JSON string")

        # Check if the result contains the expected keys
        expected_keys = ["sharp", "dull", "total", "roi_data"]
        for key in expected_keys:
            self.assertIn(key, result_dict, f"Result should contain the key: {key}")

        # Check if the results are specific values (could be wrong with detector logic changes)
        self.assertEqual(result_dict["sharp"], 26, "Expected sharp count")
        self.assertEqual(result_dict["dull"], 5, "Expected dull count")


if __name__ == "__main__":
    unittest.main()
