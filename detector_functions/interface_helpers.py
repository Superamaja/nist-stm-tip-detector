import json
import os


def extract_nm_from_path(image_path):
    """
    Extracts the nm from the image path.
    """
    try:
        temp_path = image_path
        temp_path = temp_path.split("nmx")[1]
        scan_nm = float(temp_path.split("nm")[0])
        print(f"Detected scan size: {scan_nm} nm")
    except:
        print("Could not detect scan size from the image path")
        scan_nm = float(input("Enter the scan size in nm: "))
    return scan_nm


def create_scan_configs(scan_configs):
    if not os.path.exists("configs"):
        os.makedirs("configs")
    with open("configs/scan_configs.json", "w") as f:
        json.dump(scan_configs, f, indent=4)
