import json
import os

import cv2
from tensorflow.keras.models import load_model  # type: ignore

from tip_detector.interface_helpers import create_scan_configs, extract_nm_from_path
from tip_detector.main_functions import detect_tip

# Load config file
with open("config.json") as f:
    config = json.load(f)

# Load scan configs
if os.path.exists("configs/scan_configs.json"):
    with open("configs/scan_configs.json") as f:
        scan_configs = json.load(f)
else:
    scan_configs = {}

configs_changed = False
paths = []
for path in config["SCANS"]:
    new_path = config["SCAN_FOLDER"] + path
    paths.append(new_path)
    if not new_path in scan_configs:
        print(f"No config found for {new_path}")
        scan_configs[new_path] = {
            "scan_nm": extract_nm_from_path(new_path),
            "contrast": float(input("Enter the contrast value: ")),
        }
        configs_changed = True

if configs_changed:
    print("Saving scan configs...")
    create_scan_configs(scan_configs)
    print(scan_configs)


model = load_model("model.h5")

for image_path in paths:
    scan_nm = scan_configs[image_path]["scan_nm"]
    contrast = scan_configs[image_path]["contrast"]

    img = cv2.imread(image_path)

    detect_tip(
        img,
        scan_nm=scan_nm,
        square_nm_size=config["SQUARE_NM_SIZE"],
        model=model,
        cross_size=config["DETECTOR_CROSS_SIZE"],
        contrast=contrast,
        display_results=config["DETECTOR_SCAN_DEBUG"],
        debug=config["DETECTOR_ROI_DEBUG"],
    )
