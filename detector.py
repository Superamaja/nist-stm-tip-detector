import json
import os

import cv2
from tensorflow.keras.models import load_model  # type: ignore

from detector_functions.interface_helpers import (
    create_scan_configs,
    extract_nm_from_path,
    get_configs,
)
from detector_functions.main_functions import detect_tip

SCAN_CONFIG_PARAMETERS = [
    "scan_nm",
    "contrast",
    "rotation",
]

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
    scan_configs[new_path] = get_configs(
        new_path,
        scan_configs[new_path] if new_path in scan_configs else None,
        SCAN_CONFIG_PARAMETERS,
    )

print("Saving scan configs...")
create_scan_configs(scan_configs)
print(scan_configs)


model = load_model("model.h5")

for image_path in paths:
    scan_nm = scan_configs[image_path]["scan_nm"]
    contrast = scan_configs[image_path]["contrast"]
    rotation = scan_configs[image_path]["rotation"]

    img = cv2.imread(image_path)

    output = detect_tip(
        img,
        scan_nm=scan_nm,
        square_nm_size=config["SQUARE_NM_SIZE"],
        model=model,
        cross_size=config["DETECTOR_CROSS_SIZE"],
        contrast=contrast,
        rotation=rotation,
        display_results=config["DETECTOR_SCAN_DEBUG"],
        debug=config["DETECTOR_ROI_DEBUG"],
    )
