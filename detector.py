import csv
import json
import os
import sys

# Disable OneDNN optimizations and CPU instructions messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from tensorflow.keras.models import load_model  # type: ignore

from detector_functions.interface_helpers import get_configs, initialize_scan_configs
from detector_functions.main_functions import detect_tip
from detector_functions.matrix_helpers import matrix_to_img_array

SCAN_CONFIG_PARAMETERS = [
    "scan_nm",
    "contrast",
    "rotation",
]
MATRIX_CONFIG_PARAMETERS = [
    "direction",
]

# Load config file
with open("config.json") as f:
    config = json.load(f)

# Handle arguments
if len(sys.argv) > 1:
    if "-sd" in sys.argv:
        config["DETECTOR_SCAN_DEBUG"] = True
    if "-rd" in sys.argv:
        config["DETECTOR_ROI_DEBUG"] = True

# Load scan configs
if os.path.exists("configs/scan_configs.json"):
    with open("configs/scan_configs.json") as f:
        scan_configs = json.load(f)
else:
    scan_configs = {}

configs_changed = False
paths = []
for path in config["SCANS"]:
    param = SCAN_CONFIG_PARAMETERS.copy()
    if path.endswith(".Z_mtrx"):
        param += MATRIX_CONFIG_PARAMETERS.copy()
    new_path = config["SCAN_FOLDER"] + path
    paths.append(new_path)
    scan_configs[new_path] = get_configs(
        new_path,
        scan_configs[new_path] if new_path in scan_configs else {},
        param,
    )

print("Saving scan configs...")
initialize_scan_configs(scan_configs)


model = load_model("model.h5")


outputs = []
for image_path in paths:
    scan_nm = scan_configs[image_path]["scan_nm"]
    contrast = scan_configs[image_path]["contrast"]
    rotation = scan_configs[image_path]["rotation"]

    if image_path.endswith(".Z_mtrx"):
        direction = scan_configs[image_path]["direction"]

        img = matrix_to_img_array(image_path, direction)
    else:
        img = cv2.imread(image_path)

    tip_data = detect_tip(
        img,
        scan_nm=scan_nm,
        roi_nm_size=config["ROI_NM_SIZE"],
        model=model,
        cross_size=config["DETECTOR_CROSS_SIZE"],
        contrast=contrast,
        rotation=rotation,
        scan_debug=config["DETECTOR_SCAN_DEBUG"],
        roi_debug=config["DETECTOR_ROI_DEBUG"],
    )
    output_data = {
        "scan": image_path,
        "sharp": tip_data["sharp"],
        "dull": tip_data["dull"],
        "total": tip_data["total"],
    }
    for key, value in scan_configs[image_path].items():
        output_data[key] = value

    outputs.append(output_data)

export_csv = {
    "scan": [],
    "percentage": [],
    "sharp": [],
    "dull": [],
    "total": [],
}
for key in SCAN_CONFIG_PARAMETERS:
    export_csv[key] = []

for i, output in enumerate(outputs):
    percentage = output["sharp"] / (output["total"]) * 100

    # Add data to export_csv
    for key, value in output.items():
        export_csv[key].append(value)
    for key, value in scan_configs[output["scan"]].items():
        export_csv[key].append(value)
    export_csv["percentage"].append(percentage)

    # Print results
    print()
    print(config["SCANS"][i])
    print(f"Sharp: {output['sharp']} Dull: {output['dull']} Total: {output['total']}")
    print(f"Sharp Percentage: {percentage:.2f}%")

    for key, value in output.items():
        export_csv[key].append(value)

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(export_csv.keys())
    writer.writerows(zip(*export_csv.values()))
