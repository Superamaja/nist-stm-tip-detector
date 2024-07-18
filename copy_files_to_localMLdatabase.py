import os
import shutil

items_to_copy = [
    "detector_functions/",
    "processor_functions/",
    "trainer_functions/",
    "api.py",
    "detector.py",
    "processor.py",
    "trainer.py",
]

# Set the target folder (network drive path)
# Ensure that the network drive is properly mounted and accessible
target_folder = r"Z:/localMLdatabase"


def copy_and_overwrite(src, dst):
    if os.path.isdir(src):
        # If it's a directory, remove the existing directory in the destination
        while os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        # If it's a file, remove the existing file in the destination
        while os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)


def main():
    for item in items_to_copy:
        src_path = item
        dst_path = os.path.join(target_folder, item)

        # Ensure the source path exists
        if not os.path.exists(src_path):
            print(f"Source path does not exist: {src_path}")
            continue

        copy_and_overwrite(src_path, dst_path)
        print(f"Copied and overwrote {src_path} to {dst_path}")


if __name__ == "__main__":
    main()
