import os
from typing import List


def get_ordered_fnames(img_directory: str) -> List[str]:
    """Retrieves and orders file names from the specified directory based on the index
    at the end of each file name.

    Parameters:
        img_directory (str): Directory containing the image files.

    Returns:
        list: List of ordered file names.
    """
    fnames_unordered = []
    for file in os.listdir(img_directory):
        if file.endswith(".png"):
            fnames_unordered.append(os.path.join(img_directory, file))

    # Re-order according to the index specified at the end of each file name: *_[index].png
    fname_order = []
    for fname in fnames_unordered:
        end = fname.split("_")[-1]
        num_str = end.split(".")[0]
        fname_order.append(int(num_str))

    # example_fnames will hold the correctly ordered set of file names
    fnames = [None] * len(fnames_unordered)
    old_idx = 0
    for idx in fname_order:
        fnames[idx] = fnames_unordered[old_idx]
        old_idx += 1

    return fnames
