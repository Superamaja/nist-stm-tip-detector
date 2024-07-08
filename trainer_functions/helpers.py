import os


def get_ordered_fnames(img_directory):
    fnames_unordered = []
    for file in os.listdir(img_directory):
        if file.endswith(".png"):
            fnames_unordered.append(os.path.join(img_directory, file))

    # re-order according to the index specified at the end of each file name: *_[index].png
    # (e.g. example_5.png should correspond to an index of 5)
    # this is important because the .csv file assumes this order
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
