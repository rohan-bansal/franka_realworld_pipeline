import h5py
import sys
import os

import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def print_hdf5_tree(name, obj, indent=0):
    prefix = "│   " * indent + "├── "
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}{name.split('/')[-1]}: {obj.shape} [{obj.dtype}]")
        if "actions" in name:
            vals = obj[:]
            print(vals)
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}{name.split('/')[-1]}/")

def walk_hdf5(file, indent=0):
    def visitor(name, obj):
        level = name.count('/')
        print_hdf5_tree(name, obj, indent=level)
    file.visititems(visitor)


def main(hdf5_path):
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return

    print(f"File: {os.path.basename(hdf5_path)}")
    with h5py.File(hdf5_path, "r") as f:
        walk_hdf5(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hdf5_tree.py <path_to_hdf5_file>")
    else:
        main(sys.argv[1])
