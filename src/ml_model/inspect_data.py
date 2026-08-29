"""
Inspect the structure of the CSKnow sample HDF5 files.

Usage:
    python inspect_data.py                    # inspects the default file below
    python inspect_data.py path/to/file.hdf5  # inspects a specific file

Prints every dataset inside the file with its shape and dtype, plus any
attributes and (where available) column names.
"""

import sys
import h5py

# Default file — edit this path if yours differs
DEFAULT_PATH = r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs\behaviorTreeTeamFeatureStore_28.hdf5"


def show(name, obj):
    """Called for every object in the file; prints details for datasets."""
    if isinstance(obj, h5py.Dataset):
        print(f"{name:<60} shape={str(obj.shape):<20} dtype={obj.dtype}")

        # Compound dtypes carry column names — very useful for tabular data
        if obj.dtype.names:
            print(f"{'':<60} columns: {list(obj.dtype.names)}")

        # Any per-dataset attributes
        for key, val in obj.attrs.items():
            print(f"{'':<60} attr {key} = {val}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    print(f"Opening: {path}\n")

    with h5py.File(path, "r") as f:
        # File-level attributes often describe the schema
        if f.attrs:
            print("File attributes:")
            for key, val in f.attrs.items():
                print(f"  {key} = {val}")
            print()

        print("Datasets:")
        f.visititems(show)


if __name__ == "__main__":
    main()
