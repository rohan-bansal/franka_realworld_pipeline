from robomimic.utils.file_utils import create_hdf5_filter_key
import argparse

def add_masks(hdf5_path):
    num_train_demos = [180]

    for num in num_train_demos:
        train_demo_list = [f"demo_{i}" for i in range(num)]
        key_name = f"train_first_{num}"
        ep_lengths = create_hdf5_filter_key(hdf5_path, train_demo_list, key_name)
        print(f"Train episode lengths for first {num}: {ep_lengths}")

    valid_demo_list = [f"demo_{i}" for i in range(180, 200)]
    valid_demo_key = "valid_last_20"

    ep_lengths = create_hdf5_filter_key(hdf5_path, valid_demo_list, valid_demo_key)

    print(f"Valid episode lengths: {ep_lengths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    args = parser.parse_args()

    add_masks(args.dataset)

