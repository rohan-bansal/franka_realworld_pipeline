import h5py
from copy import  deepcopy
import argparse

def make_smaller(demo_fn, out_fn=None, limit=500):
    #
    
    if out_fn is None:
        out_fn = demo_fn.replace(".hdf5", f"_{limit}_demos.hdf5")
        print(f"Saving smaller dataset to file: {out_fn}")


    out_file = h5py.File(out_fn, "w")
    out_file_grp = out_file.create_group("data")

    demo_file = h5py.File(demo_fn, 'r')
    demo_grp = demo_file['data']

    out_file_grp.attrs["env_args"] = deepcopy(demo_grp.attrs["env_args"])
    out_file_grp.attrs["total"] = 0

    demos = demo_file['data']
    counter = 0
    for demo_name in demos:
        if demo_name == "mask":
            continue
        if counter >= limit:
            break
        if (counter % 40) == 0:
            print(f"Processed demo: {counter}")
        out_file_grp.copy(demo_grp[demo_name], f"demo_{counter}")
        out_file_grp.attrs["total"] += demo_grp[demo_name].attrs["num_samples"]

        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=None,
        help="path to dataset",
    )

    parser.add_argument(
        "--out_fn",
        type=str,
        required=False,
        default=None,
        help="path to dataset",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=500,
        help="path to dataset",
    )

    args = parser.parse_args()



    make_smaller(args.dataset, args.out_fn, args.limit)



