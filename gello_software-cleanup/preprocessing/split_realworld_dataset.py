import argparse
import os
import glob
from natsort import natsorted
import shutil

def split_pretrain_dataset(files, train_folder, val_folder, train_ratio):
    """
    Split dataset files into train and validation sets and create symlinks.
    
    Args:
        files: List of files to split
        train_folder: Path to the train folder
        val_folder: Path to the validation folder
        train_ratio: Ratio of files to use for training
    """
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    train_files = files[:num_train]
    val_files = files[num_train:]
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for f in train_files:
        # Create relative symlinks
        os.system('ln -s {} {}'.format(os.path.relpath(f, train_folder), train_folder))

    for f in val_files:
        # Create relative symlinks
        os.system('ln -s {} {}'.format(os.path.relpath(f, val_folder), val_folder))

def split_bc_train_dataset(root_dir, pretrain_train_folder, file_pattern='*.hdf5', num_trains=None):
    """
    Create behavior cloning training sets of different sizes.
    
    Args:
        root_dir: Root directory where BC training folders will be created
        pretrain_train_folder: Folder containing the pretrain dataset files
        file_pattern: File pattern to match (default: *.hdf5)
        num_trains: List of number of files to use for each BC training set
    """
    if num_trains is None:
        num_trains = [2, 5, 10, 20, 40, 80]
        
    pretrain_train_files = glob.glob(os.path.join(pretrain_train_folder, file_pattern))
    pretrain_train_files = natsorted(pretrain_train_files)
    
    max_files = len(pretrain_train_files)
    valid_num_trains = [n for n in num_trains if n <= max_files]
    
    if len(valid_num_trains) < len(num_trains):
        print(f"Warning: Some num_train values exceed the number of available files ({max_files}).")
        print(f"Using only: {valid_num_trains}")
    
    for num_train in valid_num_trains:
        bc_train_folder = os.path.join(root_dir, f'bc_train_{num_train}')
        os.makedirs(bc_train_folder, exist_ok=True)
        
        bc_train_rest_folder = os.path.join(root_dir, f'bc_train_{num_train}_rest')
        os.makedirs(bc_train_rest_folder, exist_ok=True)
        
        train_files = pretrain_train_files[:num_train]
        unlabel_files = pretrain_train_files[num_train:]
        
        for f in train_files:
            # Create relative symlinks to the original files
            os.system('ln -s {} {}'.format(os.path.relpath(f, bc_train_folder), bc_train_folder))
        
        for f in unlabel_files:
            # Create relative symlinks to the original files
            os.system('ln -s {} {}'.format(os.path.relpath(f, bc_train_rest_folder), bc_train_rest_folder))
        
        print(f"Created BC training set with {num_train} files in {bc_train_folder}")


def process_flat_directory(root_dir, file_pattern, train_ratio, bc_num_trains):
    """Process a flat directory structure with files directly in it"""
    files = glob.glob(os.path.join(root_dir, file_pattern))
    
    if len(files) == 0:
        print(f"Warning: No matching files found in {root_dir}")
        return
        
    files = natsorted(files)
    print(f"Found {len(files)} files in {root_dir}")
    
    train_folder = os.path.join(root_dir, 'train')
    val_folder = os.path.join(root_dir, 'val')
    
    if not os.path.exists(train_folder):
        split_pretrain_dataset(files, train_folder, val_folder, train_ratio)
        
        # Create BC training sets from the train folder
        split_bc_train_dataset(root_dir, train_folder, file_pattern=os.path.basename(file_pattern), 
                             num_trains=bc_num_trains)


def process_nested_directory(root_dir, depth, file_pattern, train_ratio, bc_num_trains):
    """Process a nested directory structure based on specified depth"""
    if depth == 0:
        process_flat_directory(root_dir, file_pattern, train_ratio, bc_num_trains)
        return
        
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            process_nested_directory(item_path, depth-1, file_pattern, train_ratio, bc_num_trains)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split generic dataset into train and validation sets')
    parser.add_argument('--folder', type=str, required=True, 
                        help='Path to the dataset folder')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of data to use for training (default: 0.9)')
    parser.add_argument('--file_pattern', type=str, default='*.hdf5',
                        help='File pattern to match (default: *.hdf5)')
    parser.add_argument('--directory_depth', type=int, default=0,
                        help='Depth of directory structure (0: flat, 1: one level of subdirs, 2: two levels, etc.)')
    parser.add_argument('--bc_num_trains', type=int, nargs='+', default=[2, 5, 10, 20, 40, 80],
                        help='Number of files to use for behavior cloning training sets')
    parser.add_argument('--mode', choices=['libero', 'flat', 'nested'], default='flat',
                        help='Directory structure mode: libero (suite/task), flat, or nested')

    args = parser.parse_args()

    # Make sure the input folder exists
    if not os.path.exists(args.folder):
        raise ValueError(f"Input folder {args.folder} does not exist")

    # Process based on the selected mode
    if args.mode == 'libero':
        print(f"Processing dataset in libero format (suite/task) from {args.folder}")
        # The original libero structure with suite_name/task_name/files
        for suite_name in os.listdir(args.folder):
            suite_path = os.path.join(args.folder, suite_name)
            if os.path.isdir(suite_path):
                for task_name in os.listdir(suite_path):
                    task_path = os.path.join(suite_path, task_name)
                    if os.path.isdir(task_path):
                        print(f"Processing task: {suite_name}/{task_name}")
                        process_flat_directory(task_path, args.file_pattern, args.train_ratio, args.bc_num_trains)
    
    elif args.mode == 'flat':
        print(f"Processing flat directory structure from {args.folder}")
        process_flat_directory(args.folder, args.file_pattern, args.train_ratio, args.bc_num_trains)
    
    elif args.mode == 'nested':
        print(f"Processing nested directory structure (depth={args.directory_depth}) from {args.folder}")
        process_nested_directory(args.folder, args.directory_depth, args.file_pattern, 
                               args.train_ratio, args.bc_num_trains)
    
    print("Dataset splitting completed successfully.") 