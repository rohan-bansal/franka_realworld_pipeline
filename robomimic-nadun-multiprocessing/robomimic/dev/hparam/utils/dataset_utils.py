import h5py
from collections import OrderedDict

from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
from tqdm import tqdm
import numpy as np


def load_config_from_agent(agent_path, ckpt_dict = None, verbose = False):
    """
    Load the config from the agent path

    Args:
        agent_path (str): path to the agent
        ckpt_dict (dict): loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.
        verbose (bool): if True, include print statements

    Returns:
        config (Config): config object
    """
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=agent_path, ckpt_dict=ckpt_dict)
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose)

    return config

def _check_if_validation_filter_key_exists_in_dataset(validation_filter_key, dataset_path):
    if validation_filter_key is None:
        return False
    with h5py.File(dataset_path, 'r') as f:
        return validation_filter_key in f['mask'].keys()
    
def _check_two_dataset_equality(dataloader1, dataloader2, keys_to_check, upto=100):
    """
    Check if two datasets are the same by comparing the keys

    Args:
        dataloader1 (DataLoader): first dataloader
        dataloader2 (DataLoader): second dataloader
        keys_to_check (list): keys to check
        upto (int): number of data to check
    """
    num_data = len(dataloader1)

    dataloader1_iter = iter(dataloader1)
    dataloader2_iter = iter(dataloader2)
    for i in tqdm(range(num_data)):
        if i >= upto:
            break
        batch1 = next(dataloader1_iter)
        batch2 = next(dataloader2_iter)
        for key in keys_to_check:
            assert (batch1[key]==batch2[key]).all()

    
def postprocess_validation_set(validation_set, action_normalization_stats_for_train):
    """
    The action normalization stats for validation set is different from train set.
    Overwrite the action normalization stats for validation set to be the same as train set.
    """
    for idx, batch in enumerate(validation_set.getitem_cache):
        # construct action dict
        ac_dict = OrderedDict()
        for k in validation_set.action_keys:
            ac = batch[k]
            # expand action shape if needed
            if len(ac.shape) == 1:
                ac = ac.reshape(-1, 1)
            ac_dict[k] = ac
        
        ac_dict = ObsUtils.normalize_dict(ac_dict, normalization_stats=action_normalization_stats_for_train)
        ac_vector = AcUtils.action_dict_to_vector(ac_dict)
        validation_set.getitem_cache[idx]["actions"] = ac_vector

    return validation_set

def squeeze_batch(batch):
    """
    Custom collate function to squeeze batch dimension when batch_size=1
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 1:
        batch = batch[0]
    if isinstance(batch, dict):
        return {k: (v.squeeze(0) if hasattr(v, 'squeeze') else v) for k, v in batch.items()}
    return batch.squeeze(0) if hasattr(batch, 'squeeze') else batch

def get_policy_and_validation_dataloader(agent_path, dataset_path, full_observation=True):
    """
    Load the validation data loader that is not used for agent training from dataset

    Args:
        agent_path (str): path to the agent
        dataset_path (str): path to the dataset

    Returns:
        validation_loader (DataLoader): validation data loader

    Data:
        observation o_{t-1:t}, action a_{t:t+Tp}

    NOTE
        1. Adding all observations
        2. Normalizing actions according to the train set
        3. This assumes that the validation set is sequentially stacked
    """
    # Make assertion really really really sure

    # Load the config that is used for training the `agent_path`
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=agent_path, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    # Using the config, get the validation filter key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key

    assert _check_if_validation_filter_key_exists_in_dataset(valid_filter_by_attribute, dataset_path), \
        "Validation filter key not found in the dataset, \
        Check if you are using the correct dataset for validation"
    assert not config.train.hdf5_normalize_obs, "This function is not supported for normalized observation"

    if full_observation:
        with h5py.File(dataset_path, 'r') as f:
            obs_peek = f['data']["demo_0"]["obs"]
            obs_keys = list(obs_peek.keys())
        
    else:
        obs_keys = config.all_obs_keys # keys that are used for training

    # Validation set is consistent in observation regardless of which observation keys are loaded
    validation_set = TrainUtils.dataset_factory(config, 
                                                obs_keys, 
                                                filter_by_attribute=valid_filter_by_attribute)
    
    action_normalization_stats_for_train = policy.action_normalization_stats    
    validation_set = postprocess_validation_set(validation_set, action_normalization_stats_for_train)

    validation_loader = DataLoader(
        dataset=validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    
    return policy, validation_loader

def split_dataloader_by_demo(dataloader):
    """
    Split the dataloader by demo_id into a dictionary of dataloaders.
    
    Args:
        dataloader (DataLoader): Original dataloader
        
    Returns:
        dict: Dictionary mapping demo_id to corresponding DataLoader
        
    Note:
        Each demo's dataloader maintains the same batch_size and other settings 
        as the original dataloader
    """
    dataset = dataloader.dataset
    demo_dataloaders = {}
    
    # Get cumulative lengths for indexing
    demo_length_vec = list(dataset._demo_id_to_demo_length.values())
    demo_length_cumsum = np.cumsum([0] + demo_length_vec[:-1])  # Start indices for each demo
    
    # Create separate dataloader for each demo
    for demo_idx, demo_id in enumerate(dataset.demos):
        start_idx = demo_length_cumsum[demo_idx]
        end_idx = start_idx + demo_length_vec[demo_idx]
        
        # Create subset for this demo
        from torch.utils.data import Subset
        demo_subset = Subset(dataset, range(start_idx, end_idx))
        
        # Create dataloader with same settings as original
        demo_dataloaders[demo_id] = DataLoader(
            dataset=demo_subset,
            batch_size=dataloader.batch_size,
            shuffle=False,  # Keep sequential order within demo
            num_workers=dataloader.num_workers,
            drop_last=dataloader.drop_last,
        )
    
    return demo_dataloaders

def extract_validation_set_from_dataset(dataset_path, agent_path, save_path=None):
    """
    Extract validation demos from dataset and optionally save as separate HDF5 file.
    
    Args:
        dataset_path (str): Path to original dataset
        agent_path (str): Path to trained agent (needed for validation filter key)
        save_path (str, optional): Path to save extracted validation set. If None, 
                                 only returns the validation data without saving.
    
    Returns:
        dict: Validation dataset containing only validation demos
    """
    # Get validation filter key from agent config
    config_used_for_train = load_config_from_agent(agent_path)
    validation_filter_key = config_used_for_train.train.hdf5_validation_filter_key

    validation_data = {}
    
    # Open original dataset
    with h5py.File(dataset_path, 'r') as f:
        # Get validation demo keys
        validation_demo_keys = list(f['mask'][validation_filter_key][()])
        
        # Copy dataset structure and validation demos
        validation_data['data'] = {}
        validation_data['mask'] = {}
        validation_data['mask'][validation_filter_key] = validation_demo_keys
        
        # Copy each validation demo
        for demo_key in validation_demo_keys:
            validation_data['data'][demo_key] = {}
            
            # Copy all fields for this demo
            demo_group = f['data'][demo_key]
            for field in demo_group.keys():
                try:
                    # Handle different types of datasets
                    if isinstance(demo_group[field], h5py.Dataset):
                        validation_data['data'][demo_key][field] = demo_group[field][()]
                    elif isinstance(demo_group[field], h5py.Group):
                        # For nested groups (like 'next_obs')
                        validation_data['data'][demo_key][field] = {}
                        for subfield in demo_group[field].keys():
                            validation_data['data'][demo_key][field][subfield] = \
                                demo_group[field][subfield][()]
                except Exception as e:
                    print(f"Warning: Could not copy field {field} for demo {demo_key}: {str(e)}")
                    continue

    # Save to new HDF5 file if path provided
    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            # Create groups
            f.create_group('data')
            f.create_group('mask')
            
            # Save mask
            mask_group = f['mask']
            mask_group.create_dataset(validation_filter_key, data=validation_demo_keys)
            
            # Save demo data
            for demo_key in validation_demo_keys:
                demo_group = f['data'].create_group(demo_key)
                
                # Save all fields for this demo
                for field, value in validation_data['data'][demo_key].items():
                    if isinstance(value, dict):
                        # Handle nested groups (like 'next_obs')
                        field_group = demo_group.create_group(field)
                        for subfield, subvalue in value.items():
                            field_group.create_dataset(subfield, data=subvalue)
                    else:
                        demo_group.create_dataset(field, data=value)

    return validation_data

if __name__ == "__main__":
    agent_path = "/home/wjung85/Repo/projects/FastIL/diffusion_trained_models/cfg_square/square_image_action_horizon_16_additional_bl/20250122214609/models/model_epoch_400.pth"
    dataset_path = "/home/wjung85/Repo/projects/FastIL/datasets/square/ph/all_obs_new.hdf5"
    save_path = "/home/wjung85/Repo/projects/FastIL/datasets/square/ph/validation_set.hdf5"
    
    policy, validation_loader = get_policy_and_validation_dataloader(agent_path, dataset_path)

    # Split by demo
    demo_loaders = split_dataloader_by_demo(validation_loader)
    
    # Print info about split
    print("\nDataloader splits:")
    for demo_id, loader in demo_loaders.items():
        print(f"{demo_id}: {len(loader.dataset)} timesteps")