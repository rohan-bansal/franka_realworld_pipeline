import json
import os
from glob import glob
import torchvision.transforms as T

import click
import h5py
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from hydra.utils import to_absolute_path

from atm.utils.flow_utils import sample_from_mask, sample_double_grid
from atm.utils.cotracker_utils import Visualizer

from spatracker.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from spatracker.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from spatracker.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from spatracker.SpaTrackV2.models.utils import get_points_on_a_grid
from spatracker.SpaTrackV2.models.predictor import Predictor

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

EXTRA_STATES_KEYS = ['gripper_states', 'joint_states', 'ee_ori', 'ee_pos', 'ee_states']

extrinsics = np.array([[ 0.901696,    0.39898718, -0.16659397,  0.84769182],
                [ 0.38820481, -0.57743289,  0.71823972, -0.47822804],
                [ 0.1903716,  -0.71230646, -0.67555767,  0.47235281],
                [ 0.,          0.,          0.,          1.        ]])
intrinsics = np.array([
        [386.5032958984375, 0.0, 324.4791564941406],
        [0.0, 386.5032958984375, 238.82589721679688],
        [0.0, 0.0, 1.0]
    ])

def get_task_name_from_file_name(file_name):
    name = file_name.replace('_demo', '')
    if name[0].isupper():  # LIBERO-10 and LIBERO-90
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def get_task_embs(cfg, descriptions):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    if cfg.task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./data/bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./data/bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    else:
        raise ValueError("Unsupported task embedding format")
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs


def get_task_bert_embs(libero_root_dir):
    libero_h5_files = glob(os.path.join(libero_root_dir, "*/*.hdf5"))
    task_names = set([get_task_name_from_file_name(os.path.basename(file).split('.')[0]) for file in libero_h5_files])
    task_names = list(task_names)

    if not os.path.exists("libero/task_embedding_caches/task_emb_bert2.npy"):
        # set the task embeddings
        cfg = EasyDict({
            "task_embedding_format": "bert",
            "task_embedding_one_hot_offset": 1,
            "data": {"max_word_len": 25},
            "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
        })  # hardcode the config to get task embeddings according to original Libero code

        task_embs = get_task_embs(cfg, task_names).cpu().numpy()

        task_name_to_emb = {task_names[i]: task_embs[i] for i in range(len(task_names))}

        os.makedirs("libero/task_embedding_caches/", exist_ok=True)
        np.save("libero/task_embedding_caches/task_emb_bert2.npy", task_name_to_emb)
    else:
        task_name_to_emb = np.load("libero/task_embedding_caches/task_emb_bert2.npy", allow_pickle=True).item()
    return task_name_to_emb


def track_through_video(video, track_model, num_points=1000, grid_size=50, depths=None, max_size=256):
    T_, C, H, W = video.shape

    grid_pts = get_points_on_a_grid(grid_size, (H, W), device="cpu")

    video_tensor = torch.from_numpy(video).float()

    depth_tensor = torch.from_numpy(depths).float()
    # depth_tensor = depths
    extrs = np.tile(extrinsics[np.newaxis, ...], (video.shape[0], 1, 1))
    intrs = np.tile(intrinsics[np.newaxis, ...], (video.shape[0], 1, 1))
    unc_metric = None

    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = track_model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=True, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is max_size
        h, w = video.shape[2:]
        
        # Resize to square with side length max_size
        new_h, new_w = max_size, max_size
        video = T.Resize((new_h, new_w))(video)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
        point_map = T.Resize((new_h, new_w))(point_map)
        conf_depth = T.Resize((new_h, new_w))(conf_depth)
        
        # Scale the 2D track coordinates to match the new dimensions
        h_scale, w_scale = new_h / h, new_w / w
        # Convert to PyTorch tensor to avoid mixing numpy and PyTorch
        scale_tensor = torch.tensor([w_scale, h_scale], device=track2d_pred.device, dtype=track2d_pred.dtype)
        track2d_pred[...,:2] = track2d_pred[...,:2] * scale_tensor
        
        # Scale only the focal lengths (fx, fy) in the intrinsics matrix
        # intrs[:,:2,:] has shape [T, 2, 3], we only want to scale the first two elements (fx, fy)
        intrs[:,0,0] = intrs[:,0,0] * w_scale  # fx
        intrs[:,1,1] = intrs[:,1,1] * h_scale  # fy
        
        if depth_tensor is not None:
            if isinstance(depth_tensor, torch.Tensor):
                depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
            else:
                depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))
        

        # Move ALL tensors to CPU to avoid CUDA device type errors
        # This includes ALL tensors returned from the model forward pass
        video = video.cpu()
        video_tensor = video_tensor.cpu()
        point_map = point_map.cpu()
        conf_depth = conf_depth.cpu()
        track2d_pred = track2d_pred.cpu()
        intrs = intrs.cpu()
        vis_pred = vis_pred.cpu()  # This was missing!
        if depth_tensor is not None:
            depth_tensor = depth_tensor.cpu()

    pred_tracks = track2d_pred[..., :2][None, ...] # [1, T, N, 2]
    pred_vis = vis_pred[None] # [1, T, N, 1]

    print(pred_tracks.shape)
    print(pred_vis.shape)

    # Move tensors to CPU to avoid CUDA device type errors
    pred_tracks = pred_tracks.cpu()
    pred_vis = pred_vis.cpu()

    return pred_tracks, pred_vis, video


def collect_states_from_demo(h5_file, image_save_dir, demos_group, demo_k, view_names, track_model, task_emb, num_points, visualizer, save_vis=False, max_size=256):
    actions = np.array(demos_group[demo_k]['actions'])
    print(actions.shape)
    root_grp = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
    if "actions" not in root_grp:
        root_grp.create_dataset("actions", data=actions)

    if "extra_states" not in root_grp:
        extra_states_grp = root_grp.create_group("extra_states")
        for state_key in EXTRA_STATES_KEYS:
            extra_states_grp.create_dataset(state_key, data=np.array(demos_group[demo_k]['obs'][state_key]))

    if "task_emb_bert" not in root_grp:
        root_grp.create_dataset("task_emb_bert", data=task_emb)

    for view in view_names:
        rgb = np.array(demos_group[demo_k]['obs'][f'{view}_rgb'])
        # rgb = rgb[:, ::-1, :, :].copy()  # The images in the raw Libero dataset is upsidedown, so we need to flip it
        rgb = rearrange(rgb, "t h w c -> t c h w")
        rgb = rgb.astype(np.float32)

        print(rgb.shape)

        depths = np.array(demos_group[demo_k]['obs'][f'{view}_depth'])
        depths = np.squeeze(depths)
        depths = depths.astype(np.float32)

        print(depths.shape)

        pred_tracks, pred_vis, resized_video = track_through_video(rgb, track_model, num_points=num_points, depths=depths, max_size=max_size)

        if save_vis:
            # Use the resized video for visualization, not the original rgb
            visualizer.visualize(resized_video[None], pred_tracks, pred_vis, filename=f"{demo_k}_{view}")

        # verify min max of pred_tracks
        print(pred_tracks.min(), pred_tracks.max())

        # [1, T, N, 2], normalize coordinates to [0, 1] for in-picture coordinates
        # Use the new dimensions after resizing
        pred_tracks[:, :, :, 0] /= max_size
        pred_tracks[:, :, :, 1] /= max_size

        # verify min max of pred_tracks
        print(pred_tracks.min(), pred_tracks.max())

        # hierarchically save arrays under the view name
        view_grp = root_grp.create_group(view) if view not in root_grp else root_grp[view]
        if "video" not in view_grp:
            # Convert resized video tensor to numpy and save
            resized_video_np = resized_video.numpy()
            # Convert back to uint8 if needed
            if resized_video_np.dtype == np.float32 and resized_video_np.max() <= 1.0:
                resized_video_np = (resized_video_np * 255).astype(np.uint8)
            view_grp.create_dataset("video", data=resized_video_np[None])

        # we always update the tracks and vis when you run this script
        if "tracks" in view_grp:
            view_grp.__delitem__("tracks")
        if "vis" in view_grp:
            view_grp.__delitem__("vis")
        view_grp.create_dataset("tracks", data=pred_tracks.numpy())
        view_grp.create_dataset("vis", data=pred_vis.numpy()[:, :, :, 0])
        print("saved tracks and vis")

        # save image pngs
        save_images(rearrange(resized_video_np, "t c h w -> t h w c"), image_save_dir, view)
        print("saved images")


def save_images(video, image_dir, view_name):
    os.makedirs(image_dir, exist_ok=True)
    for idx, img in enumerate(video):
        # Only multiply by 255 if the data is actually in 0-1 range
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.dtype == np.float32:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(image_dir, f"{view_name}_{idx}.png"))


def inital_save_h5(path, skip_exist):
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            if ("agentview" in f["root"]) and ("eye_in_hand" in f["root"]) and skip_exist:
                return None

    f = h5py.File(path, 'w')
    return f


def get_attrs_and_view_names(demo_h5):
    """ Get preproception states from h5 file object. """
    attrs = json.loads(demo_h5.attrs['env_args'])
    views = attrs['env_kwargs']['camera_names']
    views.sort()

    views = [name.replace('robot0_', '') if name.endswith("eye_in_hand") else name for name in views]
    return attrs, views


def generate_data(source_h5_path, target_dir, track_model, task_emb, skip_exist):
    demos = h5py.File(source_h5_path, 'r')['data']
    demo_keys = natsorted(list(demos.keys()))
    # attrs, views = get_attrs_and_view_names(demos)
    views = ["agentview"]

    print(demo_keys)

    # save environment meta data
    # with open(os.path.join(target_dir, 'env_meta.json'), 'w') as fp:
    #     json.dump(attrs, fp)

    # setup visualization class
    video_path = os.path.join(target_dir, 'videos')
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)
    visualizer = Visualizer(save_dir=video_path, pad_value=0, fps=24)

    num_points = 1000
    with torch.no_grad():
        for idx in tqdm(range(len(demo_keys))):
            demo_k = demo_keys[idx]
            save_path = os.path.join(target_dir, f"{demo_k}.hdf5")
            h5_file_handle = inital_save_h5(save_path, skip_exist)
            image_save_dir = os.path.join(target_dir, "images", demo_k)

            if h5_file_handle is None:
                continue

            try:
                collect_states_from_demo(h5_file_handle, image_save_dir, demos, demo_k, views, track_model, task_emb, num_points, visualizer, save_vis=(idx%10==0), max_size=256)
                h5_file_handle.close()
                print(f"{save_path} is completed.")
            except Exception as e:
                print(f"Exception {e} when processing {save_path}")
                h5_file_handle.close()
                exit()


@click.command()
@click.option("--root", type=str, default="./data/real")
@click.option("--save", type=str, default="./data/atm_spatrack_256_img/")
@click.option("--suite", type=str, default="pickplace")
@click.option("--skip_exist", type=bool, default=False)
def main(root, save, suite, skip_exist):
    """
    root: str, root directory of original libero dataset
    save: str, target directory to save the preprocessed data
    suite: str, the name of assigned suite, [libero_spatial, libero_object, libero_goal, libero_10, libero_90]
    skip_exist: bool, whether to skip the existing preprocessed h5df file
    """
    suite_dir = os.path.join(root, suite)

    spatracker = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    spatracker.spatrack.track_num = 756
    spatracker.eval()
    spatracker.to("cuda")

    # load task name embeddings
    task_bert_embs_dict = get_task_bert_embs(root)

    for source_h5 in os.listdir(suite_dir):
        source_h5_path = os.path.join(suite_dir, source_h5)
        file_name = source_h5.split('.')[0]
        task_name = get_task_name_from_file_name(file_name)

        print("task_name", task_name)
 
        save_dir = os.path.join(save, suite, file_name)
        os.makedirs(save_dir, exist_ok=True)
        generate_data(source_h5_path, save_dir, spatracker, task_bert_embs_dict[task_name], skip_exist)


if __name__ == "__main__":
    main()
