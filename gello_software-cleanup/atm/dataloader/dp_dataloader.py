import numpy as np
import torch

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_nearest_to_grids


class DPDataset(BaseDataset):
    def __init__(self, track_obs_fs=1, T_act=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs
        self.T_act = T_act

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = index - demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, time_offset + self.frame_stack - 1, num_frames=self.track_obs_fs, backward=True)])
                    )  # 1 tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, time_offset + self.frame_stack - 1, num_frames=self.track_obs_fs, backward=True)])
                    )  # 1 tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, time_offset + self.frame_stack - 1, num_frames=self.track_obs_fs, backward=True)])
                )  # 1 tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            track_start_index = time_offset + self.frame_stack - 1
            all_view_tracks.append(demo["root"][view]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
            all_view_vis.append(demo["root"][view]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v track_len n
        track = track[:, None, ...]  # v 1 track_len n 2
        vi = vi[:, None, ...]  # v 1 track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v 1 tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            track_i, vi_i = sample_tracks_nearest_to_grids(track[i, 0], vi[i, 0], num_samples=self.num_track_ids)
            sample_track.append(track_i[None])
            sample_vi.append(vi_i[None])
        track = torch.stack(sample_track, dim=0)  # v 1 track_len n 2
        vi = torch.stack(sample_vi, dim=0)  # v 1 track_len n

        actions = demo["root"]["actions"][time_offset:time_offset + self.T_act]
        task_embs = demo["root"]["task_emb_bert"]
        extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
                        demo['root']['extra_states'].items()}

        return obs, track_transformer_obs, track, task_embs, actions, extra_states
