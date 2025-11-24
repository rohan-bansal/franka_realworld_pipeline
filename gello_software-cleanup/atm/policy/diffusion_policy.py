import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from einops import rearrange, repeat

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.policy_head import *
from atm.policy.dp_modules.blocks import get_resnet, replace_bn_with_gn, replace_submodules, ConditionalUnet1D
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

from copy import deepcopy


EXTRA_DIMS = {
    "joint_states": 7,
    "gripper_states": 2,
    "ee_states": 3,
    "robot0_joint_pos": 7,
    "robot0_gripper_qpos": 2,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "ee_pos": 3,
    "ee_ori": 4
}


def reshape_transform(tensor, h, w):
    B, _, E = tensor.shape
    result = tensor[:, 1 : 1 + h * w, :].reshape(B, h, w, E)
    return result.permute(0, 3, 1, 2)



class ATMDiffusionPolicy(nn.Module):
    def __init__(self, obs_cfg, track_cfg, load_path=None):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder()

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 8. define policy head
        self._setup_policy_head()

        self._setup_schedulers()

        self.nets = nn.ModuleDict({
            "image_encoders": self.image_encoders,
            "noise_pred_net": self.noise_pred_net
        })

        if load_path is not None:
            self.load(load_path)
            self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")

    def setup_ema(self):
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            power=0.75)

        # save as a deep copy
        self.ema_nets = deepcopy(self.nets)

        return self.ema

    def _process_obs_shapes(self, obs_shapes, num_views, T_obs, T_act, out_dim, extra_states, img_mean, img_std, max_seq_len, horizon, use_lang=False):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.use_lang = use_lang
        self.T_obs = T_obs
        self.T_act = T_act # T_p in diffusion policy paper
        self.out_dim = out_dim
        self.max_seq_len = max_seq_len
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)
        self.action_queue = deque(maxlen=horizon)
        self.horizon = horizon # T_a in diffusion policy paper

        assert self.T_obs < self.T_act, "T_obs should be less than T_act"
        assert self.horizon + self.T_obs < self.T_act, "action horizon should be less than T_act"

        print(f"""
              shapes: {obs_shapes}
                num_views: {num_views}
                T_obs: {T_obs}
                T_act: {T_act}
                horizon: {horizon}
                out_dim: {out_dim}
                extra_states: {extra_states}
                lang: {use_lang}
              """)

    def _setup_image_encoder(self):
        self.image_encoders = []
        for _ in range(self.num_views):
            vision_encoder = get_resnet('resnet18')
            vision_encoder = replace_bn_with_gn(vision_encoder)
            self.image_encoders.append(vision_encoder)
        self.image_encoders = nn.ModuleList(self.image_encoders)
        self.vision_feature_dim = 512 * self.num_views

    def _setup_track(self, track_fn, use_zero_track=False):
        """
        track_fn: path to the track model
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track

        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = eval(track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)
        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts

    def _setup_policy_head(self):
        self.act_shape = (self.T_act, self.out_dim)
        self.out_shape = np.prod(self.act_shape)
        cond_dim = self.vision_feature_dim
        if self.use_lang:
            cond_dim += 768
        for key in self.extra_state_keys:
            cond_dim += EXTRA_DIMS[key]

        cond_dim *= self.T_obs

        cond_dim += self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

        print(f"cond_dim: {cond_dim}")
        print(f"input_dim: {self.out_shape}")
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.out_dim,
            global_cond_dim=cond_dim
        )

    def _setup_schedulers(self):

        self.nets = nn.ModuleDict({
            "image_encoders": self.image_encoders,
            "noise_pred_net": self.noise_pred_net
        })

        self.ema_nets = self.nets

        self.num_diffusion_iters = 100
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.)
        return rgb

    def track_encode(self, track_obs, task_emb):
        """
        Args:
            track_obs: b v t tt_fs c h w
            task_emb: b e
        Returns: b v t track_len n 2
        """
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
            grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        return _recon_tr

    def forward(self, obs, track_obs, track, task_emb, extra_states, noised_action, diffusion_iter):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            unnormalized_obs: b v t c h w
            track: b v t track_len n 2
            extra_states: {k: b t e}
            sample: noised action b t a
            task_emb: b emb_size
        """
        # vision encoder
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                TensorUtils.time_distributed(
                    obs[:, view_idx, ...], self.nets['image_encoders'][view_idx]
                ),
            )  # (b, t, c)
        img_features = torch.cat(img_encoded, -1)  # (b, t, vc)

        _recon_track = self.track_encode(track_obs, task_emb)  # _recon_track: (b, v, track_len, n, 2)
        _recon_track = rearrange(_recon_track, "b v t tl n d -> b (t v tl n d)")

        if self.use_lang and task_emb is not None:
            if len(task_emb.shape) == 2:
                task_emb = repeat(task_emb, "b e -> b t e", t=img_features.shape[1])
            feat = torch.cat([img_features, task_emb], dim=-1)
        else:
            feat = img_features

        for key in self.extra_state_keys:
            if key in extra_states:
                feat = torch.cat([feat, extra_states[key]], dim=-1)

        feat = rearrange(feat, "b t e -> b (t e)")
        feat = torch.cat([feat, _recon_track], dim=-1)
        noise = self.nets["noise_pred_net"](noised_action, diffusion_iter, feat)

        return noise

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            task_emb: b emb_size
            action: b t act_dim
        """
        B = action.shape[0]
        obs = self._preprocess_rgb(obs)

        # add gaussian noise to action
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.num_diffusion_iters, (B,), device=obs.device)
        noised_action = self.ddpm_scheduler.add_noise(action, noise, timesteps)

        pred_noise = self.forward(obs, track_obs, track, task_emb, extra_states, noised_action, timesteps)
        loss = F.mse_loss(pred_noise, noise)

        ret_dict = {
            "noise pred loss": loss.item()
        }

        return loss, ret_dict

    def diffuse_action(self, cond):
        B = cond.shape[0]
        naction = torch.randn(B, self.T_act, self.out_dim, device=cond.device, dtype=cond.dtype)

        self.inference_scheduler.set_timesteps(self.num_diffusion_iters // 10)

        for k in self.inference_scheduler.timesteps:
                # predict noise
                noise_pred = self.ema_nets["noise_pred_net"](
                    sample=naction,
                    timestep=k,
                    global_cond=cond
                )

                # inverse diffusion step (remove noise)
                naction = self.inference_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        return naction

    def forward_act(self, obs, track_obs, task_emb, extra_states):
        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            # vision encoder
            img_encoded = []
            for view_idx in range(self.num_views):
                img_encoded.append(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.ema_nets["image_encoders"][view_idx]
                    ),
                )

            _recon_track = self.track_encode(track_obs, task_emb)  # _recon_track: (b, v, track_len, n, 2)
            _recon_track = rearrange(_recon_track, "b v t tl n d -> b (t v tl n d)")

            img_features = torch.cat(img_encoded, -1)
            if self.use_lang and task_emb is not None:
                if len(task_emb.shape) == 2:
                    task_emb = repeat(task_emb, "b e -> b t e", t=img_features.shape[1])
                feat = torch.cat([img_features, task_emb], dim=-1)
            else:
                feat = img_features
            for key in self.extra_state_keys:
                if key in extra_states:
                    feat = torch.cat([feat, extra_states[key]], dim=-1)

            feat = rearrange(feat, "b t e -> b (t e)")

            feat = torch.cat([feat, _recon_track], dim=-1)
            naction = self.diffuse_action(feat)

        return naction

    def act(self, obs, task_emb, extra_states):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        self.eval()
        B = obs.shape[0]

        if len(obs.shape) == 4:  # expand batch dimension
            obs = rearrange(obs, "v h w c -> 1 v h w c").copy()
            extra_states = {k: rearrange(v, "e -> 1 e") for k, v in extra_states.items()}

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

        while len(self.track_obs_queue) < self.max_seq_len:
            self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            # vision encoder
            img_encoded = []
            for view_idx in range(self.num_views):
                img_encoded.append(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.ema_nets["image_encoders"][view_idx]
                    ),
                )

            _recon_track = self.track_encode(track_obs, task_emb)  # _recon_track: (b, v, track_len, n, 2)
            _recon_track = rearrange(_recon_track, "b v t tl n d -> b (t v tl n d)")

            img_features = torch.cat(img_encoded, -1)
            if self.use_lang and task_emb is not None:
                if len(task_emb.shape) == 2:
                    task_emb = repeat(task_emb, "b e -> b t e", t=img_features.shape[1])
                feat = torch.cat([img_features, task_emb], dim=-1)
            else:
                feat = img_features
            for key in self.extra_state_keys:
                if key in extra_states:
                    feat = torch.cat([feat, extra_states[key]], dim=-1)

            # note that t = 1. add this to the latent queue
            self.latent_queue.append(feat)

            while len(self.latent_queue) < self.T_obs:
                self.latent_queue.append(feat)

        if len(self.action_queue) == 0:
            feat = torch.cat(list(self.latent_queue), dim=1)
            feat = rearrange(feat, "b t e -> b (t e)")

            feat = torch.cat([feat, _recon_track], dim=-1)
            naction = self.diffuse_action(feat)

            s = self.T_obs - 1  # 1
            e = s + self.horizon  # 5
            for i in range(s, e):
                self.action_queue.append(naction[:, i, :].detach().cpu().numpy())

        # act = self.action_queue.popleft()

        # if act.shape[0] == 1:
        #     act = act[0]  # HACK: this is to deal with robomimic batch size 1. fix for libero

        # act = np.clip(act, -1, 1)

        full_horizon = list(self.action_queue)
        for i in range(len(full_horizon)):
            print(full_horizon[i].shape)
            if full_horizon[i].shape[0] == 1:
                full_horizon[i] = full_horizon[i][0]

            # full_horizon[i] = np.clip(full_horizon[i], -1, 1)
            self.action_queue.popleft()

        return full_horizon, None

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()
        self.action_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()
        self._update_ema()

    def _update_ema(self):
        self.ema.copy_to(self.ema_nets.parameters())
