"""
Config for Guiding algorithm for Diffusion Policy
"""

from robomimic.config.base_config import BaseConfig

class GuideConfig(BaseConfig):
    ALGO_NAME = "guided_diffusion_policy"
    def __init__(self, dict_to_load=None):
        super().__init__(dict_to_load=dict_to_load)
        self.unlock_keys()
        self.guide_config()
        self.lock_keys()

    def guide_config(self):
        self.guide.enabled        = False # use guiding
        self.guide.timestep_start = 100   # diffusion timestep to start guidance
        self.guide.timestep_end   = 0     # diffusion timestep to end guidance
        self.guide.n_actions_ref  = 4     # diffusion guidance based on `n_actions` of previous actions
        self.guide.ddim_eta       = 1     # (float) stochasticity of noise scheduler, 1 if DDPM, 0 if DDIM
        
        # inpainting
        self.guide.inpainting.enabled    = False # (bool) use inpainting
        self.guide.inpainting.n_resample = 1     # (int) number of resampling (time-travel)
        self.guide.inpainting.blur_inpaint = False
        self.guide.inpainting.blend_inpaint = False
        self.guide.inpainting.blend_inpaint_last = False
        self.guide.inpainting.blend_curve = "linear" # can be 'linear', 'parabolic', 'convex'

        # consistency_loss
        self.guide.consistency_loss.enabled               = False # (bool) use consistency_loss
        self.guide.consistency_loss.n_resample            = 1     # number of resampling (time-travel)
        self.guide.consistency_loss.loss_type             = "mse" # (str) loss type to use
        self.guide.consistency_loss.weight                = 10 # (float) strength of the guidance
        self.guide.consistency_loss.N_sample_monte_carlo  = 1  # (int) number of samples to estimate \nabla_{x_t}\log\prob(y|x_t)
        self.guide.consistency_loss.std_monte_carlo       = 0  # (float) std of samples to estimate \nabla_{x_t}\log\prob(y|x_t)
        
        # for sparc loss
        self.guide.consistency_loss.sparc.window_size     = 4    # (int) number of actions to use for consistency loss
        self.guide.consistency_loss.sparc.amp_th          = 0.05 # (float) amplitude threshold
        self.guide.consistency_loss.sparc.fc              = 10.0 # (float) cut-off frequency
        self.guide.consistency_loss.sparc.paddinglevel    = 0    # (int) padding level
        self.guide.consistency_loss.sparc.action_offset   = 0.0  # (float) action offset
        self.guide.consistency_loss.sparc.action_scale    = 1.0  # (float) action scale

        # for classifier-free guidance
        self.guide.cfg.enabled = False
        self.guide.cfg.weight = 1.0

        # for classifier-based guidance
        self.guide.classifier.enabled = False
        self.guide.classifier.weight  = 1.0
        
