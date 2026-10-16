"""Bridge between the trained Gaussians' rendered views and cycle-diffusion's
unpaired DDPM/DDIM domain translation (pipeline/generic_ddpm_wrapper.py).

cycle-diffusion is vendored as a git submodule at ../cycle-diffusion-for-anime-face.
Its DDPM/DDIM translation needs two separately-trained diffusion checkpoints: one for
the source (content) domain and one for the target (style) domain. Both are expected
to be the "oneres-cosine" checkpoint pair (ckpts/ddpm/ffhq-oneres-cosine-020000.pt /
anime-oneres-cosine-020000.pt in the submodule): a 256-channel UNet with
num_res_blocks=1 and a cosine noise schedule, reverse-engineered from the checkpoint
tensors in cycle-diffusion-for-anime-face/pipeline/step_e12_translate_matched_arch.py's
module docstring. This architecture has no i_DDPM() preset in
model/lib/ddpm_ddim/models/improved_ddpm/script_util.py, so GenericDDPMWrapper
(which builds the UNet directly from explicit arch kwargs) is used instead of
model/gan_wrapper/ddpm_ddim_wrapper.py's DDPMDDIMWrapper.
"""

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CYCLE_DIFFUSION_ROOT = os.path.join(_REPO_ROOT, "cycle-diffusion-for-anime-face")

# cycle-diffusion has its own top-level "utils" package, which would otherwise
# shadow this repo's utils/ package for every later `import utils.xxx` in the
# process. Only keep cycle-diffusion on sys.path for the one import that needs it.
_saved_sys_path = list(sys.path)
sys.path.insert(0, _CYCLE_DIFFUSION_ROOT)
try:
    from pipeline.generic_ddpm_wrapper import GenericDDPMWrapper  # noqa: E402
finally:
    sys.path[:] = _saved_sys_path

# Architecture of the ffhq-oneres-cosine-020000.pt / anime-oneres-cosine-020000.pt
# checkpoint pair: same 256-channel UNet as cycle-diffusion's flagship
# ffhq512_*/anime_aligned_curated_* checkpoints, but with num_res_blocks reduced
# from 2 to 1 (see pipeline/step_e12_translate_matched_arch.py's module docstring
# and its --*_arch_num_res_blocks/--*_beta_schedule override example).
_ONERES_COSINE_ARCH_KWARGS = dict(
    arch_num_channels=256, arch_num_res_blocks=1, arch_attention_resolutions="32,16,8",
    arch_num_heads=4, arch_num_head_channels=-1, arch_channel_mult="",
    arch_dropout=0.0, arch_resblock_updown=False, arch_use_scale_shift_norm=True,
    arch_class_cond=False, arch_learn_sigma=True,
)


class DDPMStyleTransfer:
    """Runs cycle-diffusion's unpaired image-to-image DDPM/DDIM translation on
    individual rendered views: encode with the source-domain model, decode with
    the target-domain (style) model.
    """

    def __init__(self, source_model_path, target_model_path,
                 sample_type="ddim", custom_steps=1000, eta=0.1,
                 refine_steps=100, es_steps=850, t_0=None, device="cuda"):
        self.device = device

        def build(model_path):
            return GenericDDPMWrapper(
                model_path=model_path, image_size=256, channels=3,
                arch_kwargs=_ONERES_COSINE_ARCH_KWARGS,
                beta_schedule="cosine", beta_start=0.0001, beta_end=0.02,
                num_diffusion_timesteps=1000,
                sample_type=sample_type, custom_steps=custom_steps, es_steps=es_steps,
                eta=eta, t_0=t_0, refine_steps=refine_steps,
            ).to(self.device).eval()

        cwd = os.getcwd()
        try:
            # Checkpoint paths (e.g. ckpts/ddpm/ffhq-oneres-cosine-020000.pt) are
            # resolved relative to the submodule root, matching DDPMDDIMWrapper's
            # existing convention.
            os.chdir(_CYCLE_DIFFUSION_ROOT)
            self.source = build(source_model_path)
            self.target = build(target_model_path)
        finally:
            os.chdir(cwd)

        assert self.source.resolution == self.target.resolution, (
            "source and target DDPM models must share the same native resolution"
        )
        self.resolution = self.source.resolution

    @torch.no_grad()
    def stylize(self, image: torch.Tensor) -> torch.Tensor:
        """image: (3, H, W) float tensor in [0, 1].
        Returns a (3, H, W) float tensor in [0, 1], resized back to (H, W).

        The DDPM models operate at a fixed native resolution (typically 256), so the
        image is downsampled for translation and the stylized result is upsampled
        back, without preserving aspect ratio.
        """
        _, h, w = image.shape
        low_res = F.interpolate(
            image.unsqueeze(0).to(self.device), size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        z = self.source.encode(low_res)
        styled = self.target(z=z)
        styled = torch.clamp(styled, 0.0, 1.0)
        styled = F.interpolate(styled, size=(h, w), mode="bilinear", align_corners=False)
        return styled.squeeze(0)
