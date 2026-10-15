"""Bridge between the trained Gaussians' rendered views and cycle-diffusion's
unpaired DDPM/DDIM domain translation (model/gan_wrapper/ddpm_ddim_wrapper.py).

cycle-diffusion is vendored as a git submodule at ../cycle-diffusion-for-anime-face.
Its DDPM/DDIM translation needs two separately-trained diffusion checkpoints: one for
the source (content) domain and one for the target (style) domain. Both must be
placed under cycle-diffusion-for-anime-face/ckpts/ddpm/ (weights + the matching
config yaml under cycle-diffusion-for-anime-face/ckpts/ddpm/configs/), following
cycle-diffusion's own README.
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
    from model.gan_wrapper.ddpm_ddim_wrapper import DDPMDDIMWrapper  # noqa: E402
finally:
    sys.path[:] = _saved_sys_path


class DDPMStyleTransfer:
    """Runs cycle-diffusion's unpaired image-to-image DDPM/DDIM translation on
    individual rendered views: encode with the source-domain model, decode with
    the target-domain (style) model.
    """

    def __init__(self, source_model_type, target_model_type, source_model_path=None,
                 target_model_path=None, sample_type="ddim", custom_steps=1000, eta=0.1,
                 refine_steps=100, es_steps=850, t_0=None, device="cuda"):
        self.device = device

        cwd = os.getcwd()
        try:
            # ddpm_ddim_wrapper resolves checkpoint/config paths relative to cwd.
            os.chdir(_CYCLE_DIFFUSION_ROOT)
            self.source = DDPMDDIMWrapper(
                source_model_type=source_model_type, source_model_path=source_model_path,
                sample_type=sample_type, custom_steps=custom_steps, eta=eta,
                refine_steps=refine_steps, es_steps=es_steps, t_0=t_0,
            ).to(device).eval()
            self.target = DDPMDDIMWrapper(
                source_model_type=target_model_type, source_model_path=target_model_path,
                sample_type=sample_type, custom_steps=custom_steps, eta=eta,
                refine_steps=refine_steps, es_steps=es_steps, t_0=t_0,
            ).to(device).eval()
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
