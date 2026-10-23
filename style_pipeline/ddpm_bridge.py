"""Bridge between the trained Gaussians' rendered views and cycle-diffusion's
unpaired DDPM/DDIM domain translation (pipeline/generic_ddpm_wrapper.py).

Every setting comes from a style config yml (see configs/style_transfer/), never
from code defaults or CLI flags:
  - model_config: one of the submodule's ckpts/ddpm/configs/*.yml, supplying the
    UNet architecture (`model:`), native resolution (`data:`) and noise schedule
    (`diffusion:`) shared by both checkpoints.
  - source_model_path / target_model_path: the two checkpoints.
  - sampling: CycleDiffusion encode/decode settings (sample_type, custom_steps,
    es_steps, eta, refine_steps, t_0).
  - face_crop.scale: crop size around the detected face (see stylize()).
All paths in it are relative to the cycle-diffusion-for-anime-face/ submodule.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from style_pipeline.face_region import detect_face_bbox

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


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class DDPMStyleTransfer:
    """Runs cycle-diffusion's unpaired image-to-image DDPM/DDIM translation on
    the face region of individual rendered views: encode with the source-domain
    model, decode with the target-domain (style) model.
    """

    def __init__(self, config_path, device="cuda"):
        self.device = device
        cfg = _load_yaml(config_path)
        model_cfg = _load_yaml(os.path.join(_CYCLE_DIFFUSION_ROOT, cfg["model_config"]))
        sampling = cfg["sampling"]
        self.crop_scale = float(cfg["face_crop"]["scale"])

        arch_kwargs = {f"arch_{k}": v for k, v in model_cfg["model"].items()}
        diffusion = model_cfg["diffusion"]
        data = model_cfg["data"]

        def build(model_path):
            return GenericDDPMWrapper(
                model_path=model_path, image_size=data["image_size"], channels=data["channels"],
                arch_kwargs=arch_kwargs,
                beta_schedule=diffusion["beta_schedule"], beta_start=diffusion["beta_start"],
                beta_end=diffusion["beta_end"], num_diffusion_timesteps=diffusion["num_diffusion_timesteps"],
                sample_type=sampling["sample_type"], custom_steps=sampling["custom_steps"],
                es_steps=sampling["es_steps"], eta=sampling["eta"], t_0=sampling["t_0"],
                refine_steps=sampling["refine_steps"],
            ).to(self.device).eval()

        print(f"[ddpm] config={config_path} model_config={cfg['model_config']}")
        print(f"[ddpm] model={model_cfg['model']} diffusion={diffusion} sampling={sampling} "
              f"face_crop.scale={self.crop_scale}")

        cwd = os.getcwd()
        try:
            os.chdir(_CYCLE_DIFFUSION_ROOT)
            self.source = build(cfg["source_model_path"])
            self.target = build(cfg["target_model_path"])
        finally:
            os.chdir(cwd)

        assert self.source.resolution == self.target.resolution, (
            "source and target DDPM models must share the same native resolution"
        )
        self.resolution = self.source.resolution

    @torch.no_grad()
    def translate(self, image: torch.Tensor) -> torch.Tensor:
        """image: (3, R, R) float tensor in [0, 1] at the model's native
        resolution R. Returns the translated (3, R, R) tensor in [0, 1]."""
        z = self.source.encode(image.unsqueeze(0).to(self.device))
        return torch.clamp(self.target(z=z), 0.0, 1.0).squeeze(0)

    @torch.no_grad()
    def stylize(self, image: torch.Tensor, bg_color: torch.Tensor, conf_threshold=0.5):
        """image: (3, H, W) float tensor in [0, 1]; bg_color: (3,) RGB in [0, 1].

        The models are trained on square, face-centered crops, so the whole
        (non-square, mostly non-face) render is never fed in directly: a square
        crop of side face_crop.scale * max(face bbox w, h) around the detected
        face is translated at native resolution and pasted back into the render.
        Parts of the crop falling outside the render are filled with bg_color
        (the render's own background color).

        Returns (stylized (3, H, W), crop_before (3, R, R), crop_after (3, R, R)),
        or None if no face is detected in this view.
        """
        _, h, w = image.shape
        image_bgr = (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)[:, :, ::-1].copy()
        bbox = detect_face_bbox(image_bgr, conf_threshold=conf_threshold)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        side = int(round(max(x2 - x1, y2 - y1) * self.crop_scale))
        cx0 = int(round((x1 + x2) / 2 - side / 2))
        cy0 = int(round((y1 + y2) / 2 - side / 2))

        # Overlap between the crop square and the image, in both coordinate frames.
        ix0, iy0 = max(cx0, 0), max(cy0, 0)
        ix1, iy1 = min(cx0 + side, w), min(cy0 + side, h)
        px0, py0 = ix0 - cx0, iy0 - cy0
        px1, py1 = px0 + (ix1 - ix0), py0 + (iy1 - iy0)

        image = image.to(self.device)
        bg = bg_color.to(self.device).view(3, 1, 1)
        crop = bg.expand(3, side, side).clone()
        crop[:, py0:py1, px0:px1] = image[:, iy0:iy1, ix0:ix1]

        crop_low = F.interpolate(
            crop.unsqueeze(0), size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False, antialias=True,
        ).squeeze(0)
        styled_low = self.translate(crop_low)
        styled_crop = F.interpolate(
            styled_low.unsqueeze(0), size=(side, side), mode="bilinear", align_corners=False,
        ).squeeze(0)

        out = image.clone()
        out[:, iy0:iy1, ix0:ix1] = styled_crop[:, py0:py1, px0:px1]
        return out, crop_low, styled_low
