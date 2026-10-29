"""Bridge between the trained Gaussians' rendered views and the
cycle-diffusion-for-anime-face submodule's photo -> anime translation.

Every setting comes from a style config yml (see configs/style_transfer/), never
from code defaults or CLI flags. `method` selects the translation:
  - cycle_diffusion: CycleDiffusion encode/decode between a source- and a
    target-domain checkpoint (submodule pipeline/generic_ddpm_wrapper.py).
  - styleid: StyleID self-attention K/V injection with one anime style reference
    image (submodule pipeline/styleid.py, same flow as translate_styleid.py).
    Requires an MP U-Net (arch: "mp") config.
Checkpoint/config paths are relative to the submodule root; style_image is
relative to this repo's root (or absolute).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from style_pipeline.face_region import detect_face_bbox

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CYCLE_DIFFUSION_ROOT = os.path.join(_REPO_ROOT, "cycle-diffusion-for-anime-face")

# cycle-diffusion has its own top-level "utils" package, which would otherwise
# shadow this repo's utils/ package for every later `import utils.xxx` in the
# process. Only keep cycle-diffusion on sys.path for the imports that need it.
_saved_sys_path = list(sys.path)
sys.path.insert(0, _CYCLE_DIFFUSION_ROOT)
try:
    from model.gan_wrapper.ddpm_ddim_wrapper import DDPMDDIMWrapper  # noqa: E402
    from pipeline import styleid  # noqa: E402
    from pipeline.generic_ddpm_wrapper import GenericDDPMWrapper  # noqa: E402
finally:
    sys.path[:] = _saved_sys_path


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class DDPMStyleTransfer:
    """Translates the face region of individual rendered views photo -> anime
    with the method selected by the style config."""

    def __init__(self, config_path, device="cuda"):
        self.device = device
        cfg = _load_yaml(config_path)
        self.method = cfg["method"]
        self.crop_scale = float(cfg["face_crop"]["scale"])
        print(f"[style] config={config_path} method={self.method} face_crop.scale={self.crop_scale}")

        cwd = os.getcwd()
        try:
            # Checkpoint/config paths are resolved relative to the submodule root.
            os.chdir(_CYCLE_DIFFUSION_ROOT)
            if self.method == "cycle_diffusion":
                self._init_cycle_diffusion(cfg)
            elif self.method == "styleid":
                self._init_styleid(cfg)
            else:
                raise ValueError(f"unknown style method {self.method!r} (cycle_diffusion | styleid)")
        finally:
            os.chdir(cwd)

    def _init_cycle_diffusion(self, cfg):
        model_cfg = _load_yaml(cfg["model_config"])
        sampling = cfg["sampling"]
        arch_kwargs = {f"arch_{k}": v for k, v in model_cfg["model"].items()}
        diffusion = model_cfg["diffusion"]
        data = model_cfg["data"]
        print(f"[style] model_config={cfg['model_config']} model={model_cfg['model']} "
              f"diffusion={diffusion} sampling={sampling}")

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

        self.source = build(cfg["source_model_path"])
        self.target = build(cfg["target_model_path"])
        assert self.source.resolution == self.target.resolution, (
            "source and target DDPM models must share the same native resolution"
        )
        self.resolution = self.source.resolution

    def _init_styleid(self, cfg):
        s = cfg["styleid"]
        num_steps = s["num_steps"]
        print(f"[style] model_type={cfg['model_type']} target={cfg['target_model_path']} "
              f"source={cfg['source_model_path']} style_image={cfg['style_image']} styleid={s}")

        def load(model_path):
            # eta/sample_type only satisfy DDPMDDIMWrapper's own asserts; only its
            # .generator and noise schedule are used (as in translate_styleid.py).
            wrapper = DDPMDDIMWrapper(
                source_model_type=cfg["model_type"], source_model_path=model_path,
                sample_type="ddim", custom_steps=num_steps, es_steps=num_steps, eta=0.1,
            ).to(self.device).eval()
            if not hasattr(wrapper.generator, "dec"):
                raise ValueError(f"{cfg['model_type']} is not an MP U-Net (arch: \"mp\") config; "
                                 "StyleID injection needs one.")
            return wrapper

        self.target = load(cfg["target_model_path"])
        if s["content_inversion"] == "target":
            self.source = self.target
        elif s["content_inversion"] == "source":
            self.source = load(cfg["source_model_path"])
            assert torch.equal(self.source.betas, self.target.betas), "source/target noise schedules differ"
        else:
            raise ValueError(f"styleid.content_inversion must be 'target' or 'source', got {s['content_inversion']!r}")
        self.resolution = self.target.resolution
        self.adain = s["adain"]
        self.consts = dict(betas=self.target.betas, logvar=self.target.logvar, learn_sigma=self.target.learn_sigma)
        self.injector = styleid.StyleIDInjector(self.target.generator, s["layers"], gamma=s["gamma"], tau=s["tau"])
        self.seq = styleid.timestep_sequence(s["t_0"], num_steps)
        print(f"[style] styleid layers={self.injector.layer_names}")

        # One shared style for every view (keeps the views as consistent as
        # possible), so its K/V are recorded once here.
        style_path = cfg["style_image"]
        if not os.path.isabs(style_path):
            style_path = os.path.join(_REPO_ROOT, style_path)
        style01 = torch.from_numpy(
            np.asarray(Image.open(style_path).convert("RGB")).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        style = F.interpolate(style01.unsqueeze(0).to(self.device), size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False)
        self.injector.style = {}
        self.x_T_style = styleid.invert_and_record(
            self.target.generator, self.target.generator, self.injector, "record_style",
            (style - 0.5) * 2.0, self.seq, **self.consts,
        )

    @torch.no_grad()
    def translate(self, image: torch.Tensor) -> torch.Tensor:
        """image: (3, R, R) float tensor in [0, 1] at the model's native
        resolution R. Returns the translated (3, R, R) tensor in [0, 1]."""
        x = image.unsqueeze(0).to(self.device)
        if self.method == "cycle_diffusion":
            z = self.source.encode(x)
            return torch.clamp(self.target(z=z), 0.0, 1.0).squeeze(0)

        self.injector.content = {}
        x_T = styleid.invert_and_record(
            self.source.generator, self.target.generator, self.injector, "record_content",
            (x - 0.5) * 2.0, self.seq, **self.consts,
        )
        if self.adain:
            x_T = styleid.adain(x_T, self.x_T_style)
        x0 = styleid.generate(self.target.generator, self.injector, x_T, self.seq, inject=True, **self.consts)
        return ((x0 + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(0)

    @torch.no_grad()
    def stylize(self, image: torch.Tensor, bg_color: torch.Tensor, conf_threshold=0.5):
        """image: (3, H, W) float tensor in [0, 1]; bg_color: (3,) RGB in [0, 1].

        The models are trained on square, face-centered crops, so the whole
        (non-square, mostly non-face) render is never fed in directly: a square
        crop of side face_crop.scale * max(face bbox w, h) around the detected
        face is translated at native resolution and pasted back into the render.
        Parts of the crop falling outside the render are filled with bg_color
        (the render's own background color).

        Returns (stylized (3, H, W), crop_before (3, R, R), crop_after (3, R, R),
        box), where box = (x0, y0, x1, y1) is the stylized region's pixel bounds
        within the render, or None if no face is detected in this view.
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
        return out, crop_low, styled_low, (ix0, iy0, ix1, iy1)
