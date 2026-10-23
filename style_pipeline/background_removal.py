"""Background removal for input photos, using rembg's general-purpose
"u2net_human_seg" model.

Follows the same rembg usage pattern as ~/anime-face-alignment's
util/anime_face_alignment.py (its anime-mode background removal, rembg
isnet-anime): lazy session init, NUMBA_DISABLE_JIT=1, alpha-composite onto a
solid bg_color.

This replaces an earlier BiSeNet (CelebAMask-HQ) port of
util/ffhq_face_alignment.py's remove_background() -- BiSeNet is trained on
tightly-cropped, face-centered portraits, and produced large misclassified
regions (background mistaken for clothing/hair/skin) on this pipeline's
actual input: multi-camera rig photos where the subject is a smaller part of
a wider, more distant shot. u2net_human_seg is a general human-segmentation
model with no such framing assumption.

Only background removal is done here, not FFHQ-style alignment/cropping/
resizing: this pipeline's input photos are multi-view captures for 3DGS/COLMAP
reconstruction, so camera geometry (image size, framing) must be preserved --
cropping or resizing per-image would break the shared camera poses COLMAP/3DGS
rely on. remove_background() below only repaints background pixels (via
alpha compositing), it never crops or resizes.
"""

import os

# rembg's pymatting dependency has numba-JIT-compiled functions; initializing a
# session in the same process as other multithreaded torch/CPU work has been
# observed (in the reference repo this was ported from) to hit a low-probability
# numba internal race (non-thread-safe TargetConfig cache), occasionally
# segfaulting. Disabling JIT falls back to pure Python/numpy for that path --
# measured as net faster there (no compile overhead) and side-effect-free, so
# always disabled rather than only working around it reactively. Must be set
# before rembg is imported.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_bg_session = None


def _get_bg_session():
    """Lazily initializes the rembg session (so callers that never remove
    backgrounds aren't forced to import/load rembg)."""
    global _bg_session
    if _bg_session is None:
        from rembg import new_session
        _bg_session = new_session("u2net_human_seg")
    return _bg_session


def remove_background(image_bgr, bg_color=(255, 255, 255)):
    """Extracts the foreground (person) with rembg(u2net_human_seg) and
    repaints the background with bg_color. bg_color is a BGR tuple, matching
    image_bgr's channel order. Composites via the predicted alpha rather than
    a hard binary mask, so soft/semi-transparent regions (hair strands, etc.)
    blend naturally into the fill color instead of leaving a hard edge.
    """
    from rembg import remove

    rgba = remove(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), session=_get_bg_session())
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    composed = fg_bgr * alpha + bg * (1.0 - alpha)
    return np.clip(composed, 0, 255).astype(np.uint8)


def remove_background_dir(input_dir, out_dir, bg_color=(255, 255, 255)):
    """Runs remove_background() on every image in input_dir, writing same-named
    outputs into out_dir (created fresh). Returns out_dir.
    """
    image_names = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_names:
        raise SystemExit(f"No images found in {input_dir}")

    os.makedirs(out_dir, exist_ok=True)
    for name in tqdm(image_names, desc="Removing backgrounds"):
        image_bgr = cv2.imread(os.path.join(input_dir, name))
        if image_bgr is None:
            raise SystemExit(f"Failed to read image: {os.path.join(input_dir, name)}")
        out = remove_background(image_bgr, bg_color=bg_color)
        cv2.imwrite(os.path.join(out_dir, name), out)

    return out_dir
