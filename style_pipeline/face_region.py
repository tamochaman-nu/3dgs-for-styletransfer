"""Restricts the reconstructed Gaussians to a 3D region around the face,
instead of the whole person/scene COLMAP registered cameras for.

Per-image 2D cropping was ruled out: cropping each input photo to a
per-frame face bounding box would change the effective field of view/
principal point frame-to-frame, breaking the shared camera geometry COLMAP/
3DGS rely on (see style_transfer.py's --remove_background, which repaints
pixels in place for the same reason). Masking out everything but the face
*before* COLMAP was also ruled out: it would shrink the already-struggling
cross-camera feature matching (see --single_camera) down to face-only
texture, making wide-baseline registration even harder.

Instead: train normally on the full scene (robust COLMAP registration,
full-detail reconstruction), then prune the *finished* Gaussian model down to
a 3D bounding sphere around the face, determined from COLMAP's own sparse
point cloud:
  1. Run a face detector on each COLMAP-registered (undistorted) training
     image, independently, to get a 2D face bounding box per image (or none).
  2. For every COLMAP sparse 3D point, look at its track (every image it was
     observed in + the 2D pixel it projected to there, both already computed
     by COLMAP -- no re-projection/camera-math needed). Score each point by
     what fraction of its observations, restricted to images with a detected
     face, fall inside that image's face bbox.
  3. Keep the 3D points that score highly enough; their centroid + max
     distance (+ margin) becomes the face region.
  4. Prune every Gaussian outside that sphere (see prune_gaussians_to_region
     below for why this doesn't reuse GaussianModel.prune_points).

Face detector: OpenCV's DNN "res10" SSD face detector (Caffe), not the
BiSeNet/rembg models used elsewhere in style_pipeline/ -- those segment
*person* pixels, not localize a face bounding box, which is what step 2
above needs. Auto-downloaded on first use, same pattern as this repo's other
pretrained-weight dependencies.
"""

import os
import struct
import urllib.request

import cv2
import numpy as np
import torch
from torch import nn

_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "face_detector")
_PROTOTXT_PATH = os.path.join(_CACHE_DIR, "deploy.prototxt")
_WEIGHTS_PATH = os.path.join(_CACHE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
_INPUT_SIZE = 300
_BLOB_MEAN = (104.0, 177.0, 123.0)

_net = None


def _get_face_net():
    global _net
    if _net is not None:
        return _net

    os.makedirs(_CACHE_DIR, exist_ok=True)
    if not os.path.exists(_PROTOTXT_PATH):
        urllib.request.urlretrieve(_PROTOTXT_URL, _PROTOTXT_PATH)
    if not os.path.exists(_WEIGHTS_PATH):
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)

    _net = cv2.dnn.readNetFromCaffe(_PROTOTXT_PATH, _WEIGHTS_PATH)
    return _net


def detect_face_bbox(image_bgr, conf_threshold=0.5):
    """Returns the highest-confidence face bbox as (x1, y1, x2, y2) in
    image_bgr's own pixel coordinates, or None if nothing scores above
    conf_threshold."""
    net = _get_face_net()
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (_INPUT_SIZE, _INPUT_SIZE)), 1.0, (_INPUT_SIZE, _INPUT_SIZE), _BLOB_MEAN,
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = conf_threshold
    best_box = None
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > best_conf:
            x1, y1, x2, y2 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_conf = conf
            best_box = (
                max(0, int(round(x1))), max(0, int(round(y1))),
                min(w, int(round(x2))), min(h, int(round(y2))),
            )
    return best_box


def _read_colmap_images_bin(path):
    """Returns {image_name: [(x, y, point3D_id), ...]} -- point3D_id is -1
    for keypoints COLMAP didn't match to any 3D point (excluded by callers)."""
    images = {}
    with open(path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg_images):
            f.read(4)  # image_id
            f.read(32)  # qvec (4 doubles)
            f.read(24)  # tvec (3 doubles)
            f.read(4)  # camera_id
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack("<2d", f.read(16))
                point3D_id = struct.unpack("<q", f.read(8))[0]
                points2D.append((x, y, point3D_id))
            images[name.decode()] = points2D
    return images


def _read_colmap_points3D_bin(path):
    """Returns {point3D_id: (x, y, z)}."""
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # (image_id uint32, point2D_idx uint32) each
            points[point3D_id] = xyz
    return points


def compute_face_region(sparse_dir, images_dir, conf_threshold=0.5, hit_ratio_threshold=0.5,
                         margin_ratio=0.6, distance_percentile=90.0):
    """Determines a 3D bounding sphere around the face from COLMAP's sparse
    reconstruction + per-image face detection (see module docstring).
    Returns (center: np.ndarray shape (3,), radius: float).
    Raises SystemExit if no face was detected in any training image, or no
    sparse point scored well enough to define a region.

    Qualifying sparse points reliably include a handful of extreme outliers
    (e.g. a point mistriangulated from a spurious wide-baseline match between
    two very different camera angles -- see style_transfer.py's
    --single_camera fix for why those matches are already fragile here): on
    a real multi-camera-rig run, the single farthest qualifying point was
    ~35x the median distance from the centroid, while the 90th percentile
    was ~1.6x. Sizing the sphere off the single farthest point (as an
    earlier version of this function did) made the "face" region balloon
    out to engulf the whole scene. center uses the per-axis median (also
    outlier-robust) instead of the mean, and radius uses
    distance_percentile (not max) for the same reason.
    """
    images = _read_colmap_images_bin(os.path.join(sparse_dir, "images.bin"))
    points3D = _read_colmap_points3D_bin(os.path.join(sparse_dir, "points3D.bin"))

    hits = {}
    totals = {}
    any_face_detected = False
    for name, points2D in images.items():
        image_path = os.path.join(images_dir, name)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        bbox = detect_face_bbox(image_bgr, conf_threshold=conf_threshold)
        if bbox is None:
            continue
        any_face_detected = True
        x1, y1, x2, y2 = bbox
        for x, y, point3D_id in points2D:
            if point3D_id < 0:
                continue
            totals[point3D_id] = totals.get(point3D_id, 0) + 1
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits[point3D_id] = hits.get(point3D_id, 0) + 1

    if not any_face_detected:
        raise SystemExit(
            "--face_only: no face was detected in any COLMAP-registered training image "
            f"(looked in {images_dir}). Cannot determine a face region to prune Gaussians to."
        )

    face_points = [
        points3D[pid] for pid, total in totals.items()
        if pid in points3D and (hits.get(pid, 0) / total) >= hit_ratio_threshold
    ]
    if not face_points:
        raise SystemExit(
            "--face_only: faces were detected, but no COLMAP sparse point scored highly enough "
            f"(hit_ratio_threshold={hit_ratio_threshold}) to define a face region. Try lowering "
            "--face_hit_ratio_threshold."
        )

    face_points = np.array(face_points)
    center = np.median(face_points, axis=0)
    dists = np.linalg.norm(face_points - center, axis=1)
    radius = float(np.percentile(dists, distance_percentile))
    radius *= (1.0 + margin_ratio)
    return center, radius


def prune_gaussians_to_region(gaussians, center, radius):
    """Prunes every Gaussian whose position falls outside the given 3D
    sphere. Returns (num_pruned, num_remaining).

    Deliberately does NOT go through GaussianModel.prune_points (used during
    training's own densify/prune cycle): that method also slices
    self.optimizer's per-parameter Adam state (exp_avg/exp_avg_sq), which
    only exists after GaussianModel.training_setup() has been called --
    true right after reconstruct_gaussians(), but NOT when --skip_training
    reloads an already-trained point cloud via load_trained_scene() (no
    optimizer is ever attached there, since nothing further is trained on
    it). This function only needs a static, pruned set of Gaussians for
    rendering (Steps 2-5 never resume training on these position/shape
    parameters -- bake_and_optimize_color() builds its own fresh optimizer
    over _features_dc only), so it slices every per-point tensor directly
    and leaves any optimizer state alone.
    """
    center_t = torch.as_tensor(center, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    dist = torch.linalg.norm(gaussians.get_xyz - center_t, dim=1)
    keep_mask = dist <= radius
    num_total = gaussians.get_xyz.shape[0]
    num_remaining = int(keep_mask.sum().item())
    if num_remaining == 0:
        raise SystemExit(
            "--face_only: the computed face region excludes every Gaussian in the scene. "
            "Try raising --face_region_margin or lowering --face_hit_ratio_threshold."
        )

    with torch.no_grad():
        gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask].requires_grad_(True))
        gaussians._features_dc = nn.Parameter(gaussians._features_dc[keep_mask].requires_grad_(True))
        gaussians._features_rest = nn.Parameter(gaussians._features_rest[keep_mask].requires_grad_(True))
        gaussians._scaling = nn.Parameter(gaussians._scaling[keep_mask].requires_grad_(True))
        gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask].requires_grad_(True))
        gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask].requires_grad_(True))
        if gaussians.max_radii2D.numel() == num_total:
            gaussians.max_radii2D = gaussians.max_radii2D[keep_mask]
        if gaussians.xyz_gradient_accum.numel() == num_total:
            gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[keep_mask]
        if gaussians.denom.numel() == num_total:
            gaussians.denom = gaussians.denom[keep_mask]

    return num_total - num_remaining, num_remaining
