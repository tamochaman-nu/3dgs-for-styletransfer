"""Export a face-only COLMAP dataset from a style_transfer.py --face_only run, so
other 3DGS stylization methods (which start from a COLMAP dataset) see the same
region and background as this pipeline.

What carries over:
  - Background removal: already applied before COLMAP when the run used
    --remove_background (colmap/images are the background-removed photos).
  - Face restriction (--face_only): the run pruned the trained Gaussians to a 3D
    sphere around the face (style_pipeline/face_region.py). Here that sphere is
    recomputed from the run's COLMAP data, the pruned photoreal Gaussians' alpha
    is rendered from every COLMAP camera, and each image is whited out (bg color)
    outside that alpha mask. The sparse points are filtered to the same sphere.
Cameras and poses are copied unchanged, so the dataset stays geometrically
identical to the original. The per-view square face crop used for 2D translation
is NOT applied (see docs/face_region_and_background.md).

Output ({out}/):
  images/{name}            masked images (same names/format as colmap/images)
  masks/{stem}.png         the masks used (white = kept)
  sparse/0/cameras.bin     copied
  sparse/0/images.bin      same poses; 2D points of removed 3D points -> -1
  sparse/0/points3D.bin    points inside the face sphere
  sparse/0/points3D.ply    regenerated from the filtered points (3DGS init)
  pretrained/point_cloud/iteration_{N}/point_cloud.ply
                           the face-pruned photoreal Gaussians, for methods
                           that start from a trained 3DGS

Usage (in the style-transfer image):
  docker compose run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \\
      -v "$(pwd)/export_colmap_face.py:/workspace/export_colmap_face.py" \\
      -v "$(pwd)/export_bench.py:/workspace/export_bench.py" \\
      -v "$(pwd)/style_pipeline:/workspace/style_pipeline" \\
      --entrypoint python3 style-transfer export_colmap_face.py \\
      --model_dir /workspace/output/sample02-styleid1 --out /workspace/output/sample02-styleid1_colmap_face
"""
import argparse
import glob
import os
import re
import shutil
import struct

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from export_bench import minicam, raster_colors
from scene import GaussianModel
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary
from scene.dataset_readers import storePly
from style_pipeline.face_region import compute_face_region, prune_gaussians_to_region


def read_images_bin(path):
    """-> list of dicts (id, qvec, tvec, camera_id, name, xys (N,2), point3D_ids (N,)) in file order."""
    out = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id, = struct.unpack("<i", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id, = struct.unpack("<i", f.read(4))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d, = struct.unpack("<Q", f.read(8))
            data = np.frombuffer(f.read(24 * n2d), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            out.append({"id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name.decode(),
                        "xys": np.stack([data["x"], data["y"]], 1), "point3D_ids": data["id"].copy()})
    return out


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i4d3di", im["id"], *im["qvec"], *im["tvec"], im["camera_id"]))
            f.write(im["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", len(im["point3D_ids"])))
            rec = np.empty(len(im["point3D_ids"]), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            rec["x"], rec["y"], rec["id"] = im["xys"][:, 0], im["xys"][:, 1], im["point3D_ids"]
            f.write(rec.tobytes())


def read_points3D_bin(path):
    """-> list of (id, xyz (3,), rgb (3,) uint8, error, track bytes)."""
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid, = struct.unpack("<Q", f.read(8))
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.frombuffer(f.read(3), dtype=np.uint8)
            err, = struct.unpack("<d", f.read(8))
            tlen, = struct.unpack("<Q", f.read(8))
            pts.append((pid, xyz, rgb, err, tlen, f.read(8 * tlen)))
    return pts


def write_points3D_bin(path, pts):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz, rgb, err, tlen, track in pts:
            f.write(struct.pack("<Q3d", pid, *xyz))
            f.write(rgb.tobytes())
            f.write(struct.pack("<dQ", err, tlen))
            f.write(track)


def face_mask(alpha: torch.Tensor, threshold: float, dilate_px: int, feather_sigma: float) -> torch.Tensor:
    """Binary alpha >= threshold, dilated by dilate_px, edges feathered -> (H, W) in [0, 1]."""
    m = (alpha >= threshold).float()[None, None]
    if dilate_px > 0:
        m = F.max_pool2d(m, 2 * dilate_px + 1, stride=1, padding=dilate_px)
    if feather_sigma > 0:
        r = int(np.ceil(3 * feather_sigma))
        x = torch.arange(-r, r + 1, dtype=m.dtype, device=m.device)
        k = torch.exp(-x ** 2 / (2 * feather_sigma ** 2))
        k = k / k.sum()
        m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return m[0, 0].clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True, help="style_transfer.py --output_dir of a --face_only run")
    p.add_argument("--out", required=True, help="new COLMAP dataset directory")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--bg_color", default="255,255,255", help="fill color outside the face region, R,G,B (0-255)")
    p.add_argument("--alpha_threshold", type=float, default=0.5)
    p.add_argument("--dilate_px", type=int, default=4, help="grow the mask so the face edge is not cut")
    p.add_argument("--feather_sigma", type=float, default=1.5)
    p.add_argument("--jpeg_quality", type=int, default=98)
    # Must match the style_transfer.py run (defaults = its defaults).
    p.add_argument("--face_conf_threshold", type=float, default=0.5)
    p.add_argument("--face_hit_ratio_threshold", type=float, default=0.5)
    p.add_argument("--face_region_percentile", type=float, default=90.0)
    p.add_argument("--face_region_margin", type=float, default=0.6)
    a = p.parse_args()

    colmap = os.path.join(a.model_dir, "colmap")
    sparse = os.path.join(colmap, "sparse", "0")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists; remove it first")
    for d in ("images", "masks", "sparse/0"):
        os.makedirs(os.path.join(a.out, d))

    center, radius = compute_face_region(sparse, os.path.join(colmap, "images"),
                                         conf_threshold=a.face_conf_threshold,
                                         hit_ratio_threshold=a.face_hit_ratio_threshold,
                                         margin_ratio=a.face_region_margin,
                                         distance_percentile=a.face_region_percentile)
    iters = glob.glob(os.path.join(a.model_dir, "point_cloud", "iteration_*"))
    last = max(iters, key=lambda d: int(re.search(r"iteration_(\d+)", d).group(1)))
    g = GaussianModel(a.sh_degree)
    g.load_ply(os.path.join(last, "point_cloud.ply"))
    n_all = g.get_xyz.shape[0]
    prune_gaussians_to_region(g, center, radius)
    print(f"[colmap_face] face sphere center={center} radius={radius:.4f}; "
          f"Gaussians {n_all} -> {g.get_xyz.shape[0]}")

    # Sparse points: keep those inside the same sphere.
    pts = read_points3D_bin(os.path.join(sparse, "points3D.bin"))
    keep = [pt for pt in pts if np.linalg.norm(pt[1] - center) <= radius]
    keep_ids = {pt[0] for pt in keep}
    write_points3D_bin(os.path.join(a.out, "sparse", "0", "points3D.bin"), keep)
    storePly(os.path.join(a.out, "sparse", "0", "points3D.ply"),
             np.array([pt[1] for pt in keep]), np.array([pt[2] for pt in keep]))
    shutil.copy(os.path.join(sparse, "cameras.bin"), os.path.join(a.out, "sparse", "0", "cameras.bin"))
    print(f"[colmap_face] sparse points {len(pts)} -> {len(keep)}")

    intr = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    images = read_images_bin(os.path.join(sparse, "images.bin"))
    r, gr, b = (int(v) for v in a.bg_color.split(","))
    bg_bgr = np.array([b, gr, r], dtype=np.float32)  # cv2 images are BGR
    coverage = []
    for im in images:
        cam = intr[im["camera_id"]]
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise SystemExit(f"unsupported camera model {cam.model} (expected undistorted PINHOLE)")
        fx = cam.params[0]
        fy = cam.params[1] if cam.model == "PINHOLE" else cam.params[0]
        R = qvec2rotmat(np.array(im["qvec"]))
        cam_dict = {"R": R.tolist(), "t": list(im["tvec"]), "width": cam.width, "height": cam.height,
                    "K": [[fx, 0.0, (cam.width - 1) / 2], [0.0, fy, (cam.height - 1) / 2], [0.0, 0.0, 1.0]]}
        with torch.no_grad():
            alpha = raster_colors(minicam(cam_dict), g, torch.ones_like(g.get_xyz))[0]
        m = face_mask(alpha, a.alpha_threshold, a.dilate_px, a.feather_sigma).cpu().numpy()
        coverage.append(float((m > 0.5).mean()))

        src = cv2.imread(os.path.join(colmap, "images", im["name"]), cv2.IMREAD_COLOR).astype(np.float32)
        out = src * m[..., None] + bg_bgr * (1.0 - m[..., None])
        params = [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality] if im["name"].lower().endswith((".jpg", ".jpeg")) else []
        cv2.imwrite(os.path.join(a.out, "images", im["name"]), np.clip(out, 0, 255).round().astype(np.uint8), params)
        cv2.imwrite(os.path.join(a.out, "masks", os.path.splitext(im["name"])[0] + ".png"),
                    (m * 255).round().astype(np.uint8))

        ids = im["point3D_ids"]
        im["point3D_ids"] = np.where(np.isin(ids, list(keep_ids)), ids, -1)
    write_images_bin(os.path.join(a.out, "sparse", "0", "images.bin"), images)

    pre = os.path.join(a.out, "pretrained", "point_cloud", os.path.basename(last))
    os.makedirs(pre)
    g.save_ply(os.path.join(pre, "point_cloud.ply"))
    cov = np.array(coverage)
    print(f"[colmap_face] {len(images)} images masked; face mask covers {cov.mean() * 100:.1f}% of the image on "
          f"average (min {cov.min() * 100:.1f}%, {int((cov == 0).sum())} images with no face region); wrote {a.out}")


if __name__ == "__main__":
    main()
