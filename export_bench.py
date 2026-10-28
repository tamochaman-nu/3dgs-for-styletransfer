"""Export a style_transfer.py output directory to the avatar-bench input layout:

    {out}/{exp}/{subject}/cameras/{train,orbit}.json
    {out}/{exp}/{subject}/reference/{renders,depth,alpha}/{split}/{cam}.{png,npy}
    {out}/{exp}/{subject}/{method}/{renders,depth,alpha}/{split}/{cam}.{png,npy}
    {out}/{exp}/{subject}/{method}/targets/train/{cam}.png

reference = the photoreal Gaussians (point_cloud/iteration_*), restricted to the
same face region style_transfer.py --face_only pruned to (recomputed from the
run's COLMAP data); method = the baked point_cloud_stylized. Views: the training cameras (`train`) plus an orbit
around the head spanning the training cameras' azimuth range (`orbit`). Depth is
the alpha-normalized camera-space z (avatar-bench depth_is_alpha_normalized:
true). There is no held-out `test` split (every view was used for training).

Usage (in the style-transfer image):
  docker compose run --rm -v ~/avatar-bench/outputs:/bench_out \\
      -v "$(pwd)/export_bench.py:/workspace/export_bench.py" --entrypoint python3 style-transfer \\
      export_bench.py --model_dir /workspace/output/sample02-styleid1 --out /bench_out \\
      --exp sample02 --subject sample02-styleid1 --method P
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import MiniCam
from style_pipeline.face_region import compute_face_region, prune_gaussians_to_region
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

ZNEAR, ZFAR = 0.01, 100.0
PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)


def bench_camera(cam_id, c2w_rot, center, fx, fy, width, height):
    """avatar-bench camera dict (OpenCV, world->camera R, t). The 3DGS
    rasterizer puts the principal point at the image center ((W-1)/2 in
    pixel-center coordinates)."""
    R = np.asarray(c2w_rot, dtype=np.float64).T
    t = -R @ np.asarray(center, dtype=np.float64)
    K = [[fx, 0.0, (width - 1) / 2], [0.0, fy, (height - 1) / 2], [0.0, 0.0, 1.0]]
    return {"id": cam_id, "K": K, "R": R.tolist(), "t": t.tolist(), "width": width, "height": height}


def minicam(c):
    R_w2c = np.asarray(c["R"])
    fovx = focal2fov(c["K"][0][0], c["width"])
    fovy = focal2fov(c["K"][1][1], c["height"])
    world_view = torch.tensor(getWorld2View2(R_w2c.T, np.asarray(c["t"]))).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(ZNEAR, ZFAR, fovx, fovy).transpose(0, 1).float().cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(c["width"], c["height"], fovy, fovx, ZNEAR, ZFAR, world_view, full)


@torch.no_grad()
def raster_colors(cam, g, colors):
    """Alpha-blended per-Gaussian colors on a black background, unclamped
    (gaussian_renderer.render clamps its output to [0, 1])."""
    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=g.active_sh_degree, campos=cam.camera_center, prefiltered=False,
        debug=False, antialiasing=False,
    )
    img, _, _ = GaussianRasterizer(settings)(
        means3D=g.get_xyz, means2D=torch.zeros_like(g.get_xyz), shs=None, colors_precomp=colors,
        opacities=g.get_opacity, scales=g.get_scaling, rotations=g.get_rotation, cov3D_precomp=None)
    return img


@torch.no_grad()
def render_all(cam_dict, g, bg):
    cam = minicam(cam_dict)
    rgb = render(cam, g, PIPE, bg)["render"]
    ones = torch.ones_like(g.get_xyz)
    alpha = raster_colors(cam, g, ones)[0]
    R = torch.tensor(cam_dict["R"], dtype=torch.float32, device="cuda")
    t = torch.tensor(cam_dict["t"], dtype=torch.float32, device="cuda")
    z = g.get_xyz @ R.T[:, 2] + t[2]
    zmax = float(z.abs().max())
    zsum = raster_colors(cam, g, (z / zmax).unsqueeze(1).repeat(1, 3))[0] * zmax
    depth = torch.where(alpha > 1e-6, zsum / alpha.clamp_min(1e-6), torch.zeros_like(zsum))
    return rgb, depth, alpha


def save_view(root, model, split, cam_id, rgb, depth, alpha):
    for kind in ("renders", "depth", "alpha"):
        os.makedirs(os.path.join(root, model, kind, split), exist_ok=True)
    torchvision.utils.save_image(rgb, os.path.join(root, model, "renders", split, f"{cam_id}.png"))
    np.save(os.path.join(root, model, "depth", split, f"{cam_id}.npy"), depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(root, model, "alpha", split, f"{cam_id}.npy"), alpha.clamp(0, 1).cpu().numpy().astype(np.float32))


def orbit_cameras(train, center, n, span_scale):
    """Cameras on a circle around `center` at the training cameras' mean
    distance and height, covering their azimuth range (times span_scale), all
    looking at `center`. Intrinsics of the first training camera."""
    centers = np.array([-np.asarray(c["R"]).T @ np.asarray(c["t"]) for c in train])
    ups = np.array([-np.asarray(c["R"])[1] for c in train])  # camera -y in world
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    rel = centers - center
    h = float((rel @ up).mean())
    flat = rel - np.outer(rel @ up, up)
    radius = float(np.linalg.norm(flat, axis=1).mean())
    a = flat.mean(0)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    az = np.arctan2(flat @ b, flat @ a)
    lo, hi = az.min(), az.max()
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * span_scale
    ref = train[0]
    cams = []
    for k in range(n):
        th = mid - half + 2 * half * k / (n - 1)
        eye = center + h * up + radius * (np.cos(th) * a + np.sin(th) * b)
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        c2w = np.stack([right, down, fwd], axis=1)
        cams.append(bench_camera(f"orbit_{k:03d}", c2w, eye, ref["K"][0][0], ref["K"][1][1],
                                 ref["width"], ref["height"]))
    return cams, float(np.degrees(2 * half))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True, help="style_transfer.py --output_dir")
    p.add_argument("--out", required=True, help="avatar-bench outputs root")
    p.add_argument("--exp", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--method", default="P")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--n_orbit", type=int, default=48)
    p.add_argument("--orbit_span_scale", type=float, default=1.0,
                   help="orbit azimuth range as a multiple of the training cameras' range")
    p.add_argument("--bg_color", default="1,1,1", help="render background, R,G,B in [0,1]")
    p.add_argument("--face_only", action=argparse.BooleanOptionalAction, default=True,
                   help="restrict the reference to the face region, as style_transfer.py --face_only did; "
                        "the four parameters below must match that run (defaults = its defaults)")
    p.add_argument("--face_conf_threshold", type=float, default=0.5)
    p.add_argument("--face_hit_ratio_threshold", type=float, default=0.5)
    p.add_argument("--face_region_percentile", type=float, default=90.0)
    p.add_argument("--face_region_margin", type=float, default=0.6)
    a = p.parse_args()

    root = os.path.join(a.out, a.exp, a.subject)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "cameras"))
    bg = torch.tensor([float(v) for v in a.bg_color.split(",")], device="cuda")

    with open(os.path.join(a.model_dir, "cameras.json")) as f:
        cj = json.load(f)
    train = [bench_camera(os.path.splitext(c["img_name"])[0], c["rotation"], c["position"],
                          c["fx"], c["fy"], c["width"], c["height"]) for c in cj]

    stylized = GaussianModel(a.sh_degree)
    stylized.load_ply(os.path.join(a.model_dir, "point_cloud_stylized", "point_cloud.ply"))
    iters = glob.glob(os.path.join(a.model_dir, "point_cloud", "iteration_*"))
    last = max(iters, key=lambda d: int(re.search(r"iteration_(\d+)", d).group(1)))
    reference = GaussianModel(a.sh_degree)
    reference.load_ply(os.path.join(last, "point_cloud.ply"))
    n_ref = reference.get_xyz.shape[0]
    if a.face_only:
        # Same region style_transfer.py --face_only pruned to (deterministic for
        # the same COLMAP data and parameters).
        colmap_dir = os.path.join(a.model_dir, "colmap")
        c, r = compute_face_region(os.path.join(colmap_dir, "sparse", "0"), os.path.join(colmap_dir, "images"),
                                   conf_threshold=a.face_conf_threshold, hit_ratio_threshold=a.face_hit_ratio_threshold,
                                   margin_ratio=a.face_region_margin, distance_percentile=a.face_region_percentile)
        prune_gaussians_to_region(reference, c, r)
    print(f"[export] reference {last}: {n_ref} -> {reference.get_xyz.shape[0]} Gaussians; "
          f"stylized: {stylized.get_xyz.shape[0]}")

    center = np.median(stylized.get_xyz.detach().cpu().numpy(), axis=0)
    orbit, span = orbit_cameras(train, center, a.n_orbit, a.orbit_span_scale)
    print(f"[export] {len(train)} train cameras, {len(orbit)} orbit cameras over {span:.1f} deg")
    for split, cams in (("train", train), ("orbit", orbit)):
        with open(os.path.join(root, "cameras", f"{split}.json"), "w") as f:
            json.dump({"cams": cams, "order": [c["id"] for c in cams]}, f, indent=1)
        for c in cams:
            for model_name, g in (("reference", reference), (a.method, stylized)):
                save_view(root, model_name, split, c["id"], *render_all(c, g, bg))

    tdir = os.path.join(root, a.method, "targets", "train")
    os.makedirs(tdir)
    n_t = 0
    for c in cj:
        src = os.path.join(a.model_dir, "stylized", f"{c['img_name']}.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tdir, f"{os.path.splitext(c['img_name'])[0]}.png"))
            n_t += 1
    print(f"[export] {n_t} train targets copied; wrote {root}")


if __name__ == "__main__":
    main()
