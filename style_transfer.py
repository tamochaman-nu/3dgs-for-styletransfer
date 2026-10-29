#
# 3DGS -> cycle-diffusion style transfer pipeline.
#
# 1. Reconstruct Gaussians from the multi-view input images (standard 3DGS training).
# 2. Render the trained Gaussians from the input camera poses.
# 3. Style-transfer each render with cycle-diffusion's unpaired DDPM/DDIM translation.
# 4. Bake the stylized colors onto the Gaussians (freeze geometry/opacity, reset to
#    diffuse-only color).
# 5. Optimize the diffuse color (SH degree-0 term) against the stylized renders.
#
# Input images are read from --input_dir (default /workspace/input, mounted from
# ./input; pass a subdirectory of it, e.g. /workspace/input/some_scene, to use a
# different shoot without remounting), or from --input_video. Everything produced
# (COLMAP data, trained Gaussians, renders, stylized images, the final stylized
# Gaussians) is written under --output_dir (default /workspace/output, mounted
# from ./output).
#

import argparse
import os
import shutil
import subprocess
import sys
from random import randint

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from style_pipeline.background_removal import remove_background_dir
from style_pipeline.ddpm_bridge import DDPMStyleTransfer
from style_pipeline.face_region import compute_face_region, prune_gaussians_to_region
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = "/workspace/input"
DEFAULT_OUTPUT_DIR = "/workspace/output"
DEFAULT_STYLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "style_transfer",
                                    "oneres_cosine.yml")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


VIDEO_ROTATE_FILTERS = {
    0: None,
    90: "transpose=1",              # 90 deg clockwise
    180: "transpose=1,transpose=1",  # 180 deg
    270: "transpose=2",              # 90 deg counter-clockwise (= 270 deg clockwise)
}


def _ffmpeg_extract_frames(video_path: str, out_dir: str, fps: float,
                            start: float, end: float, rotate: int, prefix: str = ""):
    """Run ffmpeg to slice a single video into out_dir/{prefix}frame_%05d.jpg.
    Shared by extract_video_frames (single video) and extract_video_frames_dir
    (one call per video, with prefix set to the video's stem so frames from
    different cameras/files don't collide in the shared out_dir)."""
    vf = f"fps={fps}"
    if VIDEO_ROTATE_FILTERS[rotate]:
        vf += "," + VIDEO_ROTATE_FILTERS[rotate]

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        *(["-ss", str(start)] if start is not None else []),
        *(["-to", str(end)] if end is not None else []),
        "-vf", vf,
        "-qscale:v", "2",
        os.path.join(out_dir, f"{prefix}frame_%05d.jpg"),
    ]
    print("[video] extracting frames:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _subsample_frames(out_dir: str, max_frames: int):
    """Evenly subsample out_dir down to at most max_frames files, deleting the rest."""
    frames = sorted(os.listdir(out_dir))
    if max_frames is not None and len(frames) > max_frames:
        keep_idx = {round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)}
        for i, name in enumerate(frames):
            if i not in keep_idx:
                os.remove(os.path.join(out_dir, name))
        print(f"[video] subsampled down to {max_frames} frames")


def extract_video_frames(video_path: str, out_dir: str, fps: float,
                          start: float, end: float, max_frames: int, rotate: int = 0):
    """Slice a video into frames to use as multi-view input: fps thins a
    continuous video down to a manageable, well-overlapping set of stills;
    start/end trims to a time range; max_frames further caps the count by
    evenly subsampling, in case fps/start/end still yield too many. rotate
    (0/90/180/270, clockwise) corrects footage with no rotation metadata that
    ffmpeg would otherwise extract as-stored (e.g. a phone video held in
    portrait but saved with no auto-rotate flag comes out sideways, not
    corrupted -- just rotated).

    Always re-extracts into a clean out_dir: this is only called when the
    caller wants a fresh dataset (see --skip_colmap in main()), so leftover
    frames from a previous --input_video/--video_* combination must not be
    silently reused."""
    if not os.path.isfile(video_path):
        sys.exit(f"--input_video does not exist: {video_path}")
    if rotate not in VIDEO_ROTATE_FILTERS:
        sys.exit(f"--video_rotate must be one of {sorted(VIDEO_ROTATE_FILTERS)}, got {rotate}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    _ffmpeg_extract_frames(video_path, out_dir, fps, start, end, rotate)
    print(f"[video] extracted {len(os.listdir(out_dir))} frames at {fps} fps")
    _subsample_frames(out_dir, max_frames)

    return out_dir


def extract_video_frames_dir(video_dir: str, out_dir: str, fps: float,
                              start: float, end: float, max_frames: int, rotate: int = 0):
    """Slice every video file in video_dir (e.g. one file per camera in a
    synchronized multi-camera rig) into a single shared out_dir, so all of
    them become multi-view input together. Each video's frames are prefixed
    with its filename stem (e.g. Cam1_frame_00001.jpg, Cam2_frame_00001.jpg)
    to avoid collisions. fps/start/end/rotate apply identically to every video;
    max_frames subsamples the combined set from all videos together (not
    per-video), same semantics as extract_video_frames.

    Always re-extracts into a clean out_dir, for the same reason
    extract_video_frames does."""
    if not os.path.isdir(video_dir):
        sys.exit(f"--input_video does not exist: {video_dir}")
    if rotate not in VIDEO_ROTATE_FILTERS:
        sys.exit(f"--video_rotate must be one of {sorted(VIDEO_ROTATE_FILTERS)}, got {rotate}")
    video_files = sorted(
        f for f in os.listdir(video_dir) if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not video_files:
        sys.exit(f"No video files found in {video_dir}")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for name in video_files:
        stem = os.path.splitext(name)[0]
        _ffmpeg_extract_frames(
            os.path.join(video_dir, name), out_dir, fps, start, end, rotate, prefix=f"{stem}_",
        )
    print(f"[video] extracted {len(os.listdir(out_dir))} frames from {len(video_files)} "
          f"videos ({', '.join(video_files)}) at {fps} fps")
    _subsample_frames(out_dir, max_frames)

    return out_dir


def parse_color(value: str):
    """Parses an 'R,G,B' string (each 0-255) into an OpenCV-order (BGR) tuple."""
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Color must be in 'R,G,B' format (e.g. 255,255,255): {value!r}")
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Color must be in 'R,G,B' format (e.g. 255,255,255): {value!r}")
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise argparse.ArgumentTypeError(f"Each value must be in 0-255: {value!r}")
    return (b, g, r)


def prepare_colmap_dataset(input_dir: str, colmap_dir: str, no_gpu: bool, matcher: str,
                            single_camera: bool = True, camera_per_video: bool = False):
    """Turn the raw multi-view photos in input_dir into a COLMAP dataset (images/ +
    sparse/0) under colmap_dir, in the layout Scene() expects.

    single_camera=False tells COLMAP's feature_extractor to give each image its own
    camera model/intrinsics instead of assuming one shared camera for the whole
    input_dir. Needed whenever input_dir mixes frames from multiple distinct physical
    cameras (see extract_video_frames_dir): with single_camera=True, COLMAP silently
    drops every image whose dimensions don't match the first-registered camera,
    rather than erroring -- e.g. a directory of Cam1.../Cam5... frames where even two
    cameras differ by a few pixels (encoder macroblock-padding) ends up registering
    only the subset that happens to match the first image's size, with no warning.

    camera_per_video=True (with single_camera=False) then shares one camera across
    the frames of each video (grouped by the "{stem}_frame_" prefix from
    extract_video_frames_dir) instead of one per image: per-image intrinsics let
    frames with features only near the image center overfit their distortion, which
    image_undistorter turns into a tiny 0.2x center crop (see convert.py's
    share_camera_per_video).

    Always rebuilds colmap_dir from scratch: this is only called when the caller
    wants a fresh dataset (see --skip_colmap in main()), so a stale sparse/ model
    from a previous --input_dir/--input_video run must not be silently reused."""
    if not os.path.isdir(input_dir):
        sys.exit(f"--input_dir does not exist or is not a directory: {input_dir}")
    raw_images = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not raw_images:
        sys.exit(f"No images found in {input_dir}")

    if os.path.exists(colmap_dir):
        shutil.rmtree(colmap_dir)
    colmap_input_dir = os.path.join(colmap_dir, "input")
    os.makedirs(colmap_input_dir)
    for name in raw_images:
        os.symlink(os.path.join(input_dir, name), os.path.join(colmap_input_dir, name))

    cmd = [
        sys.executable, "convert.py", "-s", colmap_dir, "--matcher", matcher,
        "--single_camera", "1" if single_camera else "0",
    ]
    if camera_per_video:
        cmd.append("--camera_per_video")
    if no_gpu:
        cmd.append("--no_gpu")
    print("[colmap] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def reconstruct_gaussians(dataset, opt, pipe):
    """Step 1: standard photometric 3DGS reconstruction from the input views."""
    from train import training
    return training(
        dataset, opt, pipe,
        testing_iterations=[], saving_iterations=[opt.iterations],
        checkpoint_iterations=[], checkpoint=None, debug_from=-1,
    )


def load_trained_scene(dataset):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    return scene, gaussians


def render_views(scene, gaussians, pipe, background, out_dir):
    """Step 2: render the trained Gaussians from every input camera pose."""
    os.makedirs(out_dir, exist_ok=True)
    rendered = {}
    with torch.no_grad():
        for cam in tqdm(scene.getTrainCameras(), desc="Rendering input views"):
            image = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
            torchvision.utils.save_image(image, os.path.join(out_dir, f"{cam.image_name}.png"))
            rendered[cam.image_name] = image.detach().cpu()
    return rendered


def stylize_views(rendered, style_transfer, background, conf_threshold, out_dir):
    """Step 3: translate the face crop of each render through cycle-diffusion's
    DDPM/DDIM model. Views with no detected face are left out, so the baking
    step only fits against views that were actually stylized.
    Returns (stylized {name: (3, H, W)}, boxes {name: (x0, y0, x1, y1)})."""
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    stylized = {}
    boxes = {}
    skipped = []
    for name, image in tqdm(rendered.items(), desc="Style-transferring views"):
        result = style_transfer.stylize(image, background, conf_threshold=conf_threshold)
        if result is None:
            skipped.append(name)
            continue
        styled, crop_before, crop_after, box = result
        torchvision.utils.save_image(styled, os.path.join(out_dir, f"{name}.png"))
        torchvision.utils.save_image(torch.cat([crop_before, crop_after], dim=2), os.path.join(crops_dir, f"{name}.png"))
        stylized[name] = styled.detach().cpu()
        boxes[name] = box
    if skipped:
        print(f"[stylize] no face detected in {len(skipped)}/{len(rendered)} views, excluded from baking: {skipped}")
    if not stylized:
        raise SystemExit("No face was detected in any rendered view; nothing to stylize.")
    return stylized, boxes


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel_edges(image):
    """(3, H, W) -> (3, H, W) per-channel Sobel gradient magnitude."""
    kx = _SOBEL_X.to(image.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = kx.transpose(2, 3)
    x = torch.nn.functional.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    gx = torch.nn.functional.conv2d(x, kx, groups=3)
    gy = torch.nn.functional.conv2d(x, ky, groups=3)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0)


def bake_and_optimize_color(scene, gaussians, pipe, background, stylized, boxes, opt, args, out_dir):
    """Steps 4-5: reset to diffuse-only color and optimize the stylized look onto
    the Gaussians. Higher-order SH bands are zeroed and frozen (lr 0) since each
    stylized view comes from an independent, multi-view-inconsistent diffusion
    sample -- fitting a single diffuse color per Gaussian keeps the result
    multi-view consistent. Positions stay frozen.

    With --bake_geometry, scaling/rotation/opacity are optimized too and the
    Gaussians are densified (clone/split by view-space gradient, as in train.py),
    so thin Gaussians can form for stroke lines that the photo reconstruction's
    large, smooth Gaussians cannot represent. --bake_lambda_edge adds an L1 loss on
    Sobel edges so those few-pixel lines weigh in against the large flat regions.

    The loss only covers each view's stylized face crop (boxes): outside it the
    target is the unstylized render, which would pull colors back to the photo."""
    with torch.no_grad():
        gaussians._features_rest.zero_()

    geo_scale = args.bake_geometry_lr_scale if args.bake_geometry else 0.0
    groups = [
        (gaussians._xyz, 0.0, "xyz"),
        (gaussians._features_dc, args.style_lr, "f_dc"),
        (gaussians._features_rest, 0.0, "f_rest"),
        (gaussians._opacity, opt.opacity_lr * geo_scale, "opacity"),
        (gaussians._scaling, opt.scaling_lr * geo_scale, "scaling"),
        (gaussians._rotation, opt.rotation_lr * geo_scale, "rotation"),
    ]
    for param, lr, _ in groups:
        param.requires_grad_(lr > 0.0)
    # GaussianModel's densify/prune helpers expect self.optimizer to hold exactly
    # these six named groups, so register it there.
    gaussians.optimizer = torch.optim.Adam(
        [{"params": [p], "lr": lr, "name": name} for p, lr, name in groups], lr=0.0, eps=1e-15,
    )
    num_points = gaussians.get_xyz.shape[0]
    gaussians.percent_dense = opt.percent_dense
    gaussians.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
    gaussians.denom = torch.zeros((num_points, 1), device="cuda")
    gaussians.max_radii2D = torch.zeros((num_points,), device="cuda")

    cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    names = list(stylized.keys())
    densify = args.bake_geometry and args.bake_densify_until > 0
    print(f"[bake] geometry={'on' if args.bake_geometry else 'off'} (lr scale {geo_scale}) "
          f"densify_until={args.bake_densify_until if densify else 0} lambda_edge={args.bake_lambda_edge} "
          f"start Gaussians={num_points}")

    progress = tqdm(range(1, args.style_iterations + 1), desc="Baking style onto Gaussians")
    ema_loss = 0.0
    for it in progress:
        name = names[randint(0, len(names) - 1)]
        cam = cams_by_name[name]
        x0, y0, x1, y1 = boxes[name]
        target = stylized[name][:, y0:y1, x0:x1].contiguous().cuda()

        pkg = render(cam, gaussians, pipe, background)
        image = pkg["render"][:, y0:y1, x0:x1].contiguous()

        Ll1 = l1_loss(image, target)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
        else:
            ssim_value = ssim(image, target)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        if args.bake_lambda_edge > 0.0:
            loss = loss + args.bake_lambda_edge * l1_loss(sobel_edges(image), sobel_edges(target))

        loss.backward()

        with torch.no_grad():
            if densify and it <= args.bake_densify_until:
                visibility_filter = pkg["visibility_filter"]
                radii = pkg["radii"]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(pkg["viewspace_points"], visibility_filter)
                if it % args.bake_densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None, radii)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if it % 10 == 0:
            progress.set_postfix({"loss": f"{ema_loss:.4f}", "n": gaussians.get_xyz.shape[0]})

    print(f"[bake] final Gaussians={gaussians.get_xyz.shape[0]}")

    ply_path = os.path.join(out_dir, "point_cloud_stylized", "point_cloud.ply")
    gaussians.save_ply(ply_path)
    print(f"[bake] stylized Gaussians saved to {ply_path}")

    final_dir = os.path.join(out_dir, "final_renders")
    os.makedirs(final_dir, exist_ok=True)
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            image = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
            torchvision.utils.save_image(image, os.path.join(final_dir, f"{cam.image_name}.png"))


def main():
    parser = argparse.ArgumentParser(description="3DGS -> cycle-diffusion style baking pipeline")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                         help="Directory of raw multi-view photos (flat, no COLMAP data yet). "
                              f"Defaults to {DEFAULT_INPUT_DIR}; pass a subdirectory of it to use "
                              "a different shoot without changing the docker-compose mount, e.g. "
                              f"{DEFAULT_INPUT_DIR}/some_scene. Ignored if --input_video or "
                              "--skip_colmap is set.")
    parser.add_argument("--input_video", type=str, default=None,
                         help="Path to a single video file, OR a directory of video files (e.g. one "
                              "per camera in a synchronized multi-camera rig), to use instead of "
                              "--input_dir. Frames are extracted with ffmpeg (see --video_fps/"
                              "--video_start/--video_end/--video_max_frames) into "
                              "<output_dir>/video_frames and used as the input photos. For a "
                              "directory, every video inside is sliced into the same shared "
                              "video_frames dir (filenames prefixed by each video's name to avoid "
                              "collisions), and --colmap_matcher defaults to 'exhaustive' instead "
                              "of 'sequential' since frames from different cameras aren't a single "
                              "continuous, naturally-ordered sequence.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write everything the pipeline produces (COLMAP data, "
                              f"trained/stylized Gaussians, renders). Defaults to {DEFAULT_OUTPUT_DIR}; "
                              "pass a subdirectory of it (or another mounted path) to keep multiple "
                              "runs side by side, e.g. /workspace/output/some_scene.")
    parser.add_argument("--video_fps", type=float, default=2.0,
                         help="Frames per second to extract from --input_video. Video is usually "
                              "30-60fps, far too dense/redundant for SfM; 1-3 fps is a reasonable range.")
    parser.add_argument("--video_start", type=float, default=None,
                         help="Start time in seconds to slice --input_video from (default: beginning)")
    parser.add_argument("--video_end", type=float, default=None,
                         help="End time in seconds to slice --input_video to (default: end of video)")
    parser.add_argument("--video_max_frames", type=int, default=None,
                         help="Cap the number of extracted frames, evenly subsampling if --video_fps "
                              "(within --video_start/--video_end) would yield more than this.")
    parser.add_argument("--video_rotate", type=int, default=0, choices=[0, 90, 180, 270],
                         help="Rotate extracted frames clockwise by this many degrees. Some phone "
                              "videos have no rotation metadata, so ffmpeg extracts them exactly as "
                              "stored (e.g. a portrait recording comes out sideways) -- the frames "
                              "aren't corrupted, just rotated; use this to correct it.")
    parser.add_argument("--skip_colmap", action="store_true",
                         help="Reuse the existing COLMAP dataset (and, if --input_video is set, the "
                              "already-extracted frames) under <output_dir> as-is, with no re-extraction "
                              "or re-matching. Without this flag, <output_dir>/colmap (and "
                              "<output_dir>/video_frames) are always wiped and rebuilt from the current "
                              "--input_dir/--input_video/--video_* settings, even if they already exist.")
    parser.add_argument("--colmap_gpu", action="store_true",
                         help="Use GPU SIFT in COLMAP (requires a CUDA-enabled colmap build)")
    parser.add_argument("--colmap_matcher", type=str, default=None, choices=["exhaustive", "sequential"],
                         help="COLMAP feature matcher. Defaults to 'sequential' when --input_video is "
                              "set (frames are naturally ordered and overlap mainly with near neighbors) "
                              "and 'exhaustive' otherwise (unordered stills).")
    parser.add_argument("--skip_training", action="store_true",
                         help="Reuse an already-trained point cloud from output/point_cloud")

    parser.add_argument("--remove_background", action=argparse.BooleanOptionalAction, default=False,
                         help="Preprocess input photos (after COLMAP-dir assembly, before COLMAP runs) by "
                              "repainting each photo's background a solid color, using rembg's general-purpose "
                              "u2net_human_seg model (see style_pipeline/background_removal.py). Only "
                              "background pixels are repainted in place -- no crop/resize -- so camera "
                              "geometry stays intact for COLMAP/3DGS. Off by default.")
    parser.add_argument("--bg_color", type=parse_color, default=(255, 255, 255),
                         help="Background color as 'R,G,B' (0-255 each, default 255,255,255 = white): "
                              "fill color for --remove_background, the background of the renders that are "
                              "style-transferred and baked against, and the padding of face crops that "
                              "extend past the render's edge.")

    parser.add_argument("--face_only", action=argparse.BooleanOptionalAction, default=False,
                         help="After Step 1 (reconstruction), prune the trained Gaussians down to a 3D "
                              "region around the face (see style_pipeline/face_region.py), instead of "
                              "keeping the whole reconstructed person/scene. Does not crop input photos -- "
                              "training still uses the full images, so COLMAP registration isn't weakened -- "
                              "it only removes Gaussians outside the detected face region afterwards. Off by "
                              "default.")
    parser.add_argument("--face_conf_threshold", type=float, default=0.5,
                         help="Minimum confidence for the face detector (OpenCV DNN res10 SSD) to accept a "
                              "detection: in training images for --face_only, and in renders for the face "
                              "crop that is style-transferred.")
    parser.add_argument("--face_hit_ratio_threshold", type=float, default=0.5,
                         help="--face_only: a COLMAP sparse point is kept (counts toward the face region) "
                              "if at least this fraction of its observations, in images with a detected "
                              "face, fall inside that image's face bounding box.")
    parser.add_argument("--face_region_percentile", type=float, default=90.0,
                         help="--face_only: percentile (0-100) of qualifying sparse points' distance from "
                              "the region center used as the base radius, before --face_region_margin is "
                              "applied. Not the max distance: qualifying points reliably include a few "
                              "extreme outliers (mistriangulated from spurious wide-baseline matches), and "
                              "sizing the sphere off the single farthest point lets one bad point balloon "
                              "the region out to engulf the whole scene.")
    parser.add_argument("--face_region_margin", type=float, default=0.6,
                         help="--face_only: fractional padding added to the face region's radius (e.g. 0.6 "
                              "= 60%% larger than the --face_region_percentile sphere enclosing the "
                              "qualifying sparse points), so the kept Gaussians cover the whole head, not "
                              "just matched feature points.")

    parser.add_argument("--style_config", type=str, default=DEFAULT_STYLE_CONFIG,
                         help="Style-transfer config yml: checkpoints, model architecture/schedule "
                              "(via the submodule's ckpts/ddpm/configs/*.yml), sampling steps/eta and "
                              f"face crop scale. Defaults to {DEFAULT_STYLE_CONFIG}.")

    parser.add_argument("--style_iterations", type=int, default=3000,
                         help="Number of appearance-optimization steps for baking the style")
    parser.add_argument("--style_lr", type=float, default=0.01)
    parser.add_argument("--bake_geometry", action=argparse.BooleanOptionalAction, default=True,
                         help="While baking, also optimize scaling/rotation/opacity and densify the Gaussians "
                              "so thin stroke lines can be represented (positions stay frozen). "
                              "--no-bake_geometry bakes color only.")
    parser.add_argument("--bake_geometry_lr_scale", type=float, default=0.1,
                         help="--bake_geometry: multiplier on the training learning rates "
                              "(--scaling_lr/--rotation_lr/--opacity_lr) used while baking. Kept low because "
                              "the stylized views disagree with each other, and full-strength geometry "
                              "updates fit those disagreements as floaters.")
    parser.add_argument("--bake_densify_until", type=int, default=1500,
                         help="--bake_geometry: densify (clone/split, prune opacity < 0.005) during the "
                              "first N baking iterations; 0 disables densification. Thresholds follow "
                              "--densify_grad_threshold/--percent_dense.")
    parser.add_argument("--bake_densification_interval", type=int, default=100)
    parser.add_argument("--bake_lambda_edge", type=float, default=0.5,
                         help="Weight of the L1 loss between Sobel edges of the render and the stylized "
                              "target, added to the L1/SSIM baking loss. 0 disables it.")

    args = parser.parse_args()
    output_dir = args.output_dir
    colmap_dir = os.path.join(output_dir, "colmap")
    args.source_path = colmap_dir
    args.model_path = output_dir

    safe_state(False)

    if not args.skip_colmap:
        input_video_is_dir = bool(args.input_video) and os.path.isdir(args.input_video)
        # Sequential matching assumes one continuous, naturally-ordered shot, which
        # only holds for a single video; a directory of per-camera videos has frames
        # from different, unrelated viewpoints concatenated together (see
        # extract_video_frames_dir), so default to exhaustive matching for that case
        # just like a plain --input_dir of stills.
        matcher = args.colmap_matcher or (
            "sequential" if (args.input_video and not input_video_is_dir) else "exhaustive"
        )
        if input_video_is_dir:
            input_dir = extract_video_frames_dir(
                args.input_video, os.path.join(output_dir, "video_frames"),
                args.video_fps, args.video_start, args.video_end, args.video_max_frames,
                rotate=args.video_rotate,
            )
        elif args.input_video:
            input_dir = extract_video_frames(
                args.input_video, os.path.join(output_dir, "video_frames"),
                args.video_fps, args.video_start, args.video_end, args.video_max_frames,
                rotate=args.video_rotate,
            )
        else:
            input_dir = args.input_dir

        prepare_colmap_dataset(
            input_dir, colmap_dir, no_gpu=not args.colmap_gpu, matcher=matcher,
            single_camera=not input_video_is_dir, camera_per_video=input_video_is_dir,
        )

        if args.remove_background:
            # Backgrounds are removed from the undistorted, COLMAP-registered images
            # (colmap/images), not the raw input photos: pose recovery runs on the
            # original, textured photos (background clutter is useful match material
            # for feature matching across widely-separated viewpoints), and only the
            # images actually used for Step 1 training/Step 2+ rendering get repainted.
            # remove_background() only repaints pixels in place (no crop/resize), so
            # this doesn't disturb the camera poses COLMAP just solved for.
            print("=== Preprocessing: removing backgrounds from reconstructed views ===")
            bg_removed_dir = remove_background_dir(
                os.path.join(colmap_dir, "images"), os.path.join(output_dir, "bg_removed"),
                bg_color=args.bg_color,
            )
            colmap_images_dir = os.path.join(colmap_dir, "images")
            for name in os.listdir(bg_removed_dir):
                shutil.copyfile(os.path.join(bg_removed_dir, name), os.path.join(colmap_images_dir, name))

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    b, g, r = args.bg_color
    background = torch.tensor([r / 255.0, g / 255.0, b / 255.0], dtype=torch.float32, device="cuda")

    print("=== Step 1: reconstructing Gaussians from the input views ===")
    if args.skip_training:
        scene, gaussians = load_trained_scene(dataset)
    else:
        scene, gaussians = reconstruct_gaussians(dataset, opt, pipe)

    if args.face_only:
        print("=== Restricting reconstruction to the face region ===")
        center, radius = compute_face_region(
            os.path.join(colmap_dir, "sparse", "0"), os.path.join(colmap_dir, "images"),
            conf_threshold=args.face_conf_threshold,
            hit_ratio_threshold=args.face_hit_ratio_threshold,
            margin_ratio=args.face_region_margin,
            distance_percentile=args.face_region_percentile,
        )
        num_pruned, num_remaining = prune_gaussians_to_region(gaussians, center, radius)
        print(f"[face_only] kept {num_remaining} Gaussians, pruned {num_pruned} "
              f"(region center={center}, radius={radius:.4f})")

    print("=== Step 2: rendering the trained Gaussians ===")
    rendered = render_views(scene, gaussians, pipe, background, os.path.join(output_dir, "renders"))

    print("=== Step 3: style transfer via cycle-diffusion (DDPM/DDIM) ===")
    style_transfer = DDPMStyleTransfer(args.style_config)
    stylized, boxes = stylize_views(
        rendered, style_transfer, background, args.face_conf_threshold, os.path.join(output_dir, "stylized"),
    )
    del style_transfer
    torch.cuda.empty_cache()

    print("=== Steps 4-5: baking + optimizing the stylized color onto the Gaussians ===")
    bake_and_optimize_color(
        scene, gaussians, pipe, background, stylized, boxes, opt, args, output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
