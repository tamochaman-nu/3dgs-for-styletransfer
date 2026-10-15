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
from style_pipeline.ddpm_bridge import DDPMStyleTransfer
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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


VIDEO_ROTATE_FILTERS = {
    0: None,
    90: "transpose=1",              # 90 deg clockwise
    180: "transpose=1,transpose=1",  # 180 deg
    270: "transpose=2",              # 90 deg counter-clockwise (= 270 deg clockwise)
}


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

    vf = f"fps={fps}"
    if VIDEO_ROTATE_FILTERS[rotate]:
        vf += "," + VIDEO_ROTATE_FILTERS[rotate]

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        *(["-ss", str(start)] if start is not None else []),
        *(["-to", str(end)] if end is not None else []),
        "-vf", vf,
        "-qscale:v", "2",
        os.path.join(out_dir, "frame_%05d.jpg"),
    ]
    print("[video] extracting frames:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = sorted(os.listdir(out_dir))
    print(f"[video] extracted {len(frames)} frames at {fps} fps")

    if max_frames is not None and len(frames) > max_frames:
        keep_idx = {round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)}
        for i, name in enumerate(frames):
            if i not in keep_idx:
                os.remove(os.path.join(out_dir, name))
        print(f"[video] subsampled down to {max_frames} frames")

    return out_dir


def prepare_colmap_dataset(input_dir: str, colmap_dir: str, no_gpu: bool, matcher: str):
    """Turn the raw multi-view photos in input_dir into a COLMAP dataset (images/ +
    sparse/0) under colmap_dir, in the layout Scene() expects.

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

    cmd = [sys.executable, "convert.py", "-s", colmap_dir, "--matcher", matcher]
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


def stylize_views(rendered, style_transfer, out_dir):
    """Step 3: translate each render through cycle-diffusion's DDPM/DDIM model."""
    os.makedirs(out_dir, exist_ok=True)
    stylized = {}
    for name, image in tqdm(rendered.items(), desc="Style-transferring views"):
        styled = style_transfer.stylize(image.cuda())
        torchvision.utils.save_image(styled, os.path.join(out_dir, f"{name}.png"))
        stylized[name] = styled.detach().cpu()
    return stylized


def bake_and_optimize_color(scene, gaussians, pipe, background, stylized,
                             iterations, lr, lambda_dssim, out_dir):
    """Steps 4-5: freeze geometry/opacity, reset to diffuse-only color, and optimize
    the SH degree-0 (diffuse) term against the stylized renders. Higher-order SH
    bands are zeroed and frozen since each stylized view comes from an independent,
    multi-view-inconsistent diffusion sample -- fitting a single diffuse color per
    Gaussian is what keeps the result multi-view consistent."""
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)

    with torch.no_grad():
        gaussians._features_rest.zero_()
    gaussians._features_rest.requires_grad_(False)
    gaussians._features_dc.requires_grad_(True)

    optimizer = torch.optim.Adam([gaussians._features_dc], lr=lr, eps=1e-15)

    cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    names = list(stylized.keys())

    progress = tqdm(range(1, iterations + 1), desc="Baking style onto Gaussians")
    ema_loss = 0.0
    for it in progress:
        name = names[randint(0, len(names) - 1)]
        cam = cams_by_name[name]
        target = stylized[name].cuda()

        image = render(cam, gaussians, pipe, background)["render"]

        Ll1 = l1_loss(image, target)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
        else:
            ssim_value = ssim(image, target)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if it % 10 == 0:
            progress.set_postfix({"loss": f"{ema_loss:.4f}"})

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
                         help="Path to a video file to use instead of --input_dir; frames are "
                              "extracted with ffmpeg (see --video_fps/--video_start/--video_end/"
                              "--video_max_frames) into <output_dir>/video_frames and used as the input photos.")
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

    parser.add_argument("--source_model_type", type=str, required=True,
                         help="cycle-diffusion DDPM domain matching the *input* photos: one of the "
                              "built-in types (celeba256, ffhq256, bedroom256, church_outdoor256, "
                              "afhqcat256, afhqdog256, afhqwild256, imagenet512), or any other name "
                              "to use a custom checkpoint (requires --source_model_path and a matching "
                              "cycle-diffusion/ckpts/ddpm/configs/<name>.yml, e.g. ffhq_custom)")
    parser.add_argument("--source_model_path", type=str, default=None,
                         help="Required for afhq* domains and any custom (non-built-in) model type; "
                              "omit for built-in types with a default checkpoint path")
    parser.add_argument("--target_model_type", type=str, required=True,
                         help="cycle-diffusion DDPM domain to translate the renders into (the style); "
                              "same rules as --source_model_type (e.g. anime for a custom anime-face checkpoint)")
    parser.add_argument("--target_model_path", type=str, default=None)
    parser.add_argument("--sample_type", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--custom_steps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--refine_steps", type=int, default=100)
    parser.add_argument("--es_steps", type=int, default=850)

    parser.add_argument("--style_iterations", type=int, default=3000,
                         help="Number of appearance-optimization steps for baking the style")
    parser.add_argument("--style_lr", type=float, default=0.01)

    args = parser.parse_args()
    output_dir = args.output_dir
    colmap_dir = os.path.join(output_dir, "colmap")
    args.source_path = colmap_dir
    args.model_path = output_dir

    safe_state(False)

    if not args.skip_colmap:
        matcher = args.colmap_matcher or ("sequential" if args.input_video else "exhaustive")
        if args.input_video:
            input_dir = extract_video_frames(
                args.input_video, os.path.join(output_dir, "video_frames"),
                args.video_fps, args.video_start, args.video_end, args.video_max_frames,
                rotate=args.video_rotate,
            )
        else:
            input_dir = args.input_dir
        prepare_colmap_dataset(input_dir, colmap_dir, no_gpu=not args.colmap_gpu, matcher=matcher)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("=== Step 1: reconstructing Gaussians from the input views ===")
    if args.skip_training:
        scene, gaussians = load_trained_scene(dataset)
    else:
        scene, gaussians = reconstruct_gaussians(dataset, opt, pipe)

    print("=== Step 2: rendering the trained Gaussians ===")
    rendered = render_views(scene, gaussians, pipe, background, os.path.join(output_dir, "renders"))

    print("=== Step 3: style transfer via cycle-diffusion (DDPM/DDIM) ===")
    style_transfer = DDPMStyleTransfer(
        source_model_type=args.source_model_type, source_model_path=args.source_model_path,
        target_model_type=args.target_model_type, target_model_path=args.target_model_path,
        sample_type=args.sample_type, custom_steps=args.custom_steps, eta=args.eta,
        refine_steps=args.refine_steps, es_steps=args.es_steps,
    )
    stylized = stylize_views(rendered, style_transfer, os.path.join(output_dir, "stylized"))
    del style_transfer
    torch.cuda.empty_cache()

    print("=== Steps 4-5: baking + optimizing the stylized color onto the Gaussians ===")
    bake_and_optimize_color(
        scene, gaussians, pipe, background, stylized,
        args.style_iterations, args.style_lr, opt.lambda_dssim, output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
