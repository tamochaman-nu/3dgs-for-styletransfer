# 3DGS x cycle-diffusion style-transfer pipeline

Bakes a cycle-diffusion style transfer onto a 3D Gaussian Splatting reconstruction,
built from multi-view input photos.

## Pipeline

1. **Reconstruct** — standard 3DGS training on the input photos (COLMAP + `train.py`'s
   photometric optimization) builds the Gaussians.
2. **Render** — the trained Gaussians are rendered from every input camera pose.
3. **Style transfer** — each render is translated by cycle-diffusion's unpaired
   DDPM/DDIM domain translation (`pipeline/generic_ddpm_wrapper.py`'s
   `GenericDDPMWrapper`): encode with a source-domain DDPM, decode with a
   target-domain (style) DDPM.
4. **Bake** — geometry, scale, rotation, and opacity are frozen; the spherical
   harmonics are reset to diffuse-only (higher-order bands zeroed and frozen).
5. **Optimize** — the diffuse (SH degree-0) color is optimized (Adam, L1 + D-SSIM)
   against the stylized renders, one random training view per step.

Steps 4-5 only fit a single diffuse color per Gaussian, on purpose: each stylized
view comes out of an independent diffusion sample, so the views are not
multi-view-consistent. Restricting the fit to the DC term is what keeps the result
coherent across views instead of overfitting per-view diffusion noise into
view-dependent color (which would show up as floaters/artifacts).

## Directories

- `input/` — put your multi-view photos here (flat directory, any of
  jpg/jpeg/png/bmp/tif/tiff), or a video file (see `--input_video` below).
  Mounted into the container at `/workspace/input`. To keep multiple shoots side
  by side, put each in its own subdirectory (e.g. `input/some_scene/`) and point
  `--input_dir` at it (`/workspace/input/some_scene`) — see `--input_dir` below.
- `output/` — everything the pipeline produces, mounted at `/workspace/output`
  by default (see `--output_dir` below to use a different path, e.g. to keep
  multiple runs side by side):
  - `video_frames/` — frames extracted from `--input_video`, if used
  - `colmap/` — COLMAP dataset built from `--input_dir`/`--input_video` (`images/`, `sparse/0/`)
  - `point_cloud/iteration_N/point_cloud.ply` — the reconstructed (pre-style) Gaussians
  - `renders/` — Step 2 renders
  - `stylized/` — Step 3 style-transferred renders
  - `point_cloud_stylized/point_cloud.ply` — final, style-baked Gaussians
  - `final_renders/` — renders of the final, style-baked Gaussians

  **`video_frames/` and `colmap/` are wiped and rebuilt from scratch on every run**
  unless `--skip_colmap` is passed — re-running with a different `--input_dir`,
  `--input_video`, or `--video_*` slicing without `--skip_colmap` always reflects
  the new input, it never silently reuses a stale dataset from an earlier run.
- `cycle-diffusion-for-anime-face/ckpts/` — cycle-diffusion checkpoints (see below). Mounted so you
  can add/swap checkpoints without rebuilding the image.

## cycle-diffusion DDPM checkpoints

The DDPM/DDIM unpaired-translation path needs **two separately-trained** diffusion
checkpoints: one for the domain that matches your input photos (source) and one for
the style you want to translate into (target), sharing one architecture/noise
schedule. Default values match the **"oneres-cosine"** self-trained
openai/improved-diffusion checkpoint pair:
- `cycle-diffusion-for-anime-face/ckpts/ddpm/ffhq-oneres-cosine-020000.pt` (source: real faces)
- `cycle-diffusion-for-anime-face/ckpts/ddpm/anime-oneres-cosine-020000.pt` (target/style: anime faces)

Both share one architecture (reverse-engineered from their tensor shapes — see the
submodule's `pipeline/step_e12_translate_matched_arch.py` module docstring): a
256-channel UNet with `attention_resolutions="32,16,8"`, `num_res_blocks=1`, and a
**cosine** noise schedule.

Both checkpoints are loaded through the submodule's
`pipeline/generic_ddpm_wrapper.py` (`GenericDDPMWrapper`), the same wrapper its
reference script `pipeline/step_e12_translate_matched_arch.py` uses.

### Style config (`--style_config`)

Every Step-3 setting is read from a config yml, default
`configs/style_transfer/oneres_cosine.yml` (mounted into the container, so edits
need no rebuild). There are no CLI flags or code defaults for any of these:

```yaml
method: cycle_diffusion
model_config: ckpts/ddpm/configs/improved_diffusion_custom_face256_cosine.yml
source_model_path: ckpts/ddpm/ffhq-oneres-cosine-020000.pt
target_model_path: ckpts/ddpm/anime-oneres-cosine-020000.pt
sampling:
  sample_type: ddim
  custom_steps: 1000
  es_steps: 850
  eta: 0.1
  refine_steps: 0
  t_0: null
face_crop:
  scale: 2.0
```

- `model_config` — one of the submodule's `ckpts/ddpm/configs/*.yml`. Its
  `model:` (UNet architecture), `data:` (resolution/channels) and `diffusion:`
  (noise schedule) sections are used as-is. Use
  `improved_diffusion_custom_face256_linear.yml` for `num_res_blocks=2` /
  linear-schedule checkpoints.
- `source_model_path` / `target_model_path` — checkpoints for the input-photo
  domain and the style domain. All paths are relative to
  `cycle-diffusion-for-anime-face/`.
- `sampling` — CycleDiffusion encode/decode settings. The values above match
  `step_e12_translate_matched_arch.py`'s defaults (notably `refine_steps: 0`).
  For stronger stylization, raise `eta` (up to 1.0) and/or lower `es_steps`, at
  the cost of identity/structure preservation.
- `method` — `cycle_diffusion` (this section) or `styleid` (see "StyleID" below).
- `face_crop.scale` — see "Face crop" below.

To use another checkpoint pair, copy the file and pass
`--style_config /workspace/configs/style_transfer/<name>.yml`. A checkpoint that
doesn't match `model_config` fails loudly at `load_state_dict`.

### StyleID (`method: styleid`)

`configs/style_transfer/styleid_mp128.yml` uses the submodule's StyleID
self-attention K/V injection (`pipeline/styleid.py`, same flow as its
`translate_styleid.py`): each face crop is DDIM-inverted, and while the anime
model regenerates it, the crop supplies the decoder self-attention queries
(layout) and one anime reference image (`style_image`) supplies the keys/values
(appearance). One style image is shared by every view, so its K/V are recorded
once at startup.

- `model_type` — name of a submodule `ckpts/ddpm/configs/<model_type>.yml`; must be
  an MP U-Net (`arch: "mp"`) config, e.g. `guided_diffusion_mp128_face256_cosine`.
- `target_model_path` — the anime MP checkpoint
  (`ckpts/ddpm/anime-mp-100000-128-ema09995.pt`). `source_model_path` (FFHQ MP
  checkpoint) is only loaded when `styleid.content_inversion: source`.
- `style_image` — anime reference, relative to this repo's root or absolute (e.g.
  `input/styles/27243460.jpg`, mounted at `/workspace/input/styles/`).
- `styleid` — `num_steps`, `t_0`, `gamma`, `tau`, `layers`, `adain`,
  `content_inversion`; defaults follow `translate_styleid.py` (see its `--help`
  for what each does).

```bash
docker compose run --rm style-transfer \
  --output_dir /workspace/output/output3 --skip_colmap --skip_training --face_only \
  --style_config /workspace/configs/style_transfer/styleid_mp128.yml
```

### Face crop

The models are trained on square, face-centered images, so a whole render
(non-square, face only a small part of it) is never fed in directly. For each
render, the face is detected (OpenCV DNN res10, `--face_conf_threshold`), a
square of side `face_crop.scale` × the face box's longer side is cut around it
(padded with `--bg_color` where it extends past the render), translated at the
model's native resolution, and pasted back. Views with no detected face are
excluded from baking, and the baking loss only covers each view's crop region
(outside it the target would be the unstylized render). Each view's crop before/after translation is saved to
`<output_dir>/stylized/crops/`.

## Running

```bash
git submodule update --init --recursive   # populates the CUDA submodules + cycle-diffusion

docker compose build

docker compose run --rm style-transfer \
  --iterations 30000 \
  --style_iterations 3000
```

Useful flags:
- `--input_dir` — directory of raw multi-view photos (flat, no COLMAP data yet).
  Defaults to `/workspace/input`; pass a subdirectory of it (e.g.
  `/workspace/input/some_scene`) to use a different shoot without touching
  `docker-compose.yml`. Ignored if `--input_video` or `--skip_colmap` is set.
- `--output_dir` — where the pipeline writes everything (see Directories above).
  Defaults to `/workspace/output`; pass a subdirectory of it (e.g.
  `/workspace/output/some_scene`) to keep multiple runs side by side without
  them clobbering each other.
- `--input_video` — use a video, OR a directory of videos (e.g. one file per
  camera in a synchronized multi-camera rig), instead of a folder of stills.
  Frames are extracted with `ffmpeg` into `<output_dir>/video_frames` and used
  as the input photos. For a directory, every video inside is sliced into the
  shared `video_frames` dir with filenames prefixed by each video's name (e.g.
  `Cam1_frame_00001.jpg`), and COLMAP treats each source video as its own
  camera (`--single_camera 0`, see below) rather than assuming they all share
  one camera's intrinsics. Slicing options (apply identically to every video
  in a directory):
  - `--video_fps` (default 2.0) — thin the video down to this many frames/sec.
    Raw video is usually 30-60fps, far denser than SfM needs; 1-3 fps is typical.
  - `--video_start` / `--video_end` — trim to a time range (seconds).
  - `--video_max_frames` — cap the total frame count, evenly subsampling if
    `--video_fps` (within the start/end range) would still yield more than this.
  - `--video_rotate` (0/90/180/270, clockwise) — some phone videos carry no
    rotation metadata, so ffmpeg extracts frames exactly as stored (e.g. a
    portrait recording comes out sideways, 1920x1080 instead of 1080x1920).
    The frames aren't corrupted in this case, just rotated — check a frame in
    `<output_dir>/video_frames` and set this if needed.
  - Every run without `--skip_colmap` re-extracts into a clean
    `<output_dir>/video_frames` (old frames are deleted first), so changing
    `--video_fps`/`--video_start`/`--video_end`/`--video_max_frames` always takes
    effect — it's never silently skipped because frames already exist there.
- `--colmap_matcher` (`exhaustive` or `sequential`) — defaults to `sequential`
  when `--input_video` is a single video file (frames overlap mainly with near
  neighbors in time, and sequential matching is much faster for this), and
  `exhaustive` otherwise (unordered stills, or a directory of per-camera
  videos, where frames from different cameras aren't a single continuous
  sequence).
- COLMAP's `--ImageReader.single_camera` is set to `0` (one camera
  model/intrinsics per source video) automatically whenever `--input_video`
  is a directory, and `1` (one shared camera for everything) otherwise. This
  matters even if the cameras look identical: with `single_camera 1`,
  COLMAP's `feature_extractor` silently drops every image whose pixel
  dimensions don't exactly match the first-registered camera, rather than
  erroring — e.g. one camera's encoder padding its frame width to a macroblock
  boundary (1088px) while another's doesn't (1080px) is enough to make COLMAP
  extract keypoints for only a subset of the cameras, with no warning, and
  the final reconstruction ends up built from whichever single camera's
  frames happened to register into the largest COLMAP sub-model.
- `--remove_background` (off by default) — preprocess input photos, after
  `video_frames`/`bg_removed` assembly but before COLMAP runs, by repainting
  each photo's background a solid color (`--bg_color`, default white
  `255,255,255`). Uses `rembg`'s general-purpose `u2net_human_seg` model
  (`style_pipeline/background_removal.py`), the same `rembg` library
  `~/anime-face-alignment`'s anime-mode background removal uses (different
  model: `u2net_human_seg` here vs. `isnet-anime` there). An earlier version
  of this used BiSeNet (CelebAMask-HQ), which is trained on tightly-cropped,
  face-centered portraits and produced large misclassified regions
  (background mistaken for clothing/hair/skin) on this pipeline's actual
  photos, where the subject is a smaller part of a wider, more distant shot;
  `u2net_human_seg` has no such framing assumption. Only background pixels
  are repainted (alpha-composited, not a hard binary mask) — no crop/resize —
  so camera geometry stays intact for COLMAP/3DGS. Output written to
  `<output_dir>/bg_removed`.
- `--skip_colmap` — reuse the existing `<output_dir>/colmap` dataset (and
  `video_frames`, if applicable) exactly as-is, with no re-extraction or
  re-matching. Without this flag, both are always wiped and rebuilt from the
  current input settings, even if they already exist from a previous run.
- `--skip_training` — reuse an already-trained point cloud from `output/point_cloud`
  (only re-runs rendering, style transfer, and color baking).
- `--colmap_gpu` — use GPU SIFT in COLMAP (needs a CUDA-enabled colmap build; the
  image installs the apt package, which is CPU-only, so this is off by default).
- `--bg_color` — also used as the background of the renders that are
  style-transferred and baked against (so regions removed by `--face_only` render
  as this color, not black), and as padding for face crops.
- `--style_config` — see "Style config" above.
- Baking (Steps 4-5): the diffuse color (SH degree 0) is always optimized;
  higher SH bands are zeroed and positions stay frozen.
  - `--bake_geometry` (default on) — also optimize scaling/rotation/opacity and
    densify, so thin Gaussians can form for stroke lines. `--no-bake_geometry`
    bakes color only.
  - `--bake_geometry_lr_scale` (default 0.1) — multiplier on
    `--scaling_lr`/`--rotation_lr`/`--opacity_lr` while baking.
  - `--bake_densify_until` (default 1500, 0 = off) /
    `--bake_densification_interval` (default 100) — densification window;
    thresholds follow `--densify_grad_threshold`/`--percent_dense`.
  - `--bake_lambda_edge` (default 0.5, 0 = off) — weight of an L1 loss on Sobel
    edges of render vs. stylized target, so few-pixel stroke lines count.
  - `--style_iterations`, `--style_lr`, `--lambda_dssim` as before.
- All standard 3DGS `train.py` flags (`--iterations`, `--resolution`, `--white_background`, …)
  are also accepted, since Step 1 reuses `train.py`'s training loop directly.

## Known limitations

- DDPM/DDIM here runs at a fixed native resolution (typically 256×256), so the
  stylized face crop is upsampled back to its size in the render and is softer than
  the render itself. Outside the face crop, the render is left unstylized.
- The apt `colmap` package has no GPU SIFT support, so `convert.py` runs
  `--no_gpu` by default; expect COLMAP to be the slowest step on large image sets.
