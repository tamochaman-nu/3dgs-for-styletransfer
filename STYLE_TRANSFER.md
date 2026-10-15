# 3DGS x cycle-diffusion style-transfer pipeline

Bakes a cycle-diffusion style transfer onto a 3D Gaussian Splatting reconstruction,
built from multi-view input photos.

## Pipeline

1. **Reconstruct** — standard 3DGS training on the input photos (COLMAP + `train.py`'s
   photometric optimization) builds the Gaussians.
2. **Render** — the trained Gaussians are rendered from every input camera pose.
3. **Style transfer** — each render is translated by cycle-diffusion's unpaired
   DDPM/DDIM domain translation (`model/gan_wrapper/ddpm_ddim_wrapper.py`): encode with
   a source-domain DDPM, decode with a target-domain (style) DDPM.
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
- `cycle-diffusion/ckpts/` — cycle-diffusion checkpoints (see below). Mounted so you
  can add/swap checkpoints without rebuilding the image.

## cycle-diffusion DDPM checkpoints

The DDPM/DDIM unpaired-translation path needs **two separately-trained** diffusion
checkpoints: one for the domain that matches your input photos (source) and one for
the style you want to translate into (target). Follow cycle-diffusion's own
["Pre-trained diffusion models"](cycle-diffusion/README.md) section to download
checkpoints (and, for `afhqcat256`/`afhqdog256`/`afhqwild256`, the shared
`configs.zip`) into `cycle-diffusion/ckpts/ddpm/` (and `cycle-diffusion/ckpts/ddpm/configs/`).

Built-in domain types (`--source_model_type` / `--target_model_type`):
`celeba256`, `ffhq256`, `bedroom256`, `church_outdoor256`, `imagenet512` (default
checkpoint path baked in) and `afhqcat256`, `afhqdog256`, `afhqwild256` (checkpoint
path must be passed explicitly via `--source_model_path` / `--target_model_path`).

There is no general-purpose "any photo, any style" checkpoint here — the source and
target domains must each have a dedicated pretrained DDPM. If your scene content or
desired style doesn't match one of the domains above, you'll need to train your own
DDPM checkpoint on that domain (see cycle-diffusion's upstream training references)
and place it under `cycle-diffusion/ckpts/ddpm/` alongside a matching config yaml.

### Custom domains (e.g. a self-trained anime-face checkpoint)

`model/gan_wrapper/ddpm_ddim_wrapper.py`'s `prepare_ddpm_ddim()` only recognizes the
built-in type names above; anything else is treated as a **custom domain**: pass
`--source_model_path`/`--target_model_path` explicitly, and cycle-diffusion looks for
a config at `cycle-diffusion/ckpts/ddpm/configs/<model_type>.yml` (this fallback is a
local patch to the vendored submodule — it won't survive a
`git submodule update --remote` from upstream cycle-diffusion; re-apply the small
`else` branch in `prepare_ddpm_ddim()` if you ever re-sync from upstream).

This only works out of the box for checkpoints trained with
**openai/improved-diffusion**. The network architecture that `i_DDPM()` builds is
hardcoded per `data.dataset` string in
`model/lib/ddpm_ddim/models/improved_ddpm/script_util.py` (not configurable via the
yaml) — cycle-diffusion's own FFHQ/AFHQ checkpoints use `AFHQ_DICT`
(num_res_blocks=1, attention at 16×16, resblock up/down). This repo's ffhq/anime
test checkpoints used a *different* architecture (num_res_blocks=2, attention at
16×16 **and** 8×8, plain conv up/down-sampling instead of resblock up/down), so a
second preset, `CUSTOM_FACE256`, was added to `script_util.py` and wired up in
`ddpm_ddim_wrapper.py`'s dataset dispatch to match it — reverse-engineered directly
from the checkpoint's `state_dict` shapes. If your checkpoint's architecture differs
from *either* preset, `script_util.py` needs a third one (compare your checkpoint's
`state_dict` shapes against `create_model()`'s arguments to derive it).

The DDIM/DDPM sampler here (`get_beta_schedule`, in
`model/lib/ddpm_ddim/utils/diffusion_utils.py`) supports both a **linear** and a
**cosine** noise schedule (`diffusion.beta_schedule` in the yaml; cosine follows
openai/improved-diffusion's `get_named_beta_schedule` formula) — cosine support was
added here too, since improved-diffusion often defaults to it and using the wrong
one causes the sampler's alphas to diverge from what the model was trained on. This
is not a cosmetic difference: testing against a real photo with `beta_schedule:
linear` produced diverging/garbage output (the per-step estimated-noise magnitude
printed by `encode()` blows up over the last ~150 of 850 steps, and the resulting
image is unrecognizable scribble); switching to `beta_schedule: cosine` fixed it
(coherent, recognizable output). If you use a different custom checkpoint, don't
assume either schedule — check what it was actually trained with, and if the output
looks like noise/scribble, try the other setting first.

Two example configs are included for an ffhq→anime setup using custom checkpoints
that match the `CUSTOM_FACE256` preset above (both `data.dataset: CUSTOM_FACE256`,
`diffusion.beta_schedule: cosine`):
- `cycle-diffusion/ckpts/ddpm/configs/ffhq_custom.yml` (source)
- `cycle-diffusion/ckpts/ddpm/configs/anime.yml` (target/style)

Drop your two checkpoints under `cycle-diffusion/ckpts/ddpm/` and run (paths below
match this repo's actual `ffhq400000.pt` / `anime400000.pt`, already verified to load
and to produce coherent, non-garbage output on a real photo):

```bash
docker compose run --rm style-transfer \
  --source_model_type ffhq_custom --source_model_path ckpts/ddpm/ffhq400000.pt \
  --target_model_type anime       --target_model_path ckpts/ddpm/anime400000.pt \
  --iterations 30000 --style_iterations 3000
```

(`--source_model_path`/`--target_model_path` are resolved relative to `cycle-diffusion/`,
matching where cycle-diffusion's own built-in checkpoints live.)

**Stylization strength**: the verification run above (default `--sample_type ddim
--custom_steps 1000 --eta 0.1 --refine_steps 100 --es_steps 850`) preserved the
source photo's structure closely and read more like a painterly repaint than a
strong anime stylization — expected given `eta=0.1` and encode/decode with a shared
latent path both bias toward preserving source identity. If you want stronger
stylization, try raising `--eta` (up to 1.0) and/or lowering `--es_steps`, at the
cost of identity/structure preservation; there's a real tradeoff here, so re-check
output quality on a single test photo before running the full multi-view pipeline.

## Running

```bash
git submodule update --init --recursive   # populates the CUDA submodules + cycle-diffusion

docker compose build

docker compose run --rm style-transfer \
  --source_model_type afhqcat256 --source_model_path ckpts/ddpm/cat_ema_0.9999_050000.pt \
  --target_model_type afhqdog256 --target_model_path ckpts/ddpm/afhq_dog_4m.pt \
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
- `--input_video` — use a video instead of a folder of stills. Frames are
  extracted with `ffmpeg` into `<output_dir>/video_frames` and used as the input
  photos. Slicing options:
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
  when `--input_video` is set (video frames overlap mainly with near
  neighbors in time, and sequential matching is much faster for this), and
  `exhaustive` otherwise (unordered stills, where every pair might overlap).
- `--skip_colmap` — reuse the existing `<output_dir>/colmap` dataset (and
  `video_frames`, if applicable) exactly as-is, with no re-extraction or
  re-matching. Without this flag, both are always wiped and rebuilt from the
  current input settings, even if they already exist from a previous run.
- `--skip_training` — reuse an already-trained point cloud from `output/point_cloud`
  (only re-runs rendering, style transfer, and color baking).
- `--colmap_gpu` — use GPU SIFT in COLMAP (needs a CUDA-enabled colmap build; the
  image installs the apt package, which is CPU-only, so this is off by default).
- `--sample_type`, `--custom_steps`, `--eta`, `--refine_steps`, `--es_steps` — passed
  straight through to cycle-diffusion's DDPM/DDIM sampler.
- All standard 3DGS `train.py` flags (`--iterations`, `--resolution`, `--white_background`, …)
  are also accepted, since Step 1 reuses `train.py`'s training loop directly.

## Known limitations

- DDPM/DDIM here runs at a fixed native resolution (typically 256×256). Renders are
  downsampled for translation and the stylized result is upsampled back to the
  render resolution without preserving aspect ratio (mild geometric distortion in the
  style-transfer step only — it's used purely as a color-baking target).
- The apt `colmap` package has no GPU SIFT support, so `convert.py` runs
  `--no_gpu` by default; expect COLMAP to be the slowest step on large image sets.
