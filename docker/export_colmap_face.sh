#!/usr/bin/env bash
# Entry point of the `export-colmap-face` service (see export_colmap_face.py).
#
#   docker compose run --rm export-colmap-face [SRC] [DST] [extra export_colmap_face.py options]
#
# SRC: a style_transfer.py --face_only output directory, relative to ./output
#      (default: $EXPORT_FACE_SRC).
# DST: name of the new COLMAP dataset directory, created inside the host
#      directory $EXPORT_FACE_DST_DIR (mounted at /export) (default: $EXPORT_FACE_DST).
# Options come from the service's environment in docker-compose.yml; extra
# options given here are appended and override them.
set -euo pipefail

src="${EXPORT_FACE_SRC:-}"
dst="${EXPORT_FACE_DST:-}"
if [[ $# -gt 0 && "$1" != -* ]]; then src="$1"; shift; fi
if [[ $# -gt 0 && "$1" != -* ]]; then dst="$1"; shift; fi
if [[ -z "$src" || -z "$dst" ]]; then
    echo "[export-colmap-face] Give SRC and DST as arguments or set EXPORT_FACE_SRC / EXPORT_FACE_DST in .env." >&2
    exit 1
fi

cd /workspace
exec python3 export_colmap_face.py \
    --model_dir "/workspace/output/${src}" \
    --out "/export/${dst}" \
    --sh_degree "${SH_DEGREE}" \
    --bg_color "${BG_COLOR}" \
    --alpha_threshold "${ALPHA_THRESHOLD}" \
    --dilate_px "${DILATE_PX}" \
    --feather_sigma "${FEATHER_SIGMA}" \
    --jpeg_quality "${JPEG_QUALITY}" \
    --face_conf_threshold "${FACE_CONF_THRESHOLD}" \
    --face_hit_ratio_threshold "${FACE_HIT_RATIO_THRESHOLD}" \
    --face_region_percentile "${FACE_REGION_PERCENTILE}" \
    --face_region_margin "${FACE_REGION_MARGIN}" \
    "$@"
