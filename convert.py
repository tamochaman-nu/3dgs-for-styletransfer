#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import shlex
import sqlite3
import struct
import subprocess
from argparse import ArgumentParser
import shutil


VIDEO_FRAME_SEP = "_frame_"


def share_camera_per_video(database_path):
    """Rewrites a feature_extractor database made with --ImageReader.single_camera 0
    (one camera per image) so every frame of the same video shares one camera, grouping
    images by the "{video stem}_frame_" filename prefix that style_transfer.py's
    extract_video_frames_dir gives them. Per-image cameras let bundle adjustment fit
    independent OPENCV intrinsics/distortion to every frame, and a frame whose features
    only cover the image center (e.g. textureless clothing/ceiling around the subject)
    overfits k1/k2 until the distortion folds over before the image corner --
    image_undistorter then clamps it to its 0.2x min_scale, i.e. a tiny center crop.
    Must run before matching/mapping; the features themselves don't depend on cameras."""
    db = sqlite3.connect(database_path)
    try:
        groups = {}
        for image_id, name, camera_id in db.execute("SELECT image_id, name, camera_id FROM images ORDER BY name"):
            if VIDEO_FRAME_SEP not in name:
                logging.error(f"--camera_per_video: '{name}' has no '{VIDEO_FRAME_SEP}' video prefix. Exiting.")
                exit(1)
            groups.setdefault(name.rsplit(VIDEO_FRAME_SEP, 1)[0], []).append((image_id, camera_id))
        for video, images in groups.items():
            camera_ids = {camera_id for _, camera_id in images}
            sizes = db.execute(
                f"SELECT DISTINCT width, height FROM cameras WHERE camera_id IN ({','.join('?' * len(camera_ids))})",
                tuple(camera_ids)).fetchall()
            if len(sizes) != 1:
                logging.error(f"--camera_per_video: frames of '{video}' have different sizes {sizes}. Exiting.")
                exit(1)
            shared_id = images[0][1]
            db.execute(f"UPDATE images SET camera_id = ? WHERE image_id IN ({','.join('?' * len(images))})",
                       (shared_id, *(image_id for image_id, _ in images)))
            db.executemany("DELETE FROM cameras WHERE camera_id = ?",
                           [(camera_id,) for camera_id in camera_ids - {shared_id}])
            print(f"[convert] video '{video}': {len(images)} frames share camera {shared_id} "
                  f"({sizes[0][0]}x{sizes[0][1]})")
        db.commit()
    finally:
        db.close()


def count_registered_images(images_bin_path):
    """Reads just the leading num_reg_images uint64 from a COLMAP sparse model's
    images.bin, without parsing the (potentially large) per-image records that follow."""
    with open(images_bin_path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def pick_best_sparse_model(sparse_root):
    """COLMAP's mapper can split a scene into several disconnected reconstructions
    (distorted/sparse/0, /1, ...) when its first seed pair can't grow into a single
    model that covers every image -- e.g. a low-parallax pair of frames from one
    camera in a multi-camera rig stalls out after a couple of images while a later
    seed pair goes on to register everything. Model "0" is just whichever one the
    mapper happened to output first, not necessarily the most complete one, so pick
    by registered image count instead of hardcoding index 0."""
    candidates = []
    for name in sorted(os.listdir(sparse_root)):
        images_bin = os.path.join(sparse_root, name, "images.bin")
        if os.path.isfile(images_bin):
            candidates.append((name, count_registered_images(images_bin)))
    if not candidates:
        logging.error(f"No COLMAP sparse models with images.bin found under {sparse_root}. Exiting.")
        exit(1)
    for name, num_images in candidates:
        print(f"[convert] sparse model '{name}': {num_images} registered images")
    best_name, best_count = max(candidates, key=lambda c: c[1])
    if best_name != "0":
        print(f"[convert] model '0' is not the most complete reconstruction -- using '{best_name}' "
              f"({best_count} images) instead.")
    return os.path.join(sparse_root, best_name)


def run_cmd(cmd):
    # os.system()'s return value is a raw wait status, not the subprocess's actual
    # exit code (e.g. a real exit code of 1 comes back as 256), so checking it with
    # `!= 0` and forwarding it straight to exit() silently produces a 0 (success)
    # process exit code for callers even when the command genuinely failed.
    # subprocess.run()'s returncode is already the real exit code.
    return subprocess.run(shlex.split(cmd)).returncode

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"],
                     help="exhaustive: match every image pair, best for unordered stills. "
                          "sequential: match nearby frames in filename order, best (and much "
                          "faster) for frames extracted from a continuous video sweep.")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--single_camera", type=int, default=1, choices=[0, 1],
                     help="Passed to COLMAP's --ImageReader.single_camera: 1 assumes every image in "
                          "--source_path/input shares one camera model/intrinsics (valid for frames "
                          "from a single physical camera). Use 0 when the input mixes frames from "
                          "multiple distinct cameras (e.g. a multi-camera rig) -- with 1, COLMAP's "
                          "feature_extractor silently drops any image whose dimensions don't match "
                          "the first-registered camera, which otherwise fails without an error.")
parser.add_argument("--camera_per_video", action="store_true",
                     help="With --single_camera 0, share one camera (intrinsics/distortion) across "
                          "all frames of the same video, grouped by the '{video stem}_frame_' filename "
                          "prefix, instead of COLMAP's one camera per image.")
args = parser.parse_args()
if args.camera_per_video and args.single_camera:
    parser.error("--camera_per_video requires --single_camera 0")
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera " + str(args.single_camera) + " \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = run_cmd(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    if args.camera_per_video:
        share_camera_per_video(args.source_path + "/distorted/database.db")

    ## Feature matching
    matcher_command = args.matcher + "_matcher"
    feat_matching_cmd = colmap_command + " " + matcher_command + " \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = run_cmd(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = run_cmd(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
best_model_path = pick_best_sparse_model(args.source_path + "/distorted/sparse")
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + best_model_path + " \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = run_cmd(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_cmd(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
