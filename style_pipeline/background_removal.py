"""Background removal for input photos, ported from ~/anime-face-alignment's
util/face_parsing.py + util/ffhq_face_alignment.py (same BiSeNet-based method
used there for FFHQ/real-photo images; the anime-specific rembg-based path in
that repo is not used here).

Only the background-removal piece is ported, not FFHQ alignment/cropping/
resizing: this pipeline's input photos are multi-view captures for 3DGS/COLMAP
reconstruction, so camera geometry (image size, framing) must be preserved —
cropping or resizing per-image would break the shared camera poses COLMAP/3DGS
rely on. remove_background() below only repaints background pixels in place.

BiSeNet architecture (ResNet18 backbone, CelebAMask-HQ 19 classes) ported from
https://github.com/yakhyo/face-parsing (MIT License, Copyright (c) 2024
Yakhyokhuja Valikhujaev), keeping the checkpoint's weight keys
(fpn.backbone.*, fpn.arm16.*, ffm.*, conv_out.*, etc.) unchanged. The ResNet
classification head (avgpool/fc) is unused for inference and omitted; the
checkpoint's fpn.backbone.fc.* keys are ignored on load (strict=False, then
asserted to be the only unexpected keys).
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_WEIGHTS_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt"
_WEIGHTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "face_parsing", "resnet18.pt")
_NUM_CLASSES = 19
_INPUT_SIZE = 512
_BACKGROUND_CLASS = 0

# BiSeNet uses an ImageNet-pretrained backbone, so inputs are normalized with
# ImageNet statistics at inference time too.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class _BasicBlock(nn.Module):
    """ResNet18 basic block (two 3x3 convs + residual connection)."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet18Backbone(nn.Module):
    """Standard ResNet18: conv1 -> maxpool -> layer1..4, emitting stride 8/16/32
    feature maps. The classification head (avgpool/fc) is unused for inference
    and omitted (the checkpoint's fpn.backbone.fc.* keys are ignored via
    strict=False on load)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [_BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class _AttentionRefinementModule(nn.Module):
    """Global-average-pool + sigmoid gate channel attention."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_block(x)
        pool = F.avg_pool2d(feat, feat.shape[2:])
        return feat * self.attention(pool)


class _ContextPath(nn.Module):
    """Applies attention-gated feature fusion/upsampling over the backbone's
    outputs to aggregate context features."""

    def __init__(self):
        super().__init__()
        self.backbone = _ResNet18Backbone()
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_head16 = _ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.conv_avg = _ConvBNReLU(512, 128, kernel_size=1, stride=1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        h8, w8 = feat8.shape[2:]
        h16, w16 = feat16.shape[2:]
        h32, w32 = feat32.shape[2:]

        avg = F.avg_pool2d(feat32, feat32.shape[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (h16, w16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (h8, w8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class _FeatureFusionModule(nn.Module):
    """Combines spatial and context features via channel-attention residual fusion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.conv_block(torch.cat([fsp, fcp], dim=1))
        attention = F.avg_pool2d(feat, feat.shape[2:])
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(attention))))
        return feat * attention + feat


class _BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(self.conv_block(x))


class _BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fpn = _ContextPath()
        self.ffm = _FeatureFusionModule(in_channels=256, out_channels=256)
        self.conv_out = _BiSeNetOutput(in_channels=256, mid_channels=256, num_classes=num_classes)
        # Training-only auxiliary heads, unused on the inference path, kept only
        # so the checkpoint's conv_out16.*/conv_out32.* weight keys have somewhere
        # to load into.
        self.conv_out16 = _BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)
        self.conv_out32 = _BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_res8, feat_cp8, _feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        logits = self.conv_out(feat_fuse)
        return F.interpolate(logits, (h, w), mode="bilinear", align_corners=True)


_model = None
_model_device = None


def _download_weights():
    os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
    torch.hub.download_url_to_file(_WEIGHTS_URL, _WEIGHTS_PATH)


def _get_model(device):
    """Cache the model globally, rebuilding only if the device changes."""
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model

    if not os.path.exists(_WEIGHTS_PATH):
        _download_weights()

    model = _BiSeNet(_NUM_CLASSES)
    state_dict = torch.load(_WEIGHTS_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"Failed to load BiSeNet weights (missing keys): {missing}"
    assert all(k.startswith("fpn.backbone.fc.") for k in unexpected), (
        f"Unexpected keys when loading BiSeNet weights: {unexpected}"
    )

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    _model = model
    _model_device = device
    return model


@torch.no_grad()
def _background_mask(img01, device):
    """img01: (3, H, W), [0,1]. Resizes to 512x512 for inference, then resizes
    the predicted class map back to the original resolution with NEAREST (to
    keep hard class boundaries rather than blurring them).
    Returns: (H, W) bool mask, True where class 0 (background) was predicted.
    """
    model = _get_model(device)
    h, w = img01.shape[1:]

    x = img01.unsqueeze(0).to(device)
    x = F.interpolate(x, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    logits = model(x)
    class_map = logits.argmax(dim=1, keepdim=True).float()
    class_map = F.interpolate(class_map, size=(h, w), mode="nearest")
    class_map = class_map.squeeze(0).squeeze(0).long()
    return (class_map == _BACKGROUND_CLASS).cpu().numpy()


def remove_background(image_bgr, device, bg_color=(255, 255, 255)):
    """Hard-repaints BiSeNet's detected background region (class 0) with
    bg_color. bg_color is a BGR tuple, matching image_bgr's channel order.
    Uses the BiSeNet class boundary directly -- no alpha blending or edge
    softening -- and leaves image size/content position unchanged (no crop or
    resize), since this pipeline's photos are multi-view 3DGS/COLMAP input and
    must keep their shared camera geometry.
    """
    img01 = torch.from_numpy(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    mask = _background_mask(img01, device)

    out = image_bgr.copy()
    out[mask] = np.array(bg_color, dtype=out.dtype)
    return out


def remove_background_dir(input_dir, out_dir, bg_color=(255, 255, 255), device=None):
    """Runs remove_background() on every image in input_dir, writing same-named
    outputs into out_dir (created fresh). Returns out_dir.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        out = remove_background(image_bgr, device, bg_color=bg_color)
        cv2.imwrite(os.path.join(out_dir, name), out)

    return out_dir
