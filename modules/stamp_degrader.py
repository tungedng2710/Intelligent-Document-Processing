"""
stamp_degrader.py
=================
Applies realistic degradation to a synthetic red stamp (RGBA, transparent background).
All effects preserve the transparent background — only the stamp ink pixels are modified.

Usage
-----
from modules.stamp_degrader import degrade_stamp, DegradationConfig

result = degrade_stamp("data/test_stamp.png", "data/output_stamp.png")

# or fine-tune every parameter
cfg = DegradationConfig(ink_fade=(0.4, 0.75), ghost_prob=0.6)
result = degrade_stamp("data/test_stamp.png", cfg=cfg)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DegradationConfig:
    """All knobs for the degradation pipeline.

    Each parameter is either a fixed value or a (min, max) range that will be
    sampled uniformly at call time (enables random augmentation).
    """

    # --- Colour ---------------------------------------------------------------
    # Hue shift in HSV space (degrees, can be negative).  Red is at 0°/180°.
    hue_shift: Tuple[float, float] = (-8, 8)
    # Saturation scale factor
    saturation_scale: Tuple[float, float] = (0.75, 1.0)
    # Value (brightness) scale factor
    value_scale: Tuple[float, float] = (0.70, 1.0)

    # --- Global ink opacity ----------------------------------------------------
    # Alpha is multiplied by this scalar globally (simulates overall fading)
    ink_fade: Tuple[float, float] = (0.45, 0.90)

    # --- Uneven ink / texture --------------------------------------------------
    # Perlin-like noise is multiplied onto the alpha channel.
    # This value is the strength (0 = no effect, 1 = full random gaps)
    noise_strength: Tuple[float, float] = (0.15, 0.55)
    # Spatial frequency of the noise (higher = finer grain)
    noise_scale: Tuple[float, float] = (40, 120)

    # --- Ink bleed (dilation) --------------------------------------------------
    # Radius in pixels for soft dilation (simulates ink spreading on paper)
    bleed_radius: Tuple[float, float] = (0.0, 2.5)

    # --- Bald spots (missing ink) ---------------------------------------------
    # Number of elliptical holes punched into the stamp
    bald_spots_count: Tuple[int, int] = (0, 6)
    # Radius range of each bald spot (pixels)
    bald_spot_radius: Tuple[int, int] = (4, 22)

    # --- Edge roughness -------------------------------------------------------
    # Add high-frequency noise to edges (uses alpha gradient as mask)
    edge_roughness: Tuple[float, float] = (0.0, 0.40)

    # --- Elastic / micro distortion ------------------------------------------
    # Max pixel displacement for elastic grid distortion
    elastic_displacement: Tuple[float, float] = (0.0, 4.0)
    # Smoothness of displacement field (higher = smoother, more like rubber)
    elastic_smooth: Tuple[float, float] = (20, 60)

    # --- Slight rotation (skew) -----------------------------------------------
    # Random rotation in degrees (applied before elastic, crops nothing)
    rotation_deg: Tuple[float, float] = (-4.0, 4.0)

    # --- Motion smear ---------------------------------------------------------
    # Applies a short directional motion blur; 0 length = disabled
    smear_length: Tuple[int, int] = (0, 4)
    # Smear angle in degrees
    smear_angle: Tuple[float, float] = (0.0, 360.0)

    # --- Ghost impression -----------------------------------------------------
    # Probability of adding a faint ghost copy
    ghost_prob: float = 0.45
    # Ghost opacity relative to main stamp alpha
    ghost_opacity: Tuple[float, float] = (0.05, 0.20)
    # Ghost pixel offset (dx, dy) range
    ghost_offset: Tuple[int, int] = (3, 14)

    # --- Output ---------------------------------------------------------------
    # If True the final alpha is clipped to [0, 1]; set False to allow slight
    # overshoots before PIL conversion (they will be clipped anyway)
    clip_alpha: bool = True

    def sample(self, rng: random.Random) -> "_SampledConfig":
        """Return a concrete config with all ranges resolved to scalar values."""

        def s(v):
            if isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    return rng.randint(lo, hi)
                return rng.uniform(lo, hi)
            return v

        return _SampledConfig(
            hue_shift=s(self.hue_shift),
            saturation_scale=s(self.saturation_scale),
            value_scale=s(self.value_scale),
            ink_fade=s(self.ink_fade),
            noise_strength=s(self.noise_strength),
            noise_scale=s(self.noise_scale),
            bleed_radius=s(self.bleed_radius),
            bald_spots_count=s(self.bald_spots_count),
            bald_spot_radius=s(self.bald_spot_radius),
            edge_roughness=s(self.edge_roughness),
            elastic_displacement=s(self.elastic_displacement),
            elastic_smooth=s(self.elastic_smooth),
            rotation_deg=s(self.rotation_deg),
            smear_length=s(self.smear_length),
            smear_angle=s(self.smear_angle),
            ghost_prob=self.ghost_prob,
            ghost_opacity=s(self.ghost_opacity),
            ghost_offset=s(self.ghost_offset),
            clip_alpha=self.clip_alpha,
        )


@dataclass
class _SampledConfig:
    """Resolved (scalar) configuration — internal use only."""
    hue_shift: float
    saturation_scale: float
    value_scale: float
    ink_fade: float
    noise_strength: float
    noise_scale: float
    bleed_radius: float
    bald_spots_count: int
    bald_spot_radius: int
    edge_roughness: float
    elastic_displacement: float
    elastic_smooth: float
    rotation_deg: float
    smear_length: int
    smear_angle: float
    ghost_prob: float
    ghost_opacity: float
    ghost_offset: int
    clip_alpha: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perlin_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Fast approximate Perlin noise using bicubic-upsampled random lattice."""
    grid_h = max(2, int(h / scale))
    grid_w = max(2, int(w / scale))
    lattice = rng.random((grid_h, grid_w)).astype(np.float32)
    noise = cv2.resize(lattice, (w, h), interpolation=cv2.INTER_CUBIC)
    # Normalise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Create a directional motion-blur kernel."""
    kernel = np.zeros((length, length), dtype=np.float32)
    cx, cy = length // 2, length // 2
    rad = math.radians(angle_deg)
    for i in range(length):
        t = i - cx
        x = int(round(cx + t * math.cos(rad)))
        y = int(round(cy + t * math.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def _elastic_distort(alpha: np.ndarray, rgb: np.ndarray,
                     displacement: float, smooth: float,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Apply elastic (rubber-sheet) distortion to alpha and rgb channels."""
    h, w = alpha.shape
    # Random displacement fields
    dx = rng.standard_normal((h, w)).astype(np.float32) * displacement
    dy = rng.standard_normal((h, w)).astype(np.float32) * displacement
    dx = gaussian_filter(dx, sigma=smooth)
    dy = gaussian_filter(dy, sigma=smooth)

    # Build coordinate grid
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    src_rows = (rows + dy).clip(0, h - 1)
    src_cols = (cols + dx).clip(0, w - 1)

    coords = [src_rows.ravel(), src_cols.ravel()]

    new_alpha = map_coordinates(alpha, coords, order=1, mode="constant", cval=0.0)
    new_alpha = new_alpha.reshape(h, w)

    new_rgb = np.stack([
        map_coordinates(rgb[..., c], coords, order=1, mode="constant", cval=0.0).reshape(h, w)
        for c in range(3)
    ], axis=-1)

    return new_alpha.astype(np.float32), new_rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _apply_degradation(
    rgba: np.ndarray,
    cfg: _SampledConfig,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> np.ndarray:
    """Apply all degradation effects.  Input/output: float32 RGBA in [0,1]."""

    alpha = rgba[..., 3].copy()   # float32 [0,1]
    rgb   = rgba[..., :3].copy()  # float32 [0,1]
    h, w  = alpha.shape

    # ------------------------------------------------------------------ #
    # 1. Rotation (very slight skew)                                      #
    # ------------------------------------------------------------------ #
    if abs(cfg.rotation_deg) > 0.1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), cfg.rotation_deg, 1.0)
        alpha = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb   = cv2.warpAffine(rgb,   M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ------------------------------------------------------------------ #
    # 2. Elastic micro-distortion                                         #
    # ------------------------------------------------------------------ #
    if cfg.elastic_displacement > 0.1:
        alpha, rgb = _elastic_distort(alpha, rgb, cfg.elastic_displacement,
                                      cfg.elastic_smooth, rng_np)

    # ------------------------------------------------------------------ #
    # 3. Ink bleed — soft dilation via max-filter approximation           #
    # ------------------------------------------------------------------ #
    if cfg.bleed_radius > 0.2:
        r = cfg.bleed_radius
        # Dilate alpha using a small Gaussian "max" approximation
        bleed = gaussian_filter(alpha, sigma=r)
        # Blend original + smeared version at edges
        alpha = np.maximum(alpha, bleed * 0.6)
        alpha = np.clip(alpha, 0, 1)

    # ------------------------------------------------------------------ #
    # 4. Motion smear                                                     #
    # ------------------------------------------------------------------ #
    if cfg.smear_length > 1:
        ks = cfg.smear_length * 2 + 1
        kernel = _motion_blur_kernel(ks, cfg.smear_angle)
        alpha = cv2.filter2D(alpha, -1, kernel)
        for c in range(3):
            rgb[..., c] = cv2.filter2D(rgb[..., c], -1, kernel)

    # ------------------------------------------------------------------ #
    # 5. Colour adjustment (HSV)                                          #
    # ------------------------------------------------------------------ #
    # rgb is float32 [0,1]; convert to uint8 HSV and back
    rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    # OpenCV: H in [0,180], S in [0,255], V in [0,255]
    hsv[..., 0] = (hsv[..., 0] + cfg.hue_shift / 2.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * cfg.saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * cfg.value_scale,     0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # ------------------------------------------------------------------ #
    # 6. Uneven ink — multiply noise onto alpha                           #
    # ------------------------------------------------------------------ #
    noise = _perlin_noise(h, w, cfg.noise_scale, rng_np)
    # Map noise [0,1] → [1-strength, 1] so we only reduce, never boost
    ink_map = 1.0 - cfg.noise_strength * (1.0 - noise)
    alpha = alpha * ink_map

    # ------------------------------------------------------------------ #
    # 7. Bald spots (missing ink ellipses)                                #
    # ------------------------------------------------------------------ #
    for _ in range(cfg.bald_spots_count):
        cx_ = rng_py.randint(0, w - 1)
        cy_ = rng_py.randint(0, h - 1)
        rx  = rng_py.randint(cfg.bald_spot_radius // 2, cfg.bald_spot_radius)
        ry  = rng_py.randint(cfg.bald_spot_radius // 2, cfg.bald_spot_radius)
        ang = rng_py.uniform(0, 360)
        # Soft mask: gaussian-weighted ellipse set to near-zero
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx_, cy_), (rx, ry), ang, 0, 360, 1.0, -1)
        mask = gaussian_filter(mask, sigma=max(rx, ry) / 3.0)
        # Reduce alpha in bald spot region
        alpha = alpha * (1.0 - mask * rng_py.uniform(0.5, 1.0))

    # ------------------------------------------------------------------ #
    # 8. Edge roughness                                                   #
    # ------------------------------------------------------------------ #
    if cfg.edge_roughness > 0.01:
        # Edge = gradient of alpha
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(grad_x**2 + grad_y**2)
        edge_mag = edge_mag / (edge_mag.max() + 1e-8)
        # High-freq noise weighted by edge
        rough_noise = rng_np.standard_normal((h, w)).astype(np.float32)
        rough_noise = gaussian_filter(rough_noise, sigma=1.0)
        alpha = alpha + cfg.edge_roughness * edge_mag * rough_noise
        alpha = np.clip(alpha, 0, 1)

    # ------------------------------------------------------------------ #
    # 9. Global ink fade                                                  #
    # ------------------------------------------------------------------ #
    alpha = alpha * cfg.ink_fade

    # ------------------------------------------------------------------ #
    # 10. Ghost impression                                                 #
    # ------------------------------------------------------------------ #
    if rng_py.random() < cfg.ghost_prob:
        dx = rng_py.randint(-cfg.ghost_offset, cfg.ghost_offset)
        dy = rng_py.randint(-cfg.ghost_offset, cfg.ghost_offset)
        M_ghost = np.float32([[1, 0, dx], [0, 1, dy]])
        ghost_alpha = cv2.warpAffine(alpha, M_ghost, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ghost_strength = rng_py.uniform(*[cfg.ghost_opacity] * 2
                                        if not isinstance(cfg.ghost_opacity, tuple)
                                        else cfg.ghost_opacity)
        # ghost_alpha overrides original partially defined by ghost strength—
        # here we resolve it simply: ghost already at ghost_strength * ink_fade level
        alpha = np.maximum(alpha, ghost_alpha * ghost_strength)

    # ------------------------------------------------------------------ #
    # Final assembly                                                      #
    # ------------------------------------------------------------------ #
    if cfg.clip_alpha:
        alpha = np.clip(alpha, 0, 1)

    out = np.dstack([rgb, alpha[..., np.newaxis]])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def degrade_stamp(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    cfg: Optional[DegradationConfig] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """Load an RGBA stamp PNG, degrade it, optionally save, return PIL Image.

    Parameters
    ----------
    input_path  : path to the source RGBA stamp PNG
    output_path : if given, saves the result to this path
    cfg         : DegradationConfig; defaults to DegradationConfig() (all ranges)
    seed        : random seed for reproducibility; None = random

    Returns
    -------
    PIL.Image in RGBA mode (transparent background preserved)
    """
    if cfg is None:
        cfg = DegradationConfig()

    # ---- seeded RNGs -------------------------------------------------------
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    # ---- load & normalise --------------------------------------------------
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img, dtype=np.float32) / 255.0  # H×W×4  [0,1]

    # ---- sample config -----------------------------------------------------
    sampled = cfg.sample(rng_py)

    # ---- apply pipeline ----------------------------------------------------
    result = _apply_degradation(arr, sampled, rng_py, rng_np)

    # ---- convert back to uint8 PIL -----------------------------------------
    result_u8 = (result * 255).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(result_u8, mode="RGBA")

    if output_path is not None:
        out_img.save(output_path)
        print(f"Saved → {output_path}")

    return out_img


def batch_degrade(
    input_path: str | Path,
    output_dir: str | Path,
    n: int = 10,
    cfg: Optional[DegradationConfig] = None,
    base_seed: Optional[int] = None,
    prefix: str = "stamp",
) -> list[Path]:
    """Generate `n` randomly degraded variants of a stamp.

    Parameters
    ----------
    input_path  : source RGBA stamp PNG
    output_dir  : directory to write output images
    n           : number of variants to generate
    cfg         : DegradationConfig (defaults to DegradationConfig())
    base_seed   : if given, variant i uses seed base_seed+i
    prefix      : filename prefix

    Returns
    -------
    list of output Path objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(n):
        seed = None if base_seed is None else base_seed + i
        out_p = output_dir / f"{prefix}_{i:04d}.png"
        degrade_stamp(input_path, out_p, cfg=cfg, seed=seed)
        paths.append(out_p)

    print(f"Generated {n} degraded stamps in {output_dir}")
    return paths


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Degrade a synthetic stamp PNG")
    parser.add_argument("input",  help="Input RGBA stamp PNG")
    parser.add_argument("output", help="Output path or directory (for batch)")
    parser.add_argument("-n", "--count", type=int, default=1,
                        help="Number of variants to generate (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed")
    parser.add_argument("--ink-fade", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Override ink_fade range, e.g. --ink-fade 0.3 0.7")
    args = parser.parse_args()

    user_cfg = DegradationConfig()
    if args.ink_fade:
        user_cfg.ink_fade = tuple(args.ink_fade)

    if args.count == 1:
        degrade_stamp(args.input, args.output, cfg=user_cfg, seed=args.seed)
    else:
        batch_degrade(args.input, args.output, n=args.count,
                      cfg=user_cfg, base_seed=args.seed)
