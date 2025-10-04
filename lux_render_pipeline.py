#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luxury Real Estate Render Refinement Pipeline
---------------------------------------------
Transforms basic 3D model images into high-resolution marketing visuals by:
- Preprocessing (Canny + optional Depth)
- Structure-preserving image-to-image diffusion (ControlNet)
- Optional SDXL refiner
- High-resolution upscaling (latent x2 and/or Real-ESRGAN)
- Photo finishing (ACES tonemap, white balance, bloom, vignette, grain)
- Brand overlays (logo + caption)
- Batch CLI

Usage examples:
    # Interior (day), keep structure tight, 2K output
    python lux_render_pipeline.py \
        --input ./drafts/*.png --out ./final \
        --prompt "luxury interior, natural daylight, oak wood floor, marble countertops, photo-realistic, architectural photography, ultra-detailed, 35mm, subtle styling" \
        --neg "low detail, cartoon, blurry, extra walls, distortion, watermark, text" \
        --width 1024 --height 768 --steps 30 --strength 0.45 --gs 7.5 \
        --brand_text "The Veridian | Penthouse 21B" --logo ./brand/logo.png

    # Exterior (golden hour), 4K output + Real-ESRGAN
    python lux_render_pipeline.py \
        --input ./exteriors/*.jpg --out ./final4k --w4k \
        --prompt "luxury real estate exterior, golden hour, glass and stone facade, landscaped garden, dramatic sky, cinematic, photo-real, ultra-detailed" \
        --neg "overexposed, warped structure, crooked lines, lowres, noise, text, logo" \
        --width 1024 --height 576 --steps 35 --strength 0.5 --gs 7.0 \
        --use_realesrgan --no-depth
"""
from __future__ import annotations
import os, math, glob, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import Generator

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionLatentUpscalePipeline,
    UniPCMultistepScheduler,
)
# SDXL (optional). Imported lazily only if used.
try:
    from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline
except Exception:
    StableDiffusionXLControlNetPipeline = None
    StableDiffusionXLImg2ImgPipeline = None

# Annotators
from controlnet_aux import CannyDetector, MidasDetector

# Optional Real-ESRGAN
try:
    from realesrgan import RealESRGAN
    _HAS_REALESRGAN = True
except Exception:
    _HAS_REALESRGAN = False

# --------------------------

# Utilities
# --------------------------
def seed_all(seed: int) -> Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

def load_image(path: Union[str, Path]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def save_image(img: Image.Image, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def pil_to_np(img: Image.Image, to_float: bool = True) -> np.ndarray:
    arr = np.array(img)
    if to_float:
        arr = arr.astype(np.float32) / 255.0
    return arr

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)

def resize_to_multiple(
    img: Image.Image,
    mult: int = 8,
    target: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    if target is None:
        w, h = img.size
    else:
        w, h = target
    w = (w // mult) * mult
    h = (h // mult) * mult
    return img.resize((w, h), Image.Resampling.LANCZOS)

# --------------------------
# Preprocessors (Canny + Depth)
# --------------------------
class Preprocessor:
    def __init__(self, canny_low: int = 100, canny_high: int = 200, use_depth: bool = True):
        self.canny = CannyDetector(low_threshold=canny_low, high_threshold=canny_high)
        self.depth = MidasDetector(model_type="dpt_large") if use_depth else None

    def make_canny(self, image: Image.Image) -> Image.Image:
        return self.canny(image)

    def make_depth(self, image: Image.Image) -> Optional[Image.Image]:
        if self.depth is None:
            return None
        depth = self.depth(image)
        # normalize depth for visualization/control
        arr = np.array(depth).astype(np.float32)
        arr = (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))
        return Image.fromarray((arr * 255).astype(np.uint8)).convert("L")

# --------------------------
# Post-processing (photo finish)
# --------------------------
def aces_film_tonemap(rgb: np.ndarray) -> np.ndarray:
    """ACES-like tonemap, expects float RGB in [0,1]."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e), 0.0, 1.0)

def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    mean = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
    scale = mean.mean() / mean
    return np.clip(rgb * scale, 0.0, 1.0)

def add_bloom(rgb: np.ndarray, threshold: float = 0.8, blur_radius: int = 9, intensity: float = 0.25) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    lum = 0.2126 * rgb[...,0] + 0.7152 * rgb[...,1] + 0.0722 * rgb[...,2]
    mask = (lum > threshold).astype(np.float32)
    glow = np.stack([gaussian_filter(rgb[...,i] * mask, blur_radius) for i in range(3)], axis=-1)
    return np.clip(rgb + intensity * glow, 0.0, 1.0)

def add_vignette(rgb: np.ndarray, strength: float = 0.2) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r /= r.max() + 1e-6
    mask = 1.0 - strength * (r**2)
    return np.clip(rgb * mask[...,None], 0.0, 1.0)

def add_film_grain(rgb: np.ndarray, amount: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount, size=rgb.shape).astype(np.float32)
    return np.clip(rgb + noise, 0.0, 1.0)


def apply_material_response_finishing(
    rgb: np.ndarray,
    texture_boost: float = 0.25,
    ambient_occlusion: float = 0.12,
    highlight_warmth: float = 0.08,
    haze_strength: float = 0.06,
    haze_tint: Tuple[float, float, float] = (0.82, 0.88, 0.96),
    floor_plank_contrast: float = 0.12,
    floor_specular: float = 0.18,
    textile_contrast: float = 0.18,
    leather_sheen: float = 0.16,
    fireplace_glow: float = 0.18,
    fireplace_glow_radius: float = 45.0,
    window_reflection: float = 0.12,
    bedding_relief: float = 0.16,
    wall_texture: float = 0.1,
    painting_integration: float = 0.1,
    window_light_wrap: float = 0.14,
    exterior_atmosphere: float = 0.12,
) -> np.ndarray:
    """Empirical material response layer to emphasize texture, shadowing, and atmosphere."""

    from scipy.ndimage import gaussian_filter, sobel  # Lazy import to avoid hard dependency at module import time

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    if texture_boost > 0:
        # High-frequency boost to reveal subtle grain and fabric weave
        blurred = gaussian_filter(rgb, sigma=(1.1, 1.1, 0))
        detail = rgb - blurred
        rgb = np.clip(rgb + texture_boost * detail, 0.0, 1.0)

    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    sat = np.maximum(rgb.max(axis=2) - rgb.min(axis=2), 1e-6)

    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y_norm = yy.astype(np.float32) / max(1, h - 1)
    x_norm = xx.astype(np.float32) / max(1, w - 1)

    if ambient_occlusion > 0:
        # Edge-based occlusion mask to ground furniture with the floor
        grad_x = sobel(lum, axis=1)
        grad_y = sobel(lum, axis=0)
        edge_mag = np.hypot(grad_x, grad_y)
        if edge_mag.max() > 0:
            edge_mag /= edge_mag.max()
        occlusion = gaussian_filter(edge_mag, sigma=1.2)
        occlusion = np.clip(occlusion, 0.0, 1.0)
        floor_contact = gaussian_filter(floor_mask * (1.0 - floor_mask), sigma=2.0)
        contact_weight = np.clip(floor_contact, 0.0, 1.0)
        shadow = 1.0 - ambient_occlusion * (occlusion + 0.6 * contact_weight)
        rgb = np.clip(rgb * shadow[..., None], 0.0, 1.0)

    # Floor plank definition + specular streaks guided by perspective gradient
    floor_mask = np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0)
    if floor_plank_contrast > 0:
        floor_detail = rgb * floor_mask[..., None]
        blurred_floor = gaussian_filter(floor_detail, sigma=(1.6, 1.2, 0))
        plank_detail = floor_detail - blurred_floor
        rgb = np.clip(rgb + floor_plank_contrast * plank_detail * floor_mask[..., None], 0.0, 1.0)
        wood_luma = gaussian_filter(lum * floor_mask, sigma=(0.8, 3.2))
        grain = np.abs(sobel(wood_luma, axis=1))
        if grain.max() > 0:
            grain /= grain.max()
        grain = gaussian_filter(grain, sigma=(0.8, 1.6)) * floor_mask
        warm_wood = np.array([0.86, 0.74, 0.58], dtype=np.float32)
        rgb = np.clip(rgb + 0.12 * floor_plank_contrast * grain[..., None] * (warm_wood - rgb), 0.0, 1.0)

    if floor_specular > 0:
        grad_floor = sobel(lum * floor_mask, axis=1)
        grad_floor = np.abs(grad_floor)
        if grad_floor.max() > 0:
            grad_floor /= grad_floor.max()
        streaks = gaussian_filter(grad_floor, sigma=(2.2, 6.0))
        streaks = np.clip(streaks, 0.0, 1.0)
        spec_color = np.array([1.0, 0.93, 0.78], dtype=np.float32)
        rgb = np.clip(rgb + floor_specular * streaks[..., None] * floor_mask[..., None] * (spec_color - rgb), 0.0, 1.0)

    if textile_contrast > 0:
        # Identify whites/neutrals with low saturation for linen separation
        textile_mask = np.clip((lum - 0.45) / 0.35, 0.0, 1.0) * np.clip(0.28 - sat, 0.0, 1.0) / 0.28
        textile_mask = gaussian_filter(textile_mask, sigma=1.1)
        textile_detail = rgb - gaussian_filter(rgb, sigma=(1.4, 1.4, 0))
        rgb = np.clip(rgb + textile_contrast * textile_mask[..., None] * textile_detail, 0.0, 1.0)

    if leather_sheen > 0:
        # Neutral mid-tone mask for leather upholstery
        leather_mask = np.clip((0.55 - lum) / 0.35, 0.0, 1.0) * np.clip((0.22 - sat), 0.0, 1.0) / 0.22
        leather_mask = gaussian_filter(leather_mask, sigma=1.6)
        sheen = gaussian_filter(lum, sigma=3.0)
        if sheen.max() > 0:
            sheen = np.clip((sheen - 0.25) / 0.5, 0.0, 1.0)
        sheen_color = np.array([0.82, 0.73, 0.62], dtype=np.float32)
        rgb = np.clip(rgb * (1.0 - leather_sheen * leather_mask[..., None]) + sheen_color * leather_sheen * leather_mask[..., None] * sheen[..., None], 0.0, 1.0)

    if fireplace_glow > 0:
        warm_mask = ((rgb[..., 0] > 0.55) & (rgb[..., 0] - rgb[..., 1] > 0.08) & (rgb[..., 1] > rgb[..., 2])).astype(np.float32)
        warm_mask = gaussian_filter(warm_mask, sigma=1.5)
        sigma = max(1.5, fireplace_glow_radius / 18.0)
        glow = gaussian_filter(warm_mask, sigma=sigma)
        if glow.max() > 0:
            glow /= glow.max()
        fireplace_bias = np.clip((x_norm - 0.48) / 0.52, 0.0, 1.0)
        glow *= fireplace_bias
        glow_color = np.array([1.0, 0.68, 0.42], dtype=np.float32)
        rgb = np.clip(rgb + fireplace_glow * glow[..., None] * (glow_color - rgb), 0.0, 1.0)

    if window_reflection > 0:
        bright_columns = np.mean(np.clip(lum - 0.65, 0.0, 1.0), axis=0)
        bright_columns = gaussian_filter(bright_columns, sigma=4.0)
        if bright_columns.max() > 0:
            bright_columns /= bright_columns.max()
        reflection = np.tile(bright_columns, (h, 1))
        reflection = gaussian_filter(reflection, sigma=(5.0, 3.0))
        reflection *= floor_mask
        reflection_color = np.array([1.0, 0.98, 0.9], dtype=np.float32)
        rgb = np.clip(rgb + window_reflection * reflection[..., None] * (reflection_color - rgb), 0.0, 1.0)

    if window_light_wrap > 0:
        window_side = np.clip((x_norm - 0.45) / 0.55, 0.0, 1.0)
        wrap = gaussian_filter(lum * window_side, sigma=3.2)
        if wrap.max() > 0:
            wrap = np.clip((wrap - wrap.min()) / (wrap.max() - wrap.min() + 1e-6), 0.0, 1.0)
        wrap_color = np.array([1.0, 0.95, 0.82], dtype=np.float32)
        rgb = np.clip(rgb + window_light_wrap * wrap[..., None] * (wrap_color - rgb), 0.0, 1.0)

    if exterior_atmosphere > 0:
        exterior_mask = np.clip((sat - 0.18) / 0.6, 0.0, 1.0) * np.clip((lum - 0.28) / 0.72, 0.0, 1.0)
        exterior_mask *= np.clip((x_norm - 0.38) / 0.62, 0.0, 1.0)
        exterior_mask = gaussian_filter(exterior_mask, sigma=3.5)
        depth_falloff = np.clip(1.0 - y_norm, 0.0, 1.0)
        haze = exterior_atmosphere * exterior_mask * depth_falloff
        sky_tint = np.array([0.78, 0.86, 0.92], dtype=np.float32)
        rgb = np.clip(rgb * (1.0 - haze[..., None]) + sky_tint * haze[..., None], 0.0, 1.0)

    if highlight_warmth > 0:
        # Warm the brightest values to simulate fireplace spill and sunlit reflections
        highlight_mask = np.clip((lum - 0.58) / 0.35, 0.0, 1.0)
        warm_color = np.array([1.0, 0.78, 0.55], dtype=np.float32)
        rgb = np.clip(
            rgb + highlight_mask[..., None] * highlight_warmth * (warm_color - rgb),
            0.0,
            1.0,
        )

    if haze_strength > 0:
        # Vertical gradient bias with gentle falloff toward window region
        gradient = y_norm.astype(np.float32)
        haze = haze_strength * gradient
        tint = np.array(haze_tint, dtype=np.float32)
        tint = np.clip(tint, 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - haze[..., None]) + tint * haze[..., None], 0.0, 1.0)

    if bedding_relief > 0:
        bedding_mask = np.clip((lum - 0.35) / 0.35, 0.0, 1.0) * np.clip((0.3 - sat) / 0.3, 0.0, 1.0)
        bedding_mask *= np.clip((0.75 - y_norm) / 0.75, 0.0, 1.0)
        bedding_mask = gaussian_filter(bedding_mask, sigma=1.0)
        bedding_detail = rgb - gaussian_filter(rgb, sigma=(1.0, 1.0, 0))
        shading = gaussian_filter(lum, sigma=1.6) - gaussian_filter(lum, sigma=4.2)
        shading = np.clip(shading, -0.25, 0.25)
        rgb = np.clip(rgb + bedding_relief * bedding_mask[..., None] * bedding_detail, 0.0, 1.0)
        rgb = np.clip(rgb - bedding_relief * 0.35 * bedding_mask[..., None] * shading[..., None], 0.0, 1.0)

    if wall_texture > 0:
        wall_mask = np.clip((lum - 0.32) / 0.45, 0.0, 1.0) * np.clip((0.26 - sat) / 0.26, 0.0, 1.0)
        wall_mask *= np.clip(1.0 - floor_mask, 0.0, 1.0)
        wall_mask = gaussian_filter(wall_mask, sigma=1.4)
        wall_detail = rgb - gaussian_filter(rgb, sigma=(2.6, 2.6, 0))
        wall_detail = gaussian_filter(wall_detail, sigma=(0.9, 0.9, 0))
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 1.0, size=lum.shape).astype(np.float32)
        noise = gaussian_filter(noise, sigma=3.0)
        if noise.max() > noise.min():
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6) - 0.5
        wall_detail += noise[..., None] * 0.06
        rgb = np.clip(rgb + wall_texture * wall_mask[..., None] * wall_detail, 0.0, 1.0)

    if painting_integration > 0:
        art_mask = np.clip((sat - 0.24) / 0.5, 0.0, 1.0) * np.clip((lum - 0.2) / 0.7, 0.0, 1.0)
        art_mask *= np.clip((0.6 - x_norm) / 0.6, 0.0, 1.0)
        art_mask *= np.clip(1.0 - floor_mask, 0.0, 1.0)
        art_mask = gaussian_filter(art_mask, sigma=1.3)
        rim = gaussian_filter(art_mask, sigma=0.9) - gaussian_filter(art_mask, sigma=2.2)
        rim = np.clip(rim, 0.0, 1.0)
        warm_fill = np.array([1.0, 0.84, 0.68], dtype=np.float32)
        rgb = np.clip(rgb + painting_integration * rim[..., None] * (warm_fill - rgb), 0.0, 1.0)
        canvas_shadow = gaussian_filter(art_mask, sigma=3.0)
        rgb = np.clip(rgb * (1.0 - painting_integration * 0.12 * canvas_shadow[..., None]), 0.0, 1.0)

    return rgb

def adjust_contrast_saturation(rgb: np.ndarray, contrast: float = 1.08, saturation: float = 1.05) -> np.ndarray:
    # Contrast in linear light
    gray = rgb.mean(axis=2, keepdims=True)
    rgb = (rgb - gray) * contrast + gray
    # Saturation in HSV-like space (simple)
    maxc = rgb.max(axis=2, keepdims=True)
    minc = rgb.min(axis=2, keepdims=True)
    sat = (maxc - minc) + 1e-6
    mean = rgb.mean(axis=2, keepdims=True)
    rgb = (rgb - mean) * saturation + mean
    return np.clip(rgb, 0.0, 1.0)

# --------------------------
# Branding (logo + caption)
# --------------------------
def overlay_logo_caption(img: Image.Image, logo_path: Optional[str], text: Optional[str], margin: int = 36) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    W, H = canvas.size

    # Caption
    if text:
        try:
            # Use a clean sans (replace with your corporate font)
            font = ImageFont.truetype("arial.ttf", size=max(22, H // 40))
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = margin
        y = H - th - margin
        # subtle text box (white on black with 60% alpha)
        pad = 14
        box = [x - pad, y - pad, x + tw + pad, y + th + pad]
        draw.rectangle(box, fill=(0,0,0,160))
        draw.text((x, y), text, fill=(255,255,255,230), font=font)

    # Logo
    if logo_path and Path(logo_path).exists():
        logo = Image.open(logo_path).convert("RGBA")
        # scale ~10% of the short edge
        target_h = int(min(W, H) * 0.12)
        scale = target_h / logo.height
        logo = logo.resize((int(logo.width * scale), target_h), Image.Resampling.LANCZOS)
        lx = W - logo.width - margin
        ly = H - logo.height - margin
        canvas.alpha_composite(logo, (lx, ly)) if canvas.mode == "RGBA" else canvas.paste(logo, (lx, ly), logo)

    return canvas

# --------------------------
# Config
# --------------------------
@dataclass
class ModelIDs:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    controlnet_canny: str = "lllyasviel/sd-controlnet-canny"
    controlnet_depth: str = "lllyasviel/sd-controlnet-depth"
    upscaler_id: str = "stabilityai/sd-x2-latent-upscaler"
    refiner: Optional[str] = None  # e.g., "stabilityai/stable-diffusion-xl-refiner-1.0"

@dataclass
class RenderConfig:
    width: int = 1024
    height: int = 768
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.5
    seed: int = 1234

@dataclass
class FinishConfig:
    aces: bool = True
    contrast: float = 1.08
    saturation: float = 1.05
    white_balance: bool = True
    bloom: bool = True
    bloom_threshold: float = 0.82
    bloom_radius: int = 9
    bloom_intensity: float = 0.22
    vignette: bool = True
    vignette_strength: float = 0.18
    grain: bool = True
    grain_amount: float = 0.012
    material_response: bool = False
    texture_boost: float = 0.25
    ambient_occlusion: float = 0.12
    highlight_warmth: float = 0.08
    haze_strength: float = 0.06
    haze_tint: Tuple[float, float, float] = (0.82, 0.88, 0.96)
    floor_plank_contrast: float = 0.12
    floor_specular: float = 0.18
    textile_contrast: float = 0.18
    leather_sheen: float = 0.16
    fireplace_glow: float = 0.18
    fireplace_glow_radius: float = 45.0
    window_reflection: float = 0.12
    bedding_relief: float = 0.16
    wall_texture: float = 0.1
    painting_integration: float = 0.1
    window_light_wrap: float = 0.14
    exterior_atmosphere: float = 0.12

# --------------------------
# Core pipeline
# --------------------------
class LuxuryRenderPipeline:
    def __init__(
        self,
        model_ids: ModelIDs,
        device: Optional[str] = None,
        fp16: bool = True,
        use_realesrgan: bool = False,
        use_depth: bool = True,
    ):
        self.model_ids = model_ids
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
        self._use_depth = use_depth

        # Load ControlNets
        print("[Load] ControlNets...")
        self.cn_canny = ControlNetModel.from_pretrained(model_ids.controlnet_canny, torch_dtype=self.dtype)
        self.cn_depth = (
            ControlNetModel.from_pretrained(model_ids.controlnet_depth, torch_dtype=self.dtype)
            if self._use_depth
            else None
        )
        controlnets: List[ControlNetModel] = [self.cn_canny]
        if self.cn_depth is not None:
            controlnets.append(self.cn_depth)
        controlnet_arg: List[ControlNetModel] = controlnets

        # Pick pipeline type based on base model family (SD1.5 vs SDXL)
        is_sdxl = "xl" in model_ids.base_model.lower()

        if is_sdxl:
            if StableDiffusionXLControlNetPipeline is None:
                raise RuntimeError("diffusers SDXL pipeline not available; upgrade diffusers or install extras.")
            print("[Load] SDXL + ControlNet pipeline...")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_ids.base_model,
                controlnet=controlnet_arg,
                torch_dtype=self.dtype,
                add_watermarker=False,
            )
            if model_ids.refiner:
                print("[Load] SDXL refiner...")
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_ids.refiner,
                    torch_dtype=self.dtype,
                    add_watermarker=False,
                )
            else:
                self.refiner = None
        else:
            print("[Load] SD1.5 + ControlNet Img2Img pipeline...")
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model_ids.base_model,
                controlnet=controlnet_arg,
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None,
            )
            self.refiner = None

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        # Upscaler (latent)
        print("[Load] Latent x2 upscaler...")
        try:
            self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
                model_ids.upscaler_id, torch_dtype=self.dtype
            ).to(self.device)
        except Exception as e:
            print(f"[Warn] Latent upscaler not available ({e}). Will skip latent upscaling.")
            self.upscaler = None

        # Real-ESRGAN (optional)
        self._use_realesrgan = use_realesrgan and _HAS_REALESRGAN
        if use_realesrgan and not _HAS_REALESRGAN:
            print("[Warn] Real-ESRGAN requested but not installed. Run: pip install realesrgan basicsr")
        if self._use_realesrgan:
            print("[Load] Real-ESRGAN x4...")
            self.realesrgan = RealESRGAN(self.device, scale=4)
            self.realesrgan.load_weights("RealESRGAN_x4plus.pth", download=True)

        # Preprocessor
        self.pre = Preprocessor(use_depth=self._use_depth)

    @torch.inference_mode()
    def enhance(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        cfg: RenderConfig,
        finish: FinishConfig,
        brand_logo: Optional[str] = None,
        brand_text: Optional[str] = None,
        export_4k: bool = False,
    ) -> Image.Image:
        # 1) Prepare images
        init = resize_to_multiple(init_image, mult=8, target=(cfg.width, cfg.height))
        canny = self.pre.make_canny(init)
        depth = self.pre.make_depth(init)
        control_images = [canny]
        if depth is not None:
            control_images.append(depth)
        control_image_arg = control_images[0] if len(control_images) == 1 else control_images

        # 2) Diffusion pass (structure-preserving)
        g = seed_all(cfg.seed)
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=control_image_arg,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            strength=cfg.strength,
            generator=g,
        )
        print("[Run] Diffusion (ControlNet img2img)...")
        out = self.pipe(**pipe_kwargs).images[0]

        # 3) Optional SDXL refiner pass (if configured)
        if self.refiner is not None:
            print("[Run] SDXL refiner pass...")
            out = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=out,
                num_inference_steps=max(15, cfg.steps // 2),
                strength=min(0.35, cfg.strength * 0.7),
                guidance_scale=max(5.5, cfg.guidance_scale - 1.0),
                generator=g,
            ).images[0]

        # 4) Latent x2 upscale
        if self.upscaler is not None:
            print("[Run] Latent x2 upscale...")
            out = self.upscaler(
                image=out,
                num_inference_steps=10,
                guidance_scale=0.0,
                generator=g,
            ).images[0]

        # 5) Real-ESRGAN (crisp tiled upscaling to 4K/8K)
        if export_4k:
            target_min = 2160  # min edge ~ 4K height
        else:
            target_min = min(out.size)

        if self._use_realesrgan and (export_4k or min(out.size) < target_min):
            print("[Run] Real-ESRGAN x4 (as needed)...")
            # upscale progressively until min edge >= target_min
            pil = out
            while min(pil.size) < target_min:
                pil = self.realesrgan.predict(pil)
            out = pil

        # 6) Photo finishing
        print("[Finish] Photo grade...")
        rgb = pil_to_np(out)
        if finish.white_balance: rgb = gray_world_white_balance(rgb)
        if finish.aces: rgb = aces_film_tonemap(rgb)
        rgb = adjust_contrast_saturation(rgb, finish.contrast, finish.saturation)
        if finish.bloom: rgb = add_bloom(rgb, finish.bloom_threshold, finish.bloom_radius, finish.bloom_intensity)
        if finish.vignette: rgb = add_vignette(rgb, finish.vignette_strength)
        if finish.grain: rgb = add_film_grain(rgb, finish.grain_amount, seed=cfg.seed)
        if finish.material_response:
            rgb = apply_material_response_finishing(
                rgb,
                texture_boost=finish.texture_boost,
                ambient_occlusion=finish.ambient_occlusion,
                highlight_warmth=finish.highlight_warmth,
                haze_strength=finish.haze_strength,
                haze_tint=finish.haze_tint,
                floor_plank_contrast=finish.floor_plank_contrast,
                floor_specular=finish.floor_specular,
                textile_contrast=finish.textile_contrast,
                leather_sheen=finish.leather_sheen,
                fireplace_glow=finish.fireplace_glow,
                fireplace_glow_radius=finish.fireplace_glow_radius,
                window_reflection=finish.window_reflection,
                bedding_relief=finish.bedding_relief,
                wall_texture=finish.wall_texture,
                painting_integration=finish.painting_integration,
                window_light_wrap=finish.window_light_wrap,
                exterior_atmosphere=finish.exterior_atmosphere,
            )
        out = np_to_pil(rgb)

        # 7) Branding
        out = overlay_logo_caption(out, brand_logo, brand_text)

        return out

# --------------------------
# CLI
# --------------------------
import typer
app = typer.Typer(add_completion=False)


def parse_float_triplet(value: str) -> Tuple[float, float, float]:
    try:
        parts = [float(p.strip()) for p in value.split(",")]
    except ValueError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter('Expected three comma-separated floats between 0 and 1, e.g. "0.82,0.88,0.96"') from exc
    if len(parts) != 3:
        raise typer.BadParameter('Expected three comma-separated floats between 0 and 1, e.g. "0.82,0.88,0.96"')
    clamped = tuple(max(0.0, min(1.0, p)) for p in parts)
    clamped_tuple: Tuple[float, float, float] = (clamped[0], clamped[1], clamped[2])
    return clamped_tuple

@app.command()
def main(
    input: str = typer.Option(..., help="Glob of input images, e.g. './drafts/*.png'"),
    out: str = typer.Option("./final", help="Output folder"),
    prompt: str = typer.Option(..., help="Positive prompt (style, materials, time of day, lens)"),
    neg: str = typer.Option("lowres, bad geometry, blurry, text, watermark, logo", help="Negative prompt"),
    # Render config
    width: int = typer.Option(1024, min=256),
    height: int = typer.Option(768, min=256),
    steps: int = typer.Option(30, min=10),
    gs: float = typer.Option(7.5, help="Guidance scale"),
    strength: float = typer.Option(0.5, help="How far to move from init (0.3–0.6 preserves structure)"),
    seed: int = typer.Option(1234),
    # Models
    base_model: str = typer.Option("runwayml/stable-diffusion-v1-5"),
    controlnet_canny: str = typer.Option("lllyasviel/sd-controlnet-canny"),
    controlnet_depth: str = typer.Option("lllyasviel/sd-controlnet-depth"),
    upscaler_id: str = typer.Option("stabilityai/sd-x2-latent-upscaler"),
    refiner: Optional[str] = typer.Option(None, help="Only for SDXL (optional)"),
    # Finishing
    no_aces: bool = typer.Option(False, help="Disable ACES tonemap"),
    no_wb: bool = typer.Option(False, help="Disable white balance"),
    no_bloom: bool = typer.Option(False, help="Disable bloom"),
    no_vignette: bool = typer.Option(False, help="Disable vignette"),
    no_grain: bool = typer.Option(False, help="Disable grain"),
    no_depth: bool = typer.Option(False, help="Disable depth guidance (ControlNet depth)"),
    material_response: bool = typer.Option(False, help="Enable texture + atmosphere finishing enhancements"),
    texture_boost: float = typer.Option(0.25, help="Material response: detail boost strength"),
    ambient_occlusion: float = typer.Option(0.12, help="Material response: contact shadow intensity"),
    highlight_warmth: float = typer.Option(0.08, help="Material response: warm highlight mix"),
    haze_strength: float = typer.Option(0.06, help="Material response: volumetric haze blend"),
    haze_tint: str = typer.Option("0.82,0.88,0.96", help="Material response: haze tint (r,g,b in 0-1)"),
    floor_plank_contrast: float = typer.Option(0.12, help="Material response: enhance floor plank definition"),
    floor_specular: float = typer.Option(0.18, help="Material response: specular streak intensity on flooring"),
    textile_contrast: float = typer.Option(0.18, help="Material response: linen/fabric separation"),
    leather_sheen: float = typer.Option(0.16, help="Material response: leather sheen blend"),
    fireplace_glow: float = typer.Option(0.18, help="Material response: fireplace spill intensity"),
    fireplace_glow_radius: float = typer.Option(45.0, help="Material response: fireplace glow falloff radius"),
    window_reflection: float = typer.Option(0.12, help="Material response: window reflection spill on flooring"),
    bedding_relief: float = typer.Option(0.16, help="Material response: bedding wrinkle relief and occlusion"),
    wall_texture: float = typer.Option(0.1, help="Material response: wall microtexture strength"),
    painting_integration: float = typer.Option(0.1, help="Material response: wall art integration and rim light"),
    window_light_wrap: float = typer.Option(0.14, help="Material response: window light wrap onto interior surfaces"),
    exterior_atmosphere: float = typer.Option(0.12, help="Material response: exterior haze and atmospheric blend"),
    # Branding
    logo: Optional[str] = typer.Option(None, help="Path to PNG/SVG logo (PNG w/ alpha recommended)"),
    brand_text: Optional[str] = typer.Option(None, help="Caption, e.g. 'The Veridian | Penthouse 21B'"),
    # Export/resolution
    w4k: bool = typer.Option(False, help="Export ~4K. Uses latent + optional Real-ESRGAN."),
    use_realesrgan: bool = typer.Option(False, help="Use Real-ESRGAN (requires weights, GPU recommended)"),
):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ids = ModelIDs(
        base_model=base_model,
        controlnet_canny=controlnet_canny,
        controlnet_depth=controlnet_depth,
        upscaler_id=upscaler_id,
        refiner=refiner,
    )
    cfg = RenderConfig(width=width, height=height, steps=steps, guidance_scale=gs, strength=strength, seed=seed)
    finish = FinishConfig(
        aces=not no_aces,
        white_balance=not no_wb,
        bloom=not no_bloom,
        vignette=not no_vignette,
        grain=not no_grain,
        material_response=material_response,
        texture_boost=texture_boost,
        ambient_occlusion=ambient_occlusion,
        highlight_warmth=highlight_warmth,
        haze_strength=haze_strength,
        haze_tint=parse_float_triplet(haze_tint),
        floor_plank_contrast=floor_plank_contrast,
        floor_specular=floor_specular,
        textile_contrast=textile_contrast,
        leather_sheen=leather_sheen,
        fireplace_glow=fireplace_glow,
        fireplace_glow_radius=fireplace_glow_radius,
        window_reflection=window_reflection,
        bedding_relief=bedding_relief,
        wall_texture=wall_texture,
        painting_integration=painting_integration,
        window_light_wrap=window_light_wrap,
        exterior_atmosphere=exterior_atmosphere,
    )

    pipe = LuxuryRenderPipeline(
        model_ids,
        use_realesrgan=use_realesrgan,
        use_depth=not no_depth,
    )

    files = sorted([p for g in input.split(",") for p in glob.glob(g.strip())])
    if not files:
        raise SystemExit(f"No files matched: {input}")

    print(f"[Config] {asdict(cfg)}")
    print(f"[Models] {asdict(model_ids)}")
    print(f"[Finishing] {asdict(finish)}")
    print(f"[Batch] {len(files)} file(s)")

    for i, f in enumerate(files, 1):
        try:
            print(f"\n[{i}/{len(files)}] {f}")
            img = load_image(f)
            out_img = pipe.enhance(
                init_image=img,
                prompt=prompt,
                negative_prompt=neg,
                cfg=cfg,
                finish=finish,
                brand_logo=logo,
                brand_text=brand_text,
                export_4k=w4k,
            )
            fname = Path(f).stem + "_lux.png"
            save_image(out_img, out_dir / fname)
            print(f"[Saved] {out_dir / fname}")
        except Exception as e:
            print(f"[Error] {f}: {e}")

if __name__ == "__main__":
    app()
