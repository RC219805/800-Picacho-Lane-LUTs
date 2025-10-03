#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luxury Real Estate Render Refinement Pipeline
---------------------------------------------
Transforms basic 3D model images into high-resolution marketing visuals by:
- Preprocessing (Canny + Depth)
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
        --use_realesrgan
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
    ):
        self.model_ids = model_ids
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32

        # Load ControlNets
        print("[Load] ControlNets...")
        self.cn_canny = ControlNetModel.from_pretrained(model_ids.controlnet_canny, torch_dtype=self.dtype)
        self.cn_depth = ControlNetModel.from_pretrained(model_ids.controlnet_depth, torch_dtype=self.dtype)

        # Pick pipeline type based on base model family (SD1.5 vs SDXL)
        is_sdxl = "xl" in model_ids.base_model.lower()

        if is_sdxl:
            if StableDiffusionXLControlNetPipeline is None:
                raise RuntimeError("diffusers SDXL pipeline not available; upgrade diffusers or install extras.")
            print("[Load] SDXL + ControlNet pipeline...")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_ids.base_model,
                controlnet=[self.cn_canny, self.cn_depth],
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
                controlnet=[self.cn_canny, self.cn_depth],
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
        self.pre = Preprocessor()

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

        # 2) Diffusion pass (structure-preserving)
        g = seed_all(cfg.seed)
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=[canny, depth],
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
        out = np_to_pil(rgb)

        # 7) Branding
        out = overlay_logo_caption(out, brand_logo, brand_text)

        return out

# --------------------------
# CLI
# --------------------------
import typer
app = typer.Typer(add_completion=False)

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
    )

    pipe = LuxuryRenderPipeline(model_ids, use_realesrgan=use_realesrgan)

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
