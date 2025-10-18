“”“Generate a visualization showing MBAR material assignments for aerial imagery.

This script performs K-means clustering on aerial images, assigns materials to
each cluster based on predefined rules, and creates a color-coded visualization
with a detailed legend.

Usage:
python visualize_material_assignments.py –input IMAGE –output OUTPUT [OPTIONS]

Examples:
# Basic usage
python visualize_material_assignments.py –input aerial.tiff –output viz.jpg

```
# Load existing palette
python visualize_material_assignments.py --input aerial.tiff --output viz.jpg --palette palette.json

# Save palette for reuse
python visualize_material_assignments.py --input aerial.tiff --output viz.jpg --save-palette palette.json

# Custom clustering
python visualize_material_assignments.py --input aerial.tiff --output viz.jpg --clusters 12 --seed 42
```

“””

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from board_material_aerial_enhancer import (
DEFAULT_TEXTURES,
_assign_full_image,
_cluster_stats,
_downsample_image,
_kmeans,
assign_materials,
build_material_rules,
load_palette_assignments,
save_palette_assignments,
)

# Configuration constants

DEFAULT_CONFIG = {
“analysis_width”: 1280,
“max_sample_size”: 200_000,
“num_clusters”: 8,
“random_seed”: 42,
“legend_height”: 400,
“legend_margin”: 20,
“legend_x_offset”: 40,
“legend_y_start”: 40,
“color_box_size”: 30,
“row_spacing”: 45,
“jpeg_quality”: 95,
“font_size”: 14,
}

# Color palette for cluster visualization

CLUSTER_COLORS: List[Tuple[int, int, int]] = [
(255, 100, 100),  # Red
(100, 255, 100),  # Green
(100, 100, 255),  # Blue
(255, 255, 100),  # Yellow
(255, 100, 255),  # Magenta
(100, 255, 255),  # Cyan
(255, 200, 100),  # Orange
(200, 100, 255),  # Purple
(150, 75, 0),     # Brown
(255, 192, 203),  # Pink
(128, 128, 0),    # Olive
(0, 128, 128),    # Teal
]

class MaterialVisualizationError(Exception):
“”“Custom exception for visualization errors.”””
pass

def parse_arguments() -> argparse.Namespace:
“”“Parse command-line arguments.

```
Returns:
    Parsed arguments namespace.
"""
parser = argparse.ArgumentParser(
    description="Visualize MBAR material assignments for aerial imagery",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)

# Required arguments
parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Path to input aerial image (TIFF, PNG, or JPG)"
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to save visualization output (JPG recommended)"
)

# Optional palette arguments
parser.add_argument(
    "--palette",
    type=Path,
    help="Load existing material palette from JSON file"
)
parser.add_argument(
    "--save-palette",
    type=Path,
    help="Save computed material palette to JSON file"
)

# Clustering parameters
parser.add_argument(
    "--clusters",
    type=int,
    default=DEFAULT_CONFIG["num_clusters"],
    help=f"Number of K-means clusters (default: {DEFAULT_CONFIG['num_clusters']})"
)
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_CONFIG["random_seed"],
    help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG['random_seed']})"
)
parser.add_argument(
    "--analysis-width",
    type=int,
    default=DEFAULT_CONFIG["analysis_width"],
    help=f"Width for downsampled analysis (default: {DEFAULT_CONFIG['analysis_width']})"
)
parser.add_argument(
    "--sample-size",
    type=int,
    default=DEFAULT_CONFIG["max_sample_size"],
    help=f"Max pixels to sample for clustering (default: {DEFAULT_CONFIG['max_sample_size']})"
)

# Visualization parameters
parser.add_argument(
    "--quality",
    type=int,
    default=DEFAULT_CONFIG["jpeg_quality"],
    choices=range(1, 101),
    metavar="[1-100]",
    help=f"JPEG quality for output (default: {DEFAULT_CONFIG['jpeg_quality']})"
)

return parser.parse_args()
```

def validate_inputs(args: argparse.Namespace) -> None:
“”“Validate input arguments and file paths.

```
Args:
    args: Parsed command-line arguments.
    
Raises:
    MaterialVisualizationError: If validation fails.
"""
# Check input file exists
if not args.input.exists():
    raise MaterialVisualizationError(f"Input image not found: {args.input}")

# Check input is an image file
valid_extensions = {".tiff", ".tif", ".png", ".jpg", ".jpeg"}
if args.input.suffix.lower() not in valid_extensions:
    raise MaterialVisualizationError(
        f"Input must be an image file. Got: {args.input.suffix}"
    )

# Check palette file exists if specified
if args.palette and not args.palette.exists():
    raise MaterialVisualizationError(f"Palette file not found: {args.palette}")

# Ensure output directory exists
args.output.parent.mkdir(parents=True, exist_ok=True)

# Validate cluster count
if args.clusters > len(CLUSTER_COLORS):
    raise MaterialVisualizationError(
        f"Too many clusters ({args.clusters}). Maximum supported: {len(CLUSTER_COLORS)}"
    )
if args.clusters < 2:
    raise MaterialVisualizationError(
        f"Too few clusters ({args.clusters}). Minimum: 2"
    )
```

def load_and_prepare_image(input_path: Path) -> Tuple[np.ndarray, Image.Image]:
“”“Load input image and prepare arrays.

```
Args:
    input_path: Path to input image.
    
Returns:
    Tuple of (base_array, PIL Image object).
    
Raises:
    MaterialVisualizationError: If image cannot be loaded.
"""
try:
    image = Image.open(input_path).convert("RGB")
    base_array = np.asarray(image, dtype=np.float32) / 255.0
    return base_array, image
except Exception as e:
    raise MaterialVisualizationError(f"Failed to load image: {e}") from e
```

def perform_clustering(
image: Image.Image,
base_array: np.ndarray,
num_clusters: int,
analysis_width: int,
max_sample_size: int,
random_seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
“”“Perform K-means clustering on the image.

```
Args:
    image: PIL Image object.
    base_array: Full resolution image array.
    num_clusters: Number of clusters.
    analysis_width: Width for downsampled analysis.
    max_sample_size: Maximum pixels to sample.
    random_seed: Random seed for reproducibility.
    
Returns:
    Tuple of (labels array, centroids, cluster statistics).
"""
print(f"Performing K-means clustering with {num_clusters} clusters...")

# Downsample for faster processing
analysis_image = _downsample_image(image, analysis_width)
analysis_array = np.asarray(analysis_image, dtype=np.float32) / 255.0
pixels = analysis_array.reshape(-1, 3)

# Sample pixels if necessary
rng = np.random.default_rng(random_seed)
sample_size = min(len(pixels), max_sample_size)

if sample_size < len(pixels):
    indices = rng.choice(len(pixels), size=sample_size, replace=False)
    sample = pixels[indices]
    print(f"  Sampled {sample_size:,} of {len(pixels):,} pixels")
else:
    sample = pixels
    print(f"  Using all {len(pixels):,} pixels")

# Perform clustering
centroids = _kmeans(sample, num_clusters, rng)

# Assign clusters to full image
labels_small = _assign_full_image(analysis_array, centroids)
labels_small_img = Image.fromarray(labels_small.astype("uint8"))
labels_small_img = labels_small_img.convert("L")
labels_full = labels_small_img.resize(image.size, Image.Resampling.NEAREST)
labels = np.asarray(labels_full, dtype=np.uint8)

# Calculate cluster statistics
stats = _cluster_stats(base_array, labels)

print("  ✓ Clustering complete")
return labels, centroids, stats
```

def get_material_assignments(
stats: Dict,
palette_path: Path = None,
save_palette_path: Path = None
) -> Dict:
“”“Get or compute material assignments.

```
Args:
    stats: Cluster statistics.
    palette_path: Optional path to load existing palette.
    save_palette_path: Optional path to save computed palette.
    
Returns:
    Dictionary mapping cluster labels to material rules.
"""
rules = build_material_rules(DEFAULT_TEXTURES)

# Load or compute assignments
if palette_path and palette_path.exists():
    print(f"Loading palette from: {palette_path}")
    assignments = load_palette_assignments(palette_path, rules)
else:
    print("Computing material assignments...")
    assignments = assign_materials(stats, rules)

# Optionally save assignments
if save_palette_path:
    print(f"Saving palette to: {save_palette_path}")
    save_palette_assignments(assignments, save_palette_path)

return assignments
```

def load_font(size: int = DEFAULT_CONFIG[“font_size”]) -> ImageFont.FreeTypeFont:
“”“Load a font for drawing text.

```
Args:
    size: Font size in points.
    
Returns:
    Font object (falls back to default if TrueType not available).
"""
try:
    # Try common font names across platforms
    font_names = ["Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    # If no TrueType fonts found, use default
    return ImageFont.load_default()
except Exception:
    return ImageFont.load_default()
```

def create_visualization(
labels: np.ndarray,
assignments: Dict,
num_clusters: int,
output_path: Path,
quality: int
) -> None:
“”“Create and save the material assignment visualization.

```
Args:
    labels: Cluster label array.
    assignments: Material assignments dictionary.
    num_clusters: Total number of clusters.
    output_path: Path to save visualization.
    quality: JPEG quality setting.
"""
print("Creating visualization...")

# Get colors for the actual number of clusters
colors = CLUSTER_COLORS[:num_clusters]

# Create color-coded visualization
viz_array = np.zeros((*labels.shape, 3), dtype=np.uint8)
for label in range(num_clusters):
    mask = labels == label
    viz_array[mask] = colors[label]

viz_img = Image.fromarray(viz_array)

# Calculate legend height based on content
config = DEFAULT_CONFIG
rows_needed = len(assignments) + len([i for i in range(num_clusters) if i not in assignments])
legend_height = config["legend_height"] + (rows_needed * 10)  # Dynamic sizing

# Create canvas with legend space
legend_img = Image.new(
    "RGB",
    (viz_img.width, viz_img.height + legend_height),
    (255, 255, 255)
)
legend_img.paste(viz_img, (0, 0))

# Setup drawing
draw = ImageDraw.Draw(legend_img)
font = load_font(config["font_size"])

# Draw legend header
y_offset = viz_img.height + config["legend_margin"]
x_offset = config["legend_x_offset"]

draw.text(
    (x_offset, y_offset),
    "MBAR MATERIAL ASSIGNMENTS:",
    fill=(0, 0, 0),
    font=font
)
y_offset += config["legend_y_start"]

# Draw assigned materials
box_size = config["color_box_size"]
for label, rule in sorted(assignments.items(), key=lambda x: x[0]):
    # Color box
    draw.rectangle(
        [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
        fill=colors[label],
        outline=(0, 0, 0),
        width=2
    )
    
    # Material name with cluster info
    cluster_pixels = (labels == label).sum()
    percentage = (cluster_pixels / labels.size) * 100
    text = f"{rule.name.upper()} - Cluster {label} ({percentage:.1f}%)"
    draw.text(
        (x_offset + box_size + 15, y_offset + 5),
        text,
        fill=(0, 0, 0),
        font=font
    )
    y_offset += config["row_spacing"]

# Draw unassigned clusters
unassigned = [i for i in range(num_clusters) if i not in assignments]
if unassigned:
    y_offset += 10
    draw.text(
        (x_offset, y_offset),
        "UNASSIGNED CLUSTERS:",
        fill=(128, 128, 128),
        font=font
    )
    y_offset += 35
    
    for label in unassigned:
        draw.rectangle(
            [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
            fill=colors[label],
            outline=(0, 0, 0),
            width=2
        )
        
        cluster_pixels = (labels == label).sum()
        percentage = (cluster_pixels / labels.size) * 100
        text = f"Cluster {label} ({percentage:.1f}%) - no material match"
        draw.text(
            (x_offset + box_size + 15, y_offset + 5),
            text,
            fill=(128, 128, 128),
            font=font
        )
        y_offset += config["row_spacing"]

# Save visualization
legend_img.save(output_path, quality=quality)
print(f"  ✓ Visualization saved to: {output_path}")
```

def print_summary(labels: np.ndarray, assignments: Dict, num_clusters: int) -> None:
“”“Print summary statistics to console.

```
Args:
    labels: Cluster label array.
    assignments: Material assignments dictionary.
    num_clusters: Total number of clusters.
"""
print("\n" + "="*60)
print("MATERIAL ASSIGNMENT SUMMARY")
print("="*60)

print("\nAssigned Materials:")
for label, rule in sorted(assignments.items(), key=lambda x: x[0]):
    cluster_pixels = (labels == label).sum()
    percentage = (cluster_pixels / labels.size) * 100
    print(f"  • {rule.name.upper():20s} : Cluster {label} ({percentage:5.1f}%)")

unassigned = [i for i in range(num_clusters) if i not in assignments]
if unassigned:
    print(f"\nUnassigned Clusters: {len(unassigned)}")
    for label in unassigned:
        cluster_pixels = (labels == label).sum()
        percentage = (cluster_pixels / labels.size) * 100
        print(f"  • Cluster {label:2d} : {percentage:5.1f}% (below threshold)")

print("\n" + "="*60)
```

def main() -> int:
“”“Main execution function.

```
Returns:
    Exit code (0 for success, 1 for error).
"""
try:
    # Parse and validate arguments
    args = parse_arguments()
    validate_inputs(args)
    
    print("="*60)
    print("MBAR MATERIAL ASSIGNMENT VISUALIZATION")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Clusters: {args.clusters}, Seed: {args.seed}")
    print("="*60 + "\n")
    
    # Load image
    print("Loading image...")
    base_array, image = load_and_prepare_image(args.input)
    print(f"  Image size: {image.size[0]} x {image.size[1]} pixels")
    
    # Perform clustering
    labels, centroids, stats = perform_clustering(
        image,
        base_array,
        args.clusters,
        args.analysis_width,
        args.sample_size,
        args.seed
    )
    
    # Get material assignments
    assignments = get_material_assignments(
        stats,
        args.palette,
        args.save_palette
    )
    
    # Create visualization
    create_visualization(
        labels,
        assignments,
        args.clusters,
        args.output,
        args.quality
    )
    
    # Print summary
    print_summary(labels, assignments, args.clusters)
    
    print("\n✅ Processing complete!")
    return 0
    
except MaterialVisualizationError as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    return 1
except KeyboardInterrupt:
    print("\n\n⚠️  Process interrupted by user", file=sys.stderr)
    return 1
except Exception as e:
    print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    return 1
```

if **name** == “**main**”:
sys.exit(main())