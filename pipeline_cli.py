#!/usr/bin/env python3
"""Material Intelligence – Unified CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "pipeline_config.json"


def _load_default_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Failed to parse config file {path}: {exc}") from exc


def _import_substrate(use_enhanced: bool):
    if use_enhanced:
        try:
            import material_property_substrate_enhanced as substrate

            return substrate
        except Exception:
            print("[warn] Enhanced substrate requested but unavailable; using baseline.")
    import material_property_substrate as substrate

    return substrate


def _inject_enhanced_into_master(use_enhanced: bool) -> None:
    if not use_enhanced:
        return
    try:
        import material_property_substrate_enhanced as enhanced

        sys.modules["material_property_substrate"] = enhanced
        print("[info] Using enhanced material substrate inside master pipeline.")
    except Exception:
        print("[warn] Could not load enhanced substrate; master will use baseline.")


def _resolve_default(path: str | None, default: str | Path | None) -> Path | None:
    if path:
        return Path(path)
    return Path(default) if default else None


def _configure_defaults(args: argparse.Namespace) -> Dict[str, Any]:
    config = _load_default_config()
    defaults = config.get("defaults", {})
    pipeline_defaults: Dict[str, Any] = {}
    pipeline_defaults["baseline_id_mask"] = _resolve_default(
        getattr(args, "baseline_id_mask", None), defaults.get("baseline_id_mask")
    )
    pipeline_defaults["baseline_palette"] = _resolve_default(
        getattr(args, "baseline_palette", None), defaults.get("baseline_palette")
    )
    pipeline_defaults["output_dir"] = _resolve_default(
        getattr(args, "output_dir", None), defaults.get("output_dir", THIS_DIR / "pipeline_outputs")
    )
    pipeline_defaults["pixel_size"] = getattr(args, "pixel_size", defaults.get("pixel_size_meters", 0.01))
    opt_defaults = defaults.get("opt", {})
    pipeline_defaults["generations"] = getattr(args, "generations", opt_defaults.get("generations", 100))
    pipeline_defaults["population"] = getattr(args, "population", opt_defaults.get("population_size", 50))
    weights = opt_defaults.get("weights", {})
    pipeline_defaults["weights"] = {
        "cost": getattr(args, "weight_cost", weights.get("cost", 1.0)),
        "carbon": getattr(args, "weight_carbon", weights.get("carbon", 0.8)),
        "thermal": getattr(args, "weight_thermal", weights.get("thermal", 0.5)),
        "aesthetic": getattr(args, "weight_aesthetic", weights.get("aesthetic", 0.6)),
    }
    pipeline_defaults["target_conductivity"] = getattr(
        args,
        "target_conductivity",
        opt_defaults.get("target_conductivity", 0.5),
    )
    return pipeline_defaults


def cmd_run(args: argparse.Namespace) -> None:
    _inject_enhanced_into_master(args.use_enhanced)
    import master_intelligence_pipeline as master

    defaults = _configure_defaults(args)
    id_mask = defaults["baseline_id_mask"]
    palette = defaults["baseline_palette"]
    output_dir = defaults["output_dir"]

    if id_mask is None or palette is None:
        raise SystemExit("ID mask and palette paths must be provided.")

    print("[info] Running complete pipeline …")
    summary = master.run_complete_pipeline(
        id_mask,
        palette,
        output_dir,
        pixel_size_meters=defaults["pixel_size"],
        weights=defaults["weights"],
        generations=defaults["generations"],
        population=defaults["population"],
        target_conductivity=defaults["target_conductivity"],
    )
    print(json.dumps(summary, indent=2))


def _ensure_tifffile() -> None:
    try:
        import tifffile  # noqa: F401
    except ImportError:  # pragma: no cover - convenience path
        print("[info] Installing dependency 'tifffile' …")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile", "--break-system-packages"])


def cmd_tensor(args: argparse.Namespace) -> None:
    substrate = _import_substrate(args.use_enhanced)
    _ensure_tifffile()
    import tifffile

    id_mask_path = Path(args.id_mask)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Loading ID mask from {id_mask_path}")
    id_mask = tifffile.imread(str(id_mask_path))
    tensor = substrate.create_property_tensor(
        id_mask,
        substrate.MATERIAL_PHYSICS_DB,
        pixel_size_meters=args.pixel_size,
    )
    archive = substrate.save_property_tensor(tensor, output_dir)
    print(f"[done] Tensor saved to {archive}")


def _build_objectives_from_args(args: argparse.Namespace, baseline_tensor: Mapping[str, Any]):
    import material_optimizer as opt

    return [
        opt.create_cost_minimization_objective(weight=args.weight_cost),
        opt.create_carbon_minimization_objective(weight=args.weight_carbon),
        opt.create_thermal_performance_objective(
            target_conductivity=args.target_conductivity,
            weight=args.weight_thermal,
        ),
        opt.create_aesthetic_consistency_objective(
            baseline_tensor,
            weight=args.weight_aesthetic,
        ),
    ]


def cmd_optimize(args: argparse.Namespace) -> None:
    _inject_enhanced_into_master(args.use_enhanced)
    import master_intelligence_pipeline as master
    import material_property_substrate as substrate

    id_mask_path = Path(args.baseline_id_mask)
    palette_path = Path(args.baseline_palette)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_labels, assignments, palette_data = master.load_baseline_configuration(
        id_mask_path,
        palette_path,
    )

    baseline_tensor = substrate.create_property_tensor(
        cluster_labels,
        substrate.MATERIAL_PHYSICS_DB,
        pixel_size_meters=args.pixel_size,
    )

    import material_optimizer as opt

    objectives = _build_objectives_from_args(args, baseline_tensor)
    optimiser = opt.MaterialOptimizer(
        cluster_labels=cluster_labels,
        baseline_assignments=assignments,
        pixel_size_meters=args.pixel_size,
    )
    print("[info] Running optimisation …")
    result = optimiser.optimize(
        objectives=objectives,
        population_size=args.population,
        generations=args.generations,
        baseline_tensor_stats=substrate.summarise_tensor(baseline_tensor),
    )

    report_path = output_dir / "optimization_report.json"
    opt.export_optimization_report(result, report_path)
    print(f"[done] Optimisation report written to {report_path}")
    (output_dir / "palette_snapshot.json").write_text(json.dumps(palette_data, indent=2), encoding="utf-8")


def cmd_render(args: argparse.Namespace) -> None:
    substrate = _import_substrate(args.use_enhanced)
    _ensure_tifffile()
    import tifffile
    import phenomenological_rendering as phr

    id_mask = tifffile.imread(str(args.id_mask))
    tensor = substrate.create_property_tensor(
        id_mask,
        substrate.MATERIAL_PHYSICS_DB,
        pixel_size_meters=args.pixel_size,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios: Sequence[str] | None = args.scenarios
    phr.render_lighting_comparison(tensor, out_dir, scenarios=scenarios)
    print(f"[done] Renders saved to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for the Material Intelligence pipeline",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the complete pipeline")
    pr.add_argument("--baseline-id-mask", type=str, help="Path to 32-bit ID mask .tiff")
    pr.add_argument("--baseline-palette", type=str, help="Path to palette .json")
    pr.add_argument("--output-dir", type=str, help="Directory for pipeline outputs")
    pr.add_argument("--use-enhanced", action="store_true", help="Use the enhanced substrate when available")
    pr.set_defaults(func=cmd_run)

    pt = sub.add_parser("tensor", help="Create and save property tensor from an ID mask")
    pt.add_argument("--id-mask", type=str, required=True, help="Path to 32-bit ID mask .tiff")
    pt.add_argument("--output-dir", type=str, required=True, help="Directory for tensor artefacts")
    pt.add_argument("--pixel-size", type=float, default=0.01, help="Meters per pixel (default: 0.01)")
    pt.add_argument("--use-enhanced", action="store_true", help="Use the enhanced substrate when available")
    pt.set_defaults(func=cmd_tensor)

    po = sub.add_parser("optimize", help="Run the optimiser only and write a JSON report")
    po.add_argument("--baseline-id-mask", type=str, required=True, help="Path to 32-bit ID mask .tiff")
    po.add_argument("--baseline-palette", type=str, required=True, help="Path to material palette .json")
    po.add_argument("--output-dir", type=str, required=True, help="Directory to write outputs")
    po.add_argument("--generations", type=int, default=100, help="Number of GA generations (default: 100)")
    po.add_argument("--population", type=int, default=50, help="Population size (default: 50)")
    po.add_argument("--pixel-size", type=float, default=0.01, help="Meters per pixel (default: 0.01)")
    po.add_argument("--weight-cost", type=float, default=1.0, help="Objective weight: cost (default: 1.0)")
    po.add_argument("--weight-carbon", type=float, default=0.8, help="Objective weight: carbon (default: 0.8)")
    po.add_argument("--weight-thermal", type=float, default=0.5, help="Objective weight: thermal (default: 0.5)")
    po.add_argument("--weight-aesthetic", type=float, default=0.6, help="Objective weight: aesthetic (default: 0.6)")
    po.add_argument("--target-conductivity", type=float, default=0.5, help="Thermal target for conductivity W/(m·K) (default: 0.5)")
    po.add_argument("--use-enhanced", action="store_true", help="Use the enhanced substrate when available")
    po.set_defaults(func=cmd_optimize)

    prd = sub.add_parser("render", help="Render a baseline tensor under lighting scenarios")
    prd.add_argument("--id-mask", type=str, required=True, help="Path to 32-bit ID mask .tiff")
    prd.add_argument("--output-dir", type=str, required=True, help="Directory to write images")
    prd.add_argument("--scenarios", nargs="*", help="Lighting scenarios (default: golden_hour noon overcast dusk)")
    prd.add_argument("--pixel-size", type=float, default=0.01, help="Meters per pixel (default: 0.01)")
    prd.add_argument("--use-enhanced", action="store_true", help="Use the enhanced substrate when available")
    prd.set_defaults(func=cmd_render)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
