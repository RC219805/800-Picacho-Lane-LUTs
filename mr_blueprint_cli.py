# file: mr_blueprint_cli.py
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Dict, List, Tuple


# --- import helpers ---------------------------------------------------------
def _import_blueprint_module(name: str | None):
    """Import the user's blueprint module. Try common defaults if name is None."""
    candidates = [name] if name else ["blueprint", "material_blueprint", "mr_blueprint", "material_response_blueprint"]
    last_err: Exception | None = None
    for mod in candidates:
        try:
            return importlib.import_module(mod)
        except Exception as e:  # why: we want a single clean error after attempts
            last_err = e
    raise ImportError(
        f"Failed to import blueprint module. Tried {candidates}. "
        "Use --module to specify the correct module name."
    ) from last_err


# --- validation -------------------------------------------------------------
def _has_metric(scene_report, version: str, metric: str) -> bool:
    try:
        _ = scene_report.metric(version, metric)
        return True
    except Exception:
        return False


def validate_report(report) -> Dict[str, Any]:
    """Return validation summary for the given MaterialResponseReport."""
    errors: List[str] = []
    warnings: List[str] = []

    # Required by planner logic
    required_scene_for_awe = "kitchen"

    all_scenes = list(report.scenes.keys())
    if required_scene_for_awe not in report.scenes:
        errors.append(f"Missing required scene '{required_scene_for_awe}' used for awe benchmarking.")

    # For every scene present, these versions/metrics are expected by various steps.
    reg_metrics = {"luminance", "awe", "comfort", "texture_dimension", "future_alignment", "luxury_index"}
    lux_metrics = {"luxury_index"}

    for scene_name, scene in report.scenes.items():
        # regular
        for m in reg_metrics:
            if not _has_metric(scene, "regular", m):
                errors.append(f"Scene '{scene_name}' missing regular.{m}")

        # lux (needed for delta comparisons)
        for m in lux_metrics:
            if not _has_metric(scene, "lux", m):
                errors.append(f"Scene '{scene_name}' missing lux.{m}")

    # Optional scenes referenced conditionally; warn (not error) if absent.
    for optional in ("great_room", "pool", "aerial", "primary_bedroom"):
        if optional not in report.scenes:
            warnings.append(f"Optional scene '{optional}' not found; related actions will be omitted.")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "scenes": all_scenes}


# --- example schema ---------------------------------------------------------
def example_report() -> Dict[str, Any]:
    return {
        "generated": "2025-10-12T18:22:03Z",
        "analysis_version": "1.3.0",
        "scenes": [
            {
                "name": "kitchen",
                "versions": {
                    "regular": {
                        "luminance": 0.29,
                        "awe": 0.81,
                        "comfort": 0.62,
                        "texture_dimension": 2.02,
                        "future_alignment": 0.68,
                        "luxury_index": 0.66,
                    },
                    "lux": {"luxury_index": 0.72},
                },
            },
            {
                "name": "great_room",
                "versions": {
                    "regular": {
                        "luminance": 0.28,
                        "awe": 0.74,
                        "comfort": 0.57,
                        "texture_dimension": 1.95,
                        "future_alignment": 0.65,
                        "luxury_index": 0.62,
                    },
                    "lux": {"luxury_index": 0.66},
                },
            },
            {
                "name": "pool",
                "versions": {
                    "regular": {
                        "luminance": 0.27,
                        "awe": 0.7,
                        "comfort": 0.58,
                        "texture_dimension": 1.88,
                        "future_alignment": 0.63,
                        "luxury_index": 0.6,
                    },
                    "lux": {"luxury_index": 0.65},
                },
            },
            {
                "name": "aerial",
                "versions": {
                    "regular": {
                        "luminance": 0.26,
                        "awe": 0.69,
                        "comfort": 0.55,
                        "texture_dimension": 1.86,
                        "future_alignment": 0.6,
                        "luxury_index": 0.58,
                    },
                    "lux": {"luxury_index": 0.62},
                },
            },
            {
                "name": "primary_bedroom",
                "versions": {
                    "regular": {
                        "luminance": 0.31,
                        "awe": 0.64,
                        "comfort": 0.78,
                        "texture_dimension": 1.92,
                        "future_alignment": 0.67,
                        "luxury_index": 0.64,
                    },
                    "lux": {"luxury_index": 0.7},
                },
            },
        ],
    }


# --- I/O helpers ------------------------------------------------------------
def _read_json(path: str) -> Dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict[str, Any], pretty: bool) -> None:
    text = json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)
    if path == "-":
        sys.stdout.write(text + ("\n" if pretty else ""))
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


# --- main -------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mr-blueprint",
        description="Generate Material Response enhancement blueprint from a report JSON."
    )
    p.add_argument("-i", "--input", required=True, help="Path to material_response_report.json (or '-' for stdin).")
    p.add_argument("-o", "--output", default="-", help="Output path for blueprint JSON (default: stdout).")
    p.add_argument("--module", default=None, help="Python module name containing planners (default: try 'blueprint', etc.).")
    p.add_argument(
        "--planner",
        choices=["base", "aware"],
        default="aware",
        help="Use base RenderEnhancementPlanner or MaterialAwareEnhancementPlanner (default: aware).",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    p.add_argument("--validate-only", action="store_true", help="Validate input and exit without generating blueprint.")
    p.add_argument("--ignore-validation-errors", action="store_true", help="Proceed even if validation fails.")
    p.add_argument("--print-example", action="store_true", help="Print an example input schema and exit.")
    args = p.parse_args(argv)

    if args.print_example:
        _write_json("-", example_report(), pretty=True)
        return 0

    try:
        mod = _import_blueprint_module(args.module)
    except Exception as e:
        _write_json(args.output, {"ok": False, "error": f"Import error: {e}"}, pretty=args.pretty)
        return 1

    # Pull required types from the user module.
    try:
        MaterialResponseReport = getattr(mod, "MaterialResponseReport")
        RenderEnhancementPlanner = getattr(mod, "RenderEnhancementPlanner")
        MaterialAwareEnhancementPlanner = getattr(mod, "MaterialAwareEnhancementPlanner")
    except AttributeError as e:
        _write_json(args.output, {"ok": False, "error": f"Module missing expected exports: {e}"}, pretty=args.pretty)
        return 1

    # Load report
    try:
        raw = _read_json(args.input)
        report = MaterialResponseReport(
            generated=str(raw.get("generated", "")),
            analysis_version=str(raw.get("analysis_version", "")),
            scenes={
                scene["name"]: getattr(mod, "SceneReport").from_mapping(scene)
                for scene in raw.get("scenes", [])
            },
        )
    except Exception as e:
        _write_json(args.output, {"ok": False, "error": f"Failed to parse input report: {e}"}, pretty=args.pretty)
        return 1

    # Validate
    v = validate_report(report)
    if args.validate_only:
        _write_json(args.output, {"ok": v["ok"], "validation": v, "scenes": v["scenes"]}, pretty=args.pretty)
        return 0 if v["ok"] else 2

    if not v["ok"] and not args.ignore_validation_errors:
        _write_json(
            args.output,
            {
                "ok": False,
                "error": "Validation failed; fix input or pass --ignore-validation-errors.",
                "validation": v,
            },
            pretty=args.pretty,
        )
        return 2

    # Planner selection
    Planner = MaterialAwareEnhancementPlanner if args.planner == "aware" else RenderEnhancementPlanner
    try:
        blueprint = Planner(report).build_blueprint()
    except Exception as e:
        _write_json(
            args.output,
            {
                "ok": False,
                "error": f"Planner failed while building blueprint: {e}",
                "validation": v,
            },
            pretty=args.pretty,
        )
        return 1

    _write_json(
        args.output,
        {
            "ok": True,
            "analysis_version": report.analysis_version,
            "generated": report.generated,
            "scenes": v["scenes"],
            "validation": {"errors": v["errors"], "warnings": v["warnings"]},
            "blueprint": blueprint,
        },
        pretty=args.pretty,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
