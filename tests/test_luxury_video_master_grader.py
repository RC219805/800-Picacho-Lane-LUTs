import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "luxury_video_master_grader.py"
    spec = importlib.util.spec_from_file_location("luxury_video_master_grader", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader  # for mypy
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()
assess_frame_rate = MODULE.assess_frame_rate
build_command = MODULE.build_command
build_filter_graph = MODULE.build_filter_graph
determine_color_metadata = MODULE.determine_color_metadata


def test_assess_frame_rate_respects_user_override():
    probe = {"streams": [{"codec_type": "video"}]}

    plan = assess_frame_rate(probe, "25", tolerance=0.05)

    assert plan.target == "25/1"
    assert "override" in plan.note


def test_assess_frame_rate_detects_vfr():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "24000/1001",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target == "30000/1001"
    assert "variable frame-rate" in plan.note


def test_assess_frame_rate_conforms_off_standard():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "48/1",
                "r_frame_rate": "48/1",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target == "50/1"
    assert "off-standard" in plan.note


def test_assess_frame_rate_preserves_within_tolerance():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }
        ]
    }

    plan = assess_frame_rate(probe, None, tolerance=0.05)

    assert plan.target is None
    assert "preserving timing" in plan.note


def test_build_filter_graph_includes_optional_nodes(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "contrast": 1.05,
        "saturation": 1.02,
        "gamma": 0.98,
        "brightness": 0.01,
        "warmth": 0.9,
        "cool": -0.7,
        "lut_strength": 0.6,
        "denoise": "medium",
        "sharpen": "soft",
        "grain": 3.5,
        "target_fps": "24000/1001",
    }

    graph, output_label = build_filter_graph(config)

    assert output_label == "vout"
    assert f"lut3d=file={lut_path}:interp=tetrahedral" in graph
    assert "hqdn3d=luma_spatial=2.8" in graph
    assert "eq=contrast=1.0500:saturation=1.0200:gamma=0.9800:brightness=0.0100" in graph
    assert "colorbalance=rm=0.5000:gm=0.0000:bm=-0.5000" in graph
    assert "blend=all_expr='A*(1-0.6000)+B*0.6000'" in graph
    assert "unsharp=luma_msize_x=7" in graph
    assert "noise=alls=3.50:allf=t+u" in graph
    assert "fps=fps=24000/1001" in graph


def test_build_filter_graph_rejects_unknown_denoise(tmp_path):
    lut_path = tmp_path / "dummy.cube"
    lut_path.write_text("# dummy LUT\n")

    config = {
        "lut": lut_path,
        "denoise": "extreme",
    }

    with pytest.raises(ValueError):
        build_filter_graph(config)


def test_determine_color_metadata_prioritises_explicit():
    args = argparse.Namespace(
        color_primaries="bt709",
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_from_source=False,
    )
    probe = {"streams": []}

    assert determine_color_metadata(args, probe) == ("bt709", "smpte2084", "bt2020nc")


def test_determine_color_metadata_from_source_filters_unknown():
    args = argparse.Namespace(
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        color_from_source=True,
    )
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "color_primaries": "bt2020",
                "color_trc": "unknown",
                "colorspace": "bt2020nc",
            }
        ]
    }

    assert determine_color_metadata(args, probe) == ("bt2020", None, "bt2020nc")


def test_determine_color_metadata_defaults_to_none_when_missing():
    args = argparse.Namespace(
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        color_from_source=True,
    )
    probe = {"streams": []}

    assert determine_color_metadata(args, probe) == (None, None, None)


def test_build_command_includes_expected_arguments(tmp_path):
    input_path = tmp_path / "input.mov"
    output_path = tmp_path / "output.mov"

    cmd = build_command(
        input_path,
        output_path,
        "graph",
        "vout",
        overwrite=True,
        video_codec="prores_ks",
        prores_profile=3,
        bitrate="500M",
        audio_codec="pcm_s24le",
        audio_bitrate="320k",
        threads=8,
        log_level="warning",
        preview_frames=10,
        vsync="cfr",
        color_primaries="bt2020",
        color_transfer="smpte2084",
        color_space="bt2020nc",
    )

    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert cmd.count("-color_primaries") == 1
    assert cmd[-1] == str(output_path)
    assert "-filter_complex" in cmd and "graph" in cmd
    assert "-profile:v" in cmd
    assert "-pix_fmt" in cmd
    assert "-b:v" in cmd
    assert "-b:a" in cmd
    assert "-threads" in cmd
    assert "-frames:v" in cmd
    assert "-vsync" in cmd and "cfr" in cmd
    assert "-color_trc" in cmd and "smpte2084" in cmd
    assert "-colorspace" in cmd and "bt2020nc" in cmd
