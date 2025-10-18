# file: picacho_lane_luts/synthetic_viewer.py
"""Synthetic viewers for subjective video quality evaluation.

CLI:
    python -m picacho_lane_luts.synthetic_viewer score -i frames.json --archetype default --summary
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Iterator, List, Mapping, MutableSequence, Optional, Sequence

try:
    import typer  # why: lightweight, already in project deps
    _HAS_TYPER = True
except Exception:
    _HAS_TYPER = False


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp value to [minimum, maximum]."""
    return max(minimum, min(value, maximum))


@dataclass(frozen=True)
class ACUScore:
    """Aggregated aesthetic score for a video experience."""
    technical: float
    emotional: float
    memorability: float
    desire_quotient: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "technical", _clamp(self.technical))
        object.__setattr__(self, "emotional", _clamp(self.emotional))
        object.__setattr__(self, "memorability", _clamp(self.memorability))
        object.__setattr__(self, "desire_quotient", _clamp(self.desire_quotient))

    @property
    def overall(self) -> float:
        return fmean((self.technical, self.emotional, self.memorability, self.desire_quotient))

    def as_dict(self) -> Mapping[str, float]:
        return {
            "technical": self.technical,
            "emotional": self.emotional,
            "memorability": self.memorability,
            "desire_quotient": self.desire_quotient,
            "overall": self.overall,
        }


@dataclass(frozen=True)
class JourneyMoment:
    """Frame/beat of the emotional journey in [0,1]."""
    technical: float
    emotional: float
    memorability: float
    desire: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "technical", _clamp(self.technical))
        object.__setattr__(self, "emotional", _clamp(self.emotional))
        object.__setattr__(self, "memorability", _clamp(self.memorability))
        object.__setattr__(self, "desire", _clamp(self.desire))


class EmotionalJourney:
    """Sequence wrapper exposing summary statistics for moments."""

    def __init__(self, moments: Sequence[JourneyMoment]):
        if not moments:
            raise ValueError("An emotional journey requires at least one moment")
        self._moments: Sequence[JourneyMoment] = tuple(moments)

    def __iter__(self) -> Iterator[JourneyMoment]:
        return iter(self._moments)

    @property
    def moments(self) -> Sequence[JourneyMoment]:
        return self._moments

    def _mean(self, attr: str) -> float:
        return fmean(getattr(m, attr) for m in self._moments)

    def _variability(self, attr: str) -> float:
        values = [getattr(m, attr) for m in self._moments]
        return max(values) - min(values) if len(values) > 1 else 0.0

    def _rhythm(self, attr: str) -> float:
        if len(self._moments) < 2:
            return 0.0
        deltas = [abs(getattr(self._moments[i + 1], attr) - getattr(self._moments[i], attr)) for i in range(len(self._moments) - 1)]
        return fmean(deltas)

    def summary(self) -> Mapping[str, float]:
        return {
            "technical": self._mean("technical"),
            "emotional": self._mean("emotional"),
            "memorability": self._mean("memorability"),
            "desire": self._mean("desire"),
            "emotional_variability": self._variability("emotional"),
            "memorability_variability": self._variability("memorability"),
            "desire_rhythm": self._rhythm("desire"),
            "length": float(len(self._moments)),
        }


class DigitalConsciousness:
    """Normalises raw video impressions into an EmotionalJourney."""

    def __init__(self, archetype: str):
        self.archetype = archetype

    def traverse(
        self,
        video_stream: Iterable[Mapping[str, float] | JourneyMoment] | EmotionalJourney,
    ) -> EmotionalJourney:
        if isinstance(video_stream, EmotionalJourney):
            return video_stream

        moments: List[JourneyMoment] = []
        for entry in video_stream:
            if isinstance(entry, JourneyMoment):
                moments.append(entry)
                continue
            if not isinstance(entry, Mapping):  # safety net
                raise TypeError("video_stream entries must be mappings or JourneyMoment instances")

            moment = JourneyMoment(
                technical=float(entry.get("technical", 0.0)),
                emotional=float(entry.get("emotional", entry.get("emotion", 0.0))),
                memorability=float(entry.get("memorability", entry.get("memory", 0.0))),
                desire=float(entry.get("desire", entry.get("desire_quotient", 0.0))),
            )
            moments.append(moment)

        return EmotionalJourney(moments)


@dataclass(frozen=True)
class ArchetypeProfile:
    """Weighting profile used by the aesthetic cortex."""
    technical_emphasis: float = 1.0
    emotional_emphasis: float = 1.0
    memorability_emphasis: float = 1.0
    desire_emphasis: float = 1.0
    variability_bonus: float = 0.05
    rhythm_bonus: float = 0.05
    baseline_bias: float = 0.0


class ExperientialMemory:
    """Stores light-weight snapshots of journeys the viewer has taken."""

    def __init__(self):
        self._snapshots: MutableSequence[Mapping[str, float]] = []

    @property
    def snapshots(self) -> Sequence[Mapping[str, float]]:
        return tuple(self._snapshots)

    def remember(self, journey: EmotionalJourney) -> Mapping[str, float]:
        snapshot = journey.summary()
        self._snapshots.append(snapshot)
        return snapshot


class TrainedOnMillionsOfLuxuryViewings:
    """Maps an EmotionalJourney to an ACUScore."""

    def __init__(self):
        self._profiles = {
            "default": ArchetypeProfile(),
            "minimalist_millennial": ArchetypeProfile(
                technical_emphasis=0.96,
                emotional_emphasis=1.08,
                memorability_emphasis=1.02,
                desire_emphasis=1.05,
                variability_bonus=0.04,
                rhythm_bonus=0.06,
                baseline_bias=0.01,
            ),
            "traditional_luxury_connoisseur": ArchetypeProfile(
                technical_emphasis=1.05,
                emotional_emphasis=0.98,
                memorability_emphasis=1.04,
                desire_emphasis=1.02,
                variability_bonus=0.03,
                rhythm_bonus=0.04,
            ),
            "futurist_tech_executive": ArchetypeProfile(
                technical_emphasis=1.07,
                emotional_emphasis=1.02,
                memorability_emphasis=0.97,
                desire_emphasis=1.08,
                variability_bonus=0.05,
                rhythm_bonus=0.07,
                baseline_bias=0.015,
            ),
        }

    def score(self, journey: EmotionalJourney, archetype: str) -> ACUScore:
        summary = journey.summary()
        profile = self._profiles.get(archetype, self._profiles["default"])

        # why: "technical" perceived polish benefits from emotional dynamism
        technical = self._score_channel(summary["technical"], profile.technical_emphasis, summary["emotional_variability"], profile)
        emotional = self._score_channel(summary["emotional"], profile.emotional_emphasis, summary["emotional_variability"], profile)
        memorability = self._score_channel(summary["memorability"], profile.memorability_emphasis, summary["memorability_variability"], profile)
        desire = self._score_channel(summary["desire"], profile.desire_emphasis, summary["desire_rhythm"], profile)

        return ACUScore(technical=technical, emotional=emotional, memorability=memorability, desire_quotient=desire)

    @staticmethod
    def _score_channel(base_value: float, emphasis: float, dynamism: float, profile: ArchetypeProfile) -> float:
        raw = base_value * emphasis
        raw += dynamism * profile.variability_bonus
        raw += profile.baseline_bias
        return _clamp(raw)


class SyntheticViewer:
    """Synthesised viewer that reports an ACUScore for a video."""

    def __init__(self, archetype: str = "default") -> None:
        self.archetype = archetype
        self.consciousness = DigitalConsciousness(archetype)
        self.memory = ExperientialMemory()
        self.aesthetic_cortex = TrainedOnMillionsOfLuxuryViewings()

    def clone(self, archetype: str) -> "SyntheticViewer":
        return SyntheticViewer(archetype)

    def _score_from_journey(self, journey: EmotionalJourney) -> ACUScore:
        self.memory.remember(journey)
        return self.aesthetic_cortex.score(journey, self.archetype)

    def reach_aesthetic_consensus(self, scores: Sequence[ACUScore]) -> ACUScore:
        if not scores:
            raise ValueError("At least one score is required for consensus")
        return ACUScore(
            technical=fmean(s.technical for s in scores),
            emotional=fmean(s.emotional for s in scores),
            memorability=fmean(s.memorability for s in scores),
            desire_quotient=fmean(s.desire_quotient for s in scores),
        )

    def experience_content(
        self,
        video_stream: Iterable[Mapping[str, float] | JourneyMoment] | EmotionalJourney,
    ) -> ACUScore:
        """Experience a video stream and return the consensus ACUScore."""
        journey = self.consciousness.traverse(video_stream)
        primary_score = self._score_from_journey(journey)

        perspectives = [
            self.clone(archetype="minimalist_millennial"),
            self.clone(archetype="traditional_luxury_connoisseur"),
            self.clone(archetype="futurist_tech_executive"),
        ]
        perspective_scores = [v._score_from_journey(journey) for v in perspectives]
        return self.reach_aesthetic_consensus([primary_score, *perspective_scores])


# -------------------------- Convenience & CLI --------------------------

def score_video_stream(
    frames: Iterable[Mapping[str, float] | JourneyMoment] | EmotionalJourney,
    *,
    archetype: str = "default",
) -> ACUScore:
    viewer = SyntheticViewer(archetype)
    return viewer.experience_content(frames)


def score_to_json(score: ACUScore) -> str:
    return json.dumps(score.as_dict(), indent=2)


def _read_json_input(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_frames_from_json(obj: Any) -> List[Mapping[str, float]]:
    # Accept {"frames":[...]} or a list directly
    frames = obj.get("frames") if isinstance(obj, Mapping) and "frames" in obj else obj
    if not isinstance(frames, list):
        raise TypeError("JSON must be a list of frames or an object with a 'frames' array")
    out: List[Mapping[str, float]] = []
    for item in frames:
        if isinstance(item, Mapping):
            out.append(item)  # DigitalConsciousness handles coercion + defaults
        else:
            raise TypeError("Frame entries must be objects/mappings")
    return out


_CSV_ALIASES = {
    "technical": {"technical", "tech"},
    "emotional": {"emotional", "emotion"},
    "memorability": {"memorability", "memory"},
    "desire": {"desire", "desire_quotient", "desireq"},
}


def _coerce_frames_from_csv(path: str) -> List[Mapping[str, float]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        # map aliases → canonical
        def canon(key: str) -> Optional[str]:
            kl = key.strip().lower()
            for tgt, names in _CSV_ALIASES.items():
                if kl in names:
                    return tgt
            return None

        mapped_rows: List[Mapping[str, float]] = []
        for row in reader:
            rec: dict[str, float] = {}
            for k, v in row.items():
                c = canon(k)
                if c is None:
                    continue
                try:
                    rec[c] = float(v)
                except (TypeError, ValueError):
                    rec[c] = 0.0
            mapped_rows.append(rec)
        if not mapped_rows:
            raise ValueError("No usable rows found in CSV")
        return mapped_rows


def _main(argv: Optional[Sequence[str]] = None) -> int:
    if not _HAS_TYPER:
        print("This CLI requires 'typer'. Install with: pip install typer", file=sys.stderr)
        return 2

    app = typer.Typer(add_completion=False, no_args_is_help=True, help="Synthetic Viewer – ACU scoring")

    @app.command("score")
    def score_cmd(
        input: str = typer.Option(..., "--input", "-i", help="Path to JSON/CSV file or '-' for stdin (JSON)."),
        fmt: str = typer.Option("auto", "--format", "-f", help="Input format", case_sensitive=False),
        archetype: str = typer.Option("default", "--archetype", "-a"),
        summary: bool = typer.Option(False, "--summary", help="Print only the overall score."),
    ) -> None:
        fmt_l = fmt.lower()
        frames: List[Mapping[str, float]]
        if fmt_l == "auto":
            if input == "-" or input.lower().endswith(".json"):
                fmt_l = "json"
            elif input.lower().endswith(".csv"):
                fmt_l = "csv"
            else:
                raise typer.BadParameter("Cannot infer format. Use --format json|csv.")
        if fmt_l == "json":
            obj = _read_json_input(input)
            frames = _coerce_frames_from_json(obj)
        elif fmt_l == "csv":
            if input == "-":
                raise typer.BadParameter("CSV from stdin not supported; provide a file path.")
            frames = _coerce_frames_from_csv(input)
        else:
            raise typer.BadParameter("Unsupported format. Use json or csv.")

        score = score_video_stream(frames, archetype=archetype)
        if summary:
            print(f"{score.overall:.6f}")
        else:
            print(score_to_json(score))

    return app(standalone_mode=True)

# python -m support
def main() -> None:
    raise SystemExit(_main())


__all__ = [
    "ACUScore",
    "JourneyMoment",
    "EmotionalJourney",
    "DigitalConsciousness",
    "ArchetypeProfile",
    "ExperientialMemory",
    "TrainedOnMillionsOfLuxuryViewings",
    "SyntheticViewer",
    "score_video_stream",
    "score_to_json",
    "main",
]

if __name__ == "__main__":
    main()