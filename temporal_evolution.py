# file: picacho_lane_luts/temporal_roadmap.py
"""Utilities for normalising temporal evolution roadmaps.

Adds a minimal CLI:
    python -m picacho_lane_luts.temporal_roadmap render -i roadmap.yaml --output-format markdown
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableSequence, Sequence, Any, Optional, Literal

try:
    import typer  # lightweight CLI
    _HAS_TYPER = True
except Exception:
    _HAS_TYPER = False


# ----------------------------- core datamodel ------------------------------

@dataclass(frozen=True)
class EvolutionDirective:
    """A single directive inside a temporal evolution discipline.

    summary: Short label describing the directive. Mapping keys become summaries.
    detail: Optional explanatory text from mapping values. Strings imply no detail.
    """

    summary: str
    detail: str | None = None

    def to_bullet(self) -> str:
        """Return a Markdown bullet point."""
        if self.detail:
            return f"- **{self.summary}**: {self.detail}"
        return f"- {self.summary}"

    def serialise(self) -> str | dict[str, str]:
        """Return a JSON/YAML friendly representation."""
        if self.detail is None:
            return self.summary
        return {self.summary: self.detail}


@dataclass(frozen=True)
class EvolutionDiscipline:
    """A named collection of EvolutionDirective instances."""

    name: str
    directives: Sequence[EvolutionDirective]

    @property
    def human_name(self) -> str:
        """Prettified version of name for display."""
        return self.name.replace("_", " ").title()

    def to_markdown(self) -> str:
        """Render the discipline and its directives as Markdown."""
        lines = [f"### {self.human_name}"]
        lines.extend(directive.to_bullet() for directive in self.directives)
        return "\n".join(lines)

    def serialise(self) -> list[str | dict[str, str]]:
        """Return a serialisable list of directives."""
        return [directive.serialise() for directive in self.directives]


@dataclass(frozen=True)
class TemporalEvolutionRoadmap:
    """Structured representation of the ``temporal_evolution`` document."""

    disciplines: Sequence[EvolutionDiscipline]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TemporalEvolutionRoadmap":
        """Create a roadmap from a mapping (either the block itself or a container)."""
        root: object
        if "temporal_evolution" in payload:
            root = payload["temporal_evolution"]
        else:
            root = payload

        if not isinstance(root, Mapping):
            raise TypeError("temporal_evolution payload must be a mapping")

        disciplines: MutableSequence[EvolutionDiscipline] = []
        for raw_name, directives in root.items():
            if not isinstance(raw_name, str):
                raise TypeError("Discipline names must be strings")
            name = raw_name.strip()
            if not name:
                raise TypeError("Discipline names must not be empty")
            normalized = _normalise_directives(name, directives)
            disciplines.append(EvolutionDiscipline(name=name, directives=tuple(normalized)))

        return cls(disciplines=tuple(disciplines))

    @classmethod
    def from_file(cls, path: str | Path, *, fmt: Literal["auto", "yaml", "json"] = "auto") -> "TemporalEvolutionRoadmap":
        """Load YAML/JSON from *path* and produce a roadmap."""
        mapping = _load_mapping(path, fmt=fmt)
        return cls.from_mapping(mapping)

    def to_markdown(self) -> str:
        """Render the full roadmap as Markdown."""
        return "\n\n".join(discipline.to_markdown() for discipline in self.disciplines)

    def serialise(self) -> dict[str, list[str | dict[str, str]]]:
        """Return a serialisable representation of the roadmap."""
        return {discipline.name: discipline.serialise() for discipline in self.disciplines}


def _normalise_directives(name: str, directives: object) -> List[EvolutionDirective]:
    if isinstance(directives, (str, bytes)) or not isinstance(directives, Sequence):
        raise TypeError(f"Directives for discipline '{name}' must be a sequence of steps")

    result: List[EvolutionDirective] = []
    for entry in directives:
        if isinstance(entry, Mapping):
            if len(entry) != 1:
                raise TypeError(
                    f"Directive mappings for discipline '{name}' must contain a single entry"
                )
            summary, detail = next(iter(entry.items()))
            if not isinstance(summary, str) or not isinstance(detail, str):
                raise TypeError(
                    f"Directive mapping for discipline '{name}' must map strings to strings"
                )
            stripped_summary = summary.strip()
            if not stripped_summary:
                raise TypeError(
                    f"Directive mapping for discipline '{name}' must include a summary"
                )
            result.append(EvolutionDirective(stripped_summary, detail.strip()))
        elif isinstance(entry, str):
            stripped = entry.strip()
            if not stripped:
                continue
            result.append(EvolutionDirective(stripped))
        else:
            raise TypeError(
                f"Unsupported directive type for discipline '{name}': {type(entry)!r}"
            )

    return result


# ------------------------------- I/O helpers -------------------------------

def _load_mapping(path: str | Path, *, fmt: Literal["auto", "yaml", "json"] = "auto") -> Mapping[str, Any]:
    """Load a top-level mapping from YAML or JSON."""
    p = Path(path)
    if fmt == "auto":
        if p.suffix.lower() in {".yaml", ".yml"}:
            fmt = "yaml"
        elif p.suffix.lower() == ".json" or str(path) == "-":
            fmt = "json"
        else:
            fmt = "yaml"  # why: project documents are primarily YAML

    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("Reading YAML requires 'pyyaml'. Install with: pip install pyyaml") from e
        if str(path) == "-":
            data = yaml.safe_load(sys.stdin.read())
        else:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    else:
        if str(path) == "-":
            data = json.load(sys.stdin)
        else:
            data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, Mapping):
        raise TypeError("Root document must be a mapping")
    return data


# ----------------------------------- CLI -----------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    if not _HAS_TYPER:
        print("This CLI requires 'typer'. Install with: pip install typer", file=sys.stderr)
        return 2

    app = typer.Typer(add_completion=False, no_args_is_help=True, help="Temporal Evolution Roadmap utilities")

    @app.command("render")
    def render(
        input: str = typer.Option(..., "--input", "-i", help="Path to YAML/JSON file or '-' for stdin (JSON when '-')"),
        fmt: str = typer.Option("auto", "--format", "-f", help="Input format: auto|yaml|json"),
        output_format: str = typer.Option("markdown", "--output-format", "-o", help="Output: markdown|json"),
    ) -> None:
        fmt_l = fmt.lower()
        out_l = output_format.lower()
        if fmt_l not in {"auto", "yaml", "json"}:
            raise typer.BadParameter("--format must be auto, yaml or json")
        if out_l not in {"markdown", "json"}:
            raise typer.BadParameter("--output-format must be markdown or json")

        try:
            roadmap = TemporalEvolutionRoadmap.from_file(input, fmt=fmt_l)  # type: ignore[arg-type]
        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

        if out_l == "markdown":
            print(roadmap.to_markdown())
        else:
            print(json.dumps(roadmap.serialise(), indent=2, ensure_ascii=False))

    return app(standalone_mode=True)

# python -m support
def main() -> None:
    raise SystemExit(_main())


__all__ = [
    "EvolutionDirective",
    "EvolutionDiscipline",
    "TemporalEvolutionRoadmap",
    "main",
]

if __name__ == "__main__":
    main()