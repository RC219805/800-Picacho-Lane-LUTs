"""Material Response principle support utilities.

The marketing material for the 800 Picacho Lane collection repeatedly alludes
to a proprietary "Material Response" technology.  The production codebase does
not actually perform physically based rendering, but we can still document what
the principle *means* so tests and documentation have a concrete artefact to
reference.  This module provides such an artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class MaterialResponseExample:
    """Structured example describing how the principle manifests."""

    material: str
    lighting: str
    challenge: str
    response: str
    outcome: str

    def as_dict(self) -> Dict[str, str]:
        """Return the example in a serialisable dictionary form."""

        return {
            "material": self.material,
            "lighting": self.lighting,
            "challenge": self.challenge,
            "response": self.response,
            "outcome": self.outcome,
        }


class MaterialResponsePrinciple:
    """Captures the intent behind the Material Response marketing claims.

    The principle is intentionally high-level: it frames how surface-specific
    adjustments should be reasoned about even when a pipeline primarily applies
    tone and colour transforms.  Tests that refer to this principle can inspect
    the structured data returned here without coupling to any particular image
    processing implementation.
    """

    name: str = "Material Response"
    focus: str = (
        "Honor the unique light interaction of each surface instead of applying "
        "purely global transforms."
    )

    tenets: List[str] = [
        "Respect energy conservation in highlights so reflective materials retain believable sheen.",
        "Preserve midtone texture to keep organic materials tactile and dimensional.",
        "Blend transitions between adjacent materials so adjustments feel authored rather than procedural.",
    ]

    def describe(self) -> str:
        """Return a human readable description of the principle."""

        return f"{self.name}: {self.focus}"

    def guidelines(self) -> List[str]:
        """Return the set of guiding tenets."""

        return list(self.tenets)

    def generate_examples(self) -> List[Dict[str, str]]:
        """Produce canonical examples demonstrating the principle in action.

        The examples intentionally cover a variety of materials.  Returning
        dictionaries keeps the data ergonomic for documentation tooling and
        avoids a dependency on this module's dataclass by callers.
        """

        examples: Iterable[MaterialResponseExample] = [
            MaterialResponseExample(
                material="polished marble foyer",
                lighting="late-afternoon sun grazing through clerestory windows",
                challenge="Specular highlights risk clipping and flattening the stone veining.",
                response="Apply highlight recovery before clarity so micro-contrast is boosted only after energy is preserved.",
                outcome="Marble retains luminous sheen with detailed veining rather than a white patch.",
            ),
            MaterialResponseExample(
                material="brushed brass fixtures",
                lighting="mixed tungsten sconces and cool daylight",
                challenge="Mixed colour temperatures create muddy warm-cool interference on the anisotropic metal.",
                response="Balance tint locally while moderating saturation to keep the brushed texture defined.",
                outcome="Brass reads as intentionally warm while grooves stay articulated.",
            ),
            MaterialResponseExample(
                material="velvet chaise",
                lighting="soft bounced fill with narrow specular kicker",
                challenge="Global contrast lift would crush the velvet's directionality and make it plastic.",
                response="Target midtone contrast and vibrance to maintain pile shading while enriching dye density.",
                outcome="Velvet keeps tactile depth and directional sheen without waxy highlights.",
            ),
        ]

        return [example.as_dict() for example in examples]


_STOP_WORDS = {
    "the",
    "and",
    "or",
    "so",
    "that",
    "a",
    "an",
    "to",
    "of",
    "in",
    "between",
    "with",
    "for",
    "be",
    "is",
    "are",
    "as",
    "keep",
    "so",
    "rather",
}


def _extract_keywords(text: str) -> List[str]:
    """Return a list of meaningful lowercase keywords from ``text``."""

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in _STOP_WORDS and len(token) > 2]


def violates(decision: str, tenet: str) -> bool:
    """Heuristically determine whether ``decision`` conflicts with ``tenet``."""

    decision_lower = decision.lower()
    tenet_lower = tenet.lower()

    # Specific heuristics for the three marketing tenets.  These catch the most
    # common contradictions that show up in the architectural documentation and
    # associated tests.
    tenet_specific_checks = (
        (
            {"energy", "highlight"},
            {
                "clip the highlights",
                "blow out the highlights",
                "overexpose highlights",
                "ignore energy conservation",
                "treat highlights as unlimited",
            },
        ),
        (
            {"midtone", "texture"},
            {
                "flatten the midtones",
                "blur midtone texture",
                "remove midtone detail",
                "plastic midtones",
                "eliminate texture",
            },
        ),
        (
            {"blend", "transitions"},
            {
                "hard mask transition",
                "no blending between materials",
                "treat each material in isolation",
                "apply a binary mask",
            },
        ),
    )

    for keywords, forbidden_phrases in tenet_specific_checks:
        if keywords.issubset(set(_extract_keywords(tenet_lower))):
            if any(phrase in decision_lower for phrase in forbidden_phrases):
                return True

    # Generic negation handling: if the decision explicitly ignores or bypasses
    # aspects that the tenet cares about we treat it as a violation.
    negating_words = {
        "ignore",
        "bypass",
        "discard",
        "skip",
        "flatten",
        "eliminate",
        "remove",
        "crush",
        "erase",
    }
    keywords = _extract_keywords(tenet_lower)

    for negation in negating_words:
        if negation in decision_lower:
            if any(f"{negation} {keyword}" in decision_lower for keyword in keywords):
                return True
            # Fall back to a proximity check if the explicit "negation keyword"
            # pattern was not matched.  This keeps the heuristic resilient when
            # the decision references the concept in a different grammatical
            # order (e.g. "midtone texture gets flattened")
            for keyword in keywords:
                if keyword in decision_lower:
                    return True

    return False


class MarketingClaimValidator:
    """Validates that implementation decisions align with marketing principles."""

    def assess_decision(
        self, decision: str, principle: MaterialResponsePrinciple
    ) -> bool:
        """Return ``True`` when ``decision`` honours ``principle``'s tenets."""

        for tenet in principle.guidelines():
            if violates(decision, tenet):
                return False
        return True


__all__ = [
    "MaterialResponsePrinciple",
    "MaterialResponseExample",
    "MarketingClaimValidator",
    "violates",
]

