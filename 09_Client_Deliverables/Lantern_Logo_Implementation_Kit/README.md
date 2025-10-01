# Lantern Logo Implementation Kit

This package mirrors the Lantern component specification so designers and engineers can import assets without additional formatting work. It provides platform-agnostic tokens, a CSS starter, and the reference SVG, ensuring that identity, animation hooks, and accessibility affordances remain consistent across surfaces.

## Contents

- `lantern_tokens.json` — Primitive color and gradient tokens structured for Style Dictionary compilation into CSS, iOS, Android, and Figma targets.
- `lantern_logo.css` — Web starter styles illustrating how to compose semantic CSS variables, responsive padding, and hover affordances from the primitive token set.
- `lantern_logo.svg` — Accessibility-ready master mark that consumes the token variables, exposes the gradient definition, and preserves the vessel and flame geometry described in the component spec.

For governance details, geometry rules, and motion guidance, reference `../../08_Documentation/lantern_logo_component_spec.md`.

## Gradient architecture

- The reference `lantern_logo.svg` composes its flame gradient using the primitive color tokens `color.brand.azure` and `color.brand.cyan`, ensuring the SVG automatically reflects any upstream palette changes.
- When defining `gradient.brand.primary` in `lantern_tokens.json`, compose the gradient stops from the existing primitive tokens (or add new primitives first) so Style Dictionary outputs inherit the canonical palette described in the [component specification](../../08_Documentation/lantern_logo_component_spec.md#3-token-system).

> **Caution:** Always update or extend the primitive color tokens before adjusting gradient definitions, and avoid hard-coding new hexadecimal values directly into gradient tokens or SVG assets.
