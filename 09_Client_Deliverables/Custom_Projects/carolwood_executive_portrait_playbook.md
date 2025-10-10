# Carolwood "In-Command, In-Conversation" Portrait Specification

The portrait system for Carolwood leaders operates like a perceptual programming language. Each controllable element—expression, posture, light, composition, and timing—acts as an op-code that compiles into a reliable state of executive presence: decisive yet warm, exacting yet human. Use this specification as a repeatable build sheet for directors, photographers, and retouchers.

---

## 1. Presence Language: Primitives and Outputs

### State Vector (viewer response targets)
- **Competence (C)** — decisiveness, clarity, steadiness
- **Warmth (W)** — empathy, accessibility, rapport
- **Credibility (K)** — trustworthiness; emerges from consistent C × W
- **Attention (A)** — how efficiently the eye lands on and stays with the face

### Primitives (controllable op-codes)

| Primitive | Symbol | Default | Operational effect on state |
| --------- | ------ | ------: | --------------------------- |
| Eye-line height | `EYE(h)` | **h = 0.27H** | Lowering h (+C); raising h (–C); A peaks 0.25–0.30H |
| Key-light angle | `KEY(θ)` | **θ = 30–35°** | 25–40° sculpts jaw/cheek; <25° flattens (–C); >40° hardens (–W) |
| Fill ratio | `FILL(s)` | **s = –1.0 to –1.5 stops** | More fill (+W, –C contrast); less fill (+C, –W) |
| Mouth corner delta | `MOUTH(Δ°)` | **Δ° = +2–3°** | +Δ° (+W); >+5° reads performed; 0° risks chill |
| Lower-lid tension | `SQUINT(τ)` | **τ = micro** | +τ (+C focus); excessive τ reads stern |
| Chin translation | `CHIN(d)` | **d = out, then 5–10 mm down** | Defines jaw (A↑, C↑) without defiance |
| Lean | `LEAN(ℓ)` | **ℓ = forward 2–4 cm** | Invites access (+W) while maintaining C |
| Side gutters | `GUTTERS(g)` | **g = 12–16%** | Tighter framing → presence; too tight → crowding |
| Brand band | `BRAND(t)` | **t = top 10–18%** | Gives CAROLWOOD / ESTATES room; preserves hierarchy |
| Rim intensity | `RIM(r)` | **r = subtle** | Clean separation; excessive r feels fashion |
| Contrast curve | `CURVE(κ)` | **κ = gentle S, +2% mid** | Maintains texture and legibility |
| Hands placement | `HANDS(b)` | **b = 68–80% frame height** | Keeps narrative without distracting from face |

> **Harmony Note**: 27% eye placement is not exactly φ, but it functions as a classical balance point—sitting just below the rule-of-thirds for calm authority.

### Compiler (conceptual shorthand)

```
compile_presence(intent = {C: 0.55, W: 0.45}) =>
    EYE(0.27H) + KEY(33°) + FILL(-1.25) +
    MOUTH(+2.5°) + SQUINT(micro) + CHIN(7 mm) + LEAN(3 cm) +
    GUTTERS(14%) + BRAND(12%) + CURVE(gentle_S) + HANDS(0.74H)
→ portrait that maximizes A and K for the target C:W mix
```

---

## 2. Frame B as the "Eigenface of Leadership"

Treat the neutral-plus expression as a latent anchor in expression space:
- **Anchor recipe**: direct gaze, micro-squint, mouth corners +2–3°, relaxed orbicularis tension, shoulders open, hands in active rest.
- **Why it generalizes**: viewers project marginal warmth or authority adjustments based on context. The image remains stable enough for competence while soft enough for rapport, functioning as a universal solvent across collateral.
- **Policy**: default all deliverables to the Frame B anchor, then nudge ±5–10% toward either pole (authority ↔ warmth) for specific campaigns (e.g., investor deck vs. client bio).

---

## 3. Fourth Dimension Protocol (Temporal Layer)

Static frames imply static leadership. Capture the **decision moment** instead of only the expression peak.

### Capture cadence
1. Prime with the cue: “Picture the client asking, *What would you do?* Give me the *I’ve got you* look.”
2. Run a burst at **10–12 fps** for **0.8–1.2 s** around the “silent yes” micro-smile.
3. Log audio ticks or verbal counts at **2 Hz** to keep frame timing reproducible.

### Selection heuristic
- Let **t₀** mark the frame where the smile peaks (mouth corners max).
- Select **t*** = **t₀ + 0.25–0.35 s** (“echo of decision”): corners settle, lower-lid tension holds, gaze remains intent.
- Reject frames with excessive sclera exposure, gum flash, or pre-decision flatness.

### Optional micro-median synthesis
- Align frames **t*–1**, **t***, **t*+1** via optical flow.
- Blend at **70/100/70** weights to preserve texture while taming blink noise.
- Never blend beyond **0.4 s** total span; longer windows kill the sense of aliveness.

---

## 4. Production SOP (Field Workflow)

### Prelight (≈10 minutes)
- Background: neutral gray gradient (#4a4c48 → #8c8e8a) held ~⅓ stop below average skin tone with a light top grad to protect the wordmark.
- Key: 5′ octa (or 4×6 softbox) at 30–35°, 10–15° above eye-line; meter for **f/4** at low ISO.
- Fill: V-flat or umbrella opposite, held **–1.25 stops** under key.
- Rim/hair: narrow strip from camera-opposite, feathered to separate jacket from wall by 3–5 L* values.
- Negative fill: small black flag on shadow side to keep jaw definition.

### Pose & micro-direction (≈2 minutes)
- Seat height such that eyes land near 27% when framed 4×5; camera at collarbone height with <1° down-tilt.
- Direction: “Chin out, 7 mm down; shoulders open; micro-lean forward 3 cm.”
- Hands: relaxed overlap or soft steeple on desk; rotate watch correctly; remove extraneous props.

### Expression prompts (30–45 seconds)
- “Say a silent *yes* as the plan clicks.”
- “Think through the first three steps.” (engages lower-lid tension)
- “Stay with me; breathe out, eyes on me.” (relaxes jaw/temples)

### Burst & review (≈1 minute)
- Fire the burst; flag candidate frames using the temporal heuristic.
- Confirm geometry: eyes 27% ±2%; gutters 12–16%; brand band 10–18%; hands 68–80%.

### Grade & finish (5–7 minutes)
- Apply gentle S-curve with slightly lifted blacks; maintain texture.
- Micro dodge & burn eyes (+2–4 points); soften under-eye lines without erasing.
- Clean dust/lint, reduce cuff glare, even out desk speculars.
- Optional: 1/8 Black Pro-Mist for highlight bloom control (captured in-camera).
- Export hero at **2400×3000** and web variant at **1065×1330**, both in sRGB.

---

## 5. Trust-Compiler Controls (Scenario Knobs)

| Use-case | Target C:W | Parameter nudges |
| -------- | ---------- | ---------------- |
| Investor / board deck | 60:40 | `FILL(-1.4)`, add +1 squint notch, `MOUTH(+1–2°)`, contrast +5% |
| Seller listing pitch | 50:50 | `FILL(-1.2)`, `MOUTH(+2–3°)`, `LEAN(+1 cm)`, midtones +3% |
| Buyer concierge / lifestyle PR | 45:55 | `EYE(0.28H)`, `MOUTH(+3–4°)`, `FILL(-1.0)`, background +⅙ stop |

---

## 6. Guardrails, Failure Modes, and Ethics

- **Over-optimization**: big teeth display + heavy squint reads salesy. Keep both in the micro range.
- **Symmetry trap**: perfectly even smiles look synthetic; bias toward **1–2%** asymmetry.
- **Context drift**: wardrobe or setting that contradicts the subject’s real-world presence drops credibility even if metrics score high. Align story and environment.
- **Ethical scaffolding**: clarify intent in briefs—“We’re portraying you at your best, not inventing someone else.” This is intentional authenticity, not camouflage.

---

## 7. Verification Rubric (Rapid QA)

1. **Geometry** — eyes 27% ±2%; gutters 12–16%; brand band 10–18%; hands 68–80% of frame height.
2. **Micro-expression** — mouth corners +2–3°; lower-lid tension present; chin out + down; lean 2–4 cm.
3. **Light balance** — key 30–35°; fill –1.0 to –1.5; rim subtle; background ~⅓ stop under skin.
4. **Texture integrity** — pores visible; no plastic smoothing; gentle S-curve only.
5. **Temporal feel** — chosen frame sits at t* (echo of decision); peak grin and pre-decision flatness rejected.

Meet the rubric and the resulting portrait encodes the Carolwood paradox in a single glance: not just a likeness, but the precise moment a leader arrives at certainty.
