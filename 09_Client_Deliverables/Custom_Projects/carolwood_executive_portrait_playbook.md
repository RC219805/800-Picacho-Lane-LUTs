# Carolwood Presence Compiler v1.1

The Carolwood executive portrait is no longer a best-effort shoot; it is a compiled artifact. Every controllable input—composition, light, expression, timing, and finishing—functions as an op-code that produces a predictable state of leadership presence. Ship this blueprint to any studio and expect the same output: decisive yet approachable, precise yet human.

**Canonical Spec**  
Download and operate from the line-numbered codex: [`presence_compiler_spec_v1.1.txt`](presence_compiler_spec_v1.1.txt). Key anchors: lines 037–041 define `compilePresence(..)`, line 066 locks the t* echo-of-decision rule, and lines 134–135 articulate the intentional-authenticity verdict.

---

## 1. Presence Compiler — Machine-Readable Specification

```yaml
presence_compiler:
  intent:
    competence: 0.55    # target C
    warmth:     0.45    # target W
  composition:
    aspect_ratio: 4:5
    eye_line:
      target_pct: 0.27  # eyes at 27% of frame height
      tolerance: 0.02   # ±2% allowable
    side_gutters_pct_each: 0.14   # acceptable window 0.12–0.16
    brand_band_pct_top: 0.12      # acceptable window 0.10–0.18
    chest_band_pct: [0.68, 0.80]
    centerline: nose_on_vertical_center
  expression:
    mouth_corners_delta_deg: 2.5  # +2–3° lift
    lower_lid_tension: micro
    chin_translation_mm: {out: 7, down: 7}
    lean_forward_cm: 3
  lighting:
    key_angle_deg: [30, 35]
    key_height: slightly_above_eyes
    fill_relative_stops: -1.25
    rim: subtle_separation
    background_gray_hex: ["#4a4c48", "#8c8e8a"]
  temporal:
    cadence_fps: 12
    cue: silent_yes
    t_star_offset_s: [0.25, 0.35]  # echo of decision after peak
    selection_rule: corners_settle_and_eyes_hold_intent
    median_stack:
      enabled: true
      window: 3
      weights: [0.7, 1.0, 0.7]
  finishing:
    curve: gentle_S
    midtone_lift_pct: 2
    local_clarity_region: eyes_cheeks_only
    desk_glare_tame_pct: 8
    export:
      web:  {size_px: [1065, 1330], color: sRGB, quality: 90}
      hero: {size_px: [2400, 3000], color: sRGB, quality: 92}
  compliance_thresholds:
    eye_line_abs_error_pct: 0.02
    gutters_pct_range_each: [0.12, 0.16]
    brand_band_pct_range:  [0.10, 0.18]
    chest_band_pct_range:  [0.68, 0.80]
```

### State vector (viewer response targets)
- **Competence (C)** — decisiveness, clarity, steadiness
- **Warmth (W)** — empathy, accessibility, rapport
- **Credibility (K)** — emergent trust (C × W) stabilized by repeatable geometry
- **Attention (A)** — how efficiently the eye lands on and stays with the face

> 27% eye placement is not φ, but it behaves like a harmonic anchor below the rule-of-thirds, reading as calm authority.

### Compiler shorthand

```
compile_presence(intent = {C:0.55, W:0.45}) =>
    EYE(0.27H) + KEY(33°) + FILL(-1.25) +
    MOUTH(+2.5°) + SQUINT(micro) + CHIN(7 mm) + LEAN(3 cm) +
    GUTTERS(14%) + BRAND(12%) + CURVE(gentle_S) + HANDS(0.74H)
→ portrait that maximizes A and K for the target C:W mix
```

---

## 2. Frame B — The Eigenface of Leadership

- **Anchor recipe:** direct gaze, micro-squint, mouth corners +2–3°, relaxed orbicularis tension, shoulders open, hands in active rest.
- **Why it generalizes:** the expression occupies the mathematical center between authority and approachability. Viewers mentally project warmer or more formal reads based on need, making it a universal solvent across collateral.
- **Policy:** default to Frame B, then nudge ±5–10% toward either pole (authority ↔ warmth) for specific campaigns (investor deck vs. concierge outreach).

---

## 3. Fourth Dimension Protocol — Echo of Decision Capture

Leadership is experienced in motion. Encode the decisive moment by sampling around the “silent yes,” not just at it.

### Setup
- Seat height and camera placement pre-set so eyes land at 27% in a 4×5 crop.
- Key 30–35° at 10–15° above eye-line; fill −1.25 stops; subtle rim; background ⅓ stop under average skin tone; small negative fill on shadow side.

### Prompt & burst (≈1 s total)
1. “Picture the client asking, *What would you do?* Give me the *I’ve got you* look.”
2. Run a continuous burst at **10–12 fps** for **0.8–1.2 s**.
3. Mark the peak grin frame (**t₀**) with a voice tick (“one”) or metronome click.

### Selection
- Choose **t*** = **t₀ + 0.25–0.35 s** — the “echo of decision.” Mouth corners settle, lower-lid tension sustains, eyes stay intent.
- Reject sclera-dominant, gum-heavy, or pre-decision flat frames.

### Optional micro-median blend
- Align frames **t*−1**, **t***, **t*+1** via optical flow.
- Blend at **70/100/70** weights. Never extend beyond 0.4 s total span; longer windows kill aliveness.

---

## 4. Companion Disruption File — Proof-of-Life Set

Ship reality alongside precision. Deliver five auxiliary frames with the hero portrait:

1. **t*−0.20 s:** reflective pre-decision.
2. **t***: echo-of-decision (hero-adjacent).
3. **t*+0.25 s:** post-decision softening.
4. **Mid-gesture:** watch adjustment or pen reach (motion blur <1/60).
5. **Relaxed reset:** neutral mouth, steady eyes.

**Packaging requirements**
- One contact sheet PNG (2400 px tall) plus five individual JPGs.
- Metadata sidecar (JSON or YAML) noting C:W intent, capture angles, t* index, and any parameter nudges.
- Usage notes: frames 1 & 3 for press alternates, 4 for social proof, 5 for internal decks.

---

## 5. Trust Compiler Controls — Ready Recipes

| Scenario               | Target C:W | Parameter nudges                                                      |
| ---------------------- | ---------- | -------------------------------------------------------------------- |
| Investor / board deck  | 0.60:0.40  | Fill −1.4, contrast +5%, squint +1 notch, mouth corners +1°          |
| Seller listing pitch   | 0.50:0.50  | Fill −1.2, midtone +3%, lean +1 cm                                    |
| Concierge / lifestyle  | 0.45:0.55  | Eye-line 0.28H, mouth corners +3–4°, fill −1.0, background +⅙ stop   |

These dials apply on top of the Frame-B anchor.

---

## 6. Validation Harness — Measuring Efficacy

### Perceptual QA (pre-ship)
1. **Geometry gate:** auto-score eye-line, gutters, brand band, and chest band. Pass only if within thresholds.
2. **Texture gate:** verify no >10% local flatness increase from noise reduction; pores remain at 100% view.

### Field testing (post-ship)
- **A/B/C bio page test:** hero vs. warmer vs. cooler variants; track CTR to “Contact” and dwell time.
- **Pairwise forced-choice study:** 30 neutral raters score C/W/K on 0–100 sliders.
- **Success target:** +8–12% CTR uplift vs. prior portrait; C & W ≥ 60 with K ≥ 65.

### Telemetry (embedded)
- Extend EXIF/XMP keys: `presence.intent.C=0.55`, `presence.temporal.t_star=0.31`, `presence.gutters=0.14`, etc.
- Record provenance: `compiled_by=carolwood_presence_v1.0`.

---

## 7. Governance & Ethics — Keep It Human

- **Intent declaration:** “We portray you as you are at your best,” making explicit that authenticity here is intentional, not incidental.
- **No masking of permanent features;** temporary blemish cleanup only.
- **Audit trail:** retain the five-frame disruption set as documentary evidence.
- **Subject agency:** allow leaders to set their own C:W slider; compile accordingly and record the decision in metadata.
- **Context alignment:** ensure wardrobe, set, and messaging reflect the subject’s lived role so manufactured intent still maps to reality.

---

## 8. Rollout Kit — 20-Minute Field Play

1. **Pre-light (10 min):** set background level, dial key/fill/rim, confirm 4×5 framing grid.
2. **Pose & micro-direction (2 min):** “Chin out 7 mm, down 7 mm; shoulders open; lean 3 cm.” Hands in active rest.
3. **Expression prompts (45 s):** silent yes, think through first steps, breathe out with eyes locked.
4. **Burst & pick (1 min):** capture for t*, confirm geometry, tag hero.
5. **Disruption frames (3–4 min):** capture mid-gesture and reset frames.
6. **Grade & export (5–7 min):** gentle S-curve, micro D&B, lint cleanup, export hero and web sizes, generate metadata.

---

## 9. Field Validation Loop — Prove the Output

- **Bio page A/B/C:** run hero vs. warmth-forward vs. authority-forward variants; target a +8–12% lift to “Contact” CTR.  
- **Forced-choice panel:** 30 neutral raters grade C/W/K (0–100). Require C ≥ 60, W ≥ 60, K ≥ 65 before ship.  
- **Telemetry review:** confirm EXIF/XMP keys capture `presence.intent.*`, `presence.temporal.t_star`, compiler version, and disruption hash.  
- **Feedback intake:** log qualitative subject notes within 24 hours to refine prompts and posture coaching.  
- **Market drift watch:** compare observed perception vs. compiled intent; if the delta exceeds 0.05 in either axis, retune the recipe set.

----

## 10. Why This Settles Authentic vs. Manufactured

The compiler doesn’t fake authenticity—it **intentionally assembles** the version of the leader that stakeholders already rely on. By aligning geometry, light, micro-expression, and time to perceptual truths, the portrait becomes evidence of decisiveness rather than a claim about it. You are not freezing a person; you are freezing a decision.

----

## 11. Market Adaptation — Managing Competence Inflation

- **Monitor drift:** track average compiled C:W mixes over time. If the market normalizes at C ≥ 0.60, recalibrate baselines so the brand still signals above-category presence without sliding into severity.
- **Refresh primitives:** introduce minor expression or lighting variations each quarter (e.g., shift chin translation by ±1 mm) to avoid a homogenized look as competitors imitate the recipe.
- **Scenario differentiation:** keep distinct recipes for authority-heavy vs. warmth-forward deliverables so the language retains expressive range instead of collapsing into a single “default executive.”

---

## 12. Reverse Compilation Safeguards — Sword and Shield

- **Detection awareness:** the same primitives that build presence can profile existing portraits. Assume analysts can reverse-engineer C:W intent from your outputs.
- **Provenance tagging:** embed metadata hashes (`presence.hash`) tying each hero frame to its disruption set so authenticity can be verified without exposing private assets.
- **Anomaly flags:** watch for third-party images that mimic the geometry but not the temporal cue; absence of t* often reads uncanny. Use this to distinguish genuine Carolwood portraits from imitations.

---

## 13. Intentional Decompilation — Authenticity Through Controlled Breakage

- **Purposeful drift:** for leaders who need to telegraph vulnerability or renegade energy, deliberately break one parameter (e.g., widen gutters to 18% or relax lower-lid tension) while keeping others inside spec. Flag the deviation in metadata so stakeholders read it as a chosen signal, not an execution error.
- **Failure drills:** run quarterly sessions where the team compiles both perfect and intentionally imperfect frames to keep sensitivity high for when to deploy each.
- **Story alignment:** only decompile when the narrative supports it (e.g., innovation sprints, philanthropic messaging). Otherwise the break reads as sloppiness.

---

## 14. Cognitive Alignment — Photographing the P300 Moment

- **Neural resonance:** the t* selection window (t₀ + 0.25–0.35 s) aligns with the P300 response—the brain’s decision-confirmation signal. You are literally capturing the instant cognition crystallizes into commitment.
- **Behavioral proof:** referencing this correlation in briefs reassures analytical stakeholders that the temporal protocol isn’t stylistic flourish but perception science.
- **Continuous validation:** periodically review burst footage alongside EEG or eye-tracking studies (where available) to ensure the capture cadence keeps mapping to that cognitive inflection point.
