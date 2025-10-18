// code.ts — Presence Overlay Guides (with command parameters)
// Creates guides + locale prompts on current selection (frame) or a new frame.
// Default locale: US_EN. Change LOCALE to one of: US_EN, JP_JA, DE_DE, CN_ZH, IN_EN, GCC_AR.

type Aspect = "4:5" | "2:3";
type LocaleKey = keyof typeof PROMPTS;

const LOCALE: LocaleKey = "US_EN";

const PROMPTS: Record<string, string[]> = {
  US_EN: ["Silent yes", "What would you do?", "Stay with me"],
  JP_JA: ["Consider carefully", "Small nod", "Soft gaze"],
  DE_DE: ["Confirm plan", "Focus point", "Steady breath"],
  CN_ZH: ["Thought resolved", "Subtle assent", "Calm gaze"],
  IN_EN: ["Assure client", "Yes, I have it", "Warm gaze"],
  GCC_AR: ["Decision held", "Assurance", "Poised gaze"],
};

const ASPECT_CHOICES: Aspect[] = ["4:5", "2:3"];
const LOCALE_CHOICES: LocaleKey[] = ["US_EN", "JP_JA", "DE_DE", "CN_ZH", "IN_EN", "GCC_AR"] as const;

function nearestFrameFromSelection(): FrameNode | undefined {
  const sel = figma.currentPage.selection[0];
  if (!sel) return undefined;
  if (sel.type === "FRAME") return sel;
  let p: BaseNode | null = sel.parent;
  while (p) {
    if (p.type === "FRAME") return p as FrameNode;
    p = (p as any).parent ?? null;
  }
  return undefined;
}

function ensureFrame(aspect: Aspect): FrameNode {
  let frame = nearestFrameFromSelection();
  if (!frame) {
    frame = figma.createFrame();
    frame.name = `Presence Frame ${aspect}`;
    if (aspect === "4:5") frame.resize(2400, 3000);
    else frame.resize(1200, 1800);
    figma.currentPage.appendChild(frame);
  }
  frame.x = Math.round(frame.x);
  frame.y = Math.round(frame.y);
  return frame;
}

function removePreviousGuides(frame: FrameNode) {
  const toRemove: SceneNode[] = [];
  for (const n of frame.children) {
    if (n.type === "GROUP" && n.name.startsWith("Presence Guides")) {
      toRemove.push(n);
    }
  }
  toRemove.forEach((n) => n.remove());
}

async function addGuides(aspect: Aspect, locale: LocaleKey) {
  const frame = ensureFrame(aspect);
  removePreviousGuides(frame);

  const W = frame.width;
  const H = frame.height;

  const groupNodes: SceneNode[] = [];

  // Border
  const border = figma.createRectangle();
  border.resize(W, H);
  border.name = "Presence Border";
  border.strokes = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
  border.strokeWeight = 2;
  border.fills = [];
  border.locked = true;
  groupNodes.push(border);

  // Eye line
  const eyePct = aspect === "4:5" ? 0.27 : 0.36;
  const eye = figma.createRectangle();
  eye.resize(W - 80, 4);
  eye.x = 40;
  eye.y = Math.round(H * eyePct);
  eye.name = `Eye line ${Math.round(eyePct * 100)}%`;
  eye.fills = [{ type: "SOLID", color: { r: 0.3137, g: 0.7843, b: 0.4706 }, opacity: 0.9 }];
  eye.strokes = [];
  groupNodes.push(eye);

  // Gutters (14%)
  const gutter = 0.14;
  const left = figma.createRectangle();
  left.resize(W * gutter, H);
  left.x = 0;
  left.y = 0;
  left.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 }, opacity: 0.12 }];
  left.name = "Left gutter 14%";

  const right = figma.createRectangle();
  right.resize(W * gutter, H);
  right.x = W - W * gutter;
  right.y = 0;
  right.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 }, opacity: 0.12 }];
  right.name = "Right gutter 14%";

  groupNodes.push(left, right);

  // Brand band (top ~12%)
  const brand = figma.createRectangle();
  brand.resize(W, 2);
  brand.x = 0;
  brand.y = Math.round(H * 0.12);
  brand.name = "Brand band ~12%";
  brand.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 }, opacity: 0.5 }];
  brand.strokes = [];
  groupNodes.push(brand);

  // Chest/Hands band
  const b1 = aspect === "4:5" ? 0.68 : 0.78;
  const b2 = aspect === "4:5" ? 0.8 : 0.95;
  const chest = figma.createRectangle();
  chest.resize(W, Math.round(H * b2 - H * b1));
  chest.x = 0;
  chest.y = Math.round(H * b1);
  chest.name = "Chest/Hands band";
  chest.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 }, opacity: 0.1 }];
  chest.strokes = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 }, opacity: 0.35 }];
  chest.strokeWeight = 1;
  groupNodes.push(chest);

  // Prompt text
  const prompts = PROMPTS[locale] || PROMPTS["US_EN"];
  const text = figma.createText();
  text.name = `Expression prompts (${locale})`;
  // Load fonts before setting characters.
  try {
    await figma.loadFontAsync({ family: "Inter", style: "Regular" });
    text.fontName = { family: "Inter", style: "Regular" };
  } catch {
    await figma.loadFontAsync({ family: "Roboto", style: "Regular" });
    text.fontName = { family: "Roboto", style: "Regular" };
  }
  text.characters = `Expression prompts:\n- ${prompts.join("\n- ")}`;
  text.fontSize = 28;
  text.x = 64;
  text.y = 64;
  groupNodes.push(text);

  // Group into the frame
  const group = figma.group(groupNodes, frame);
  group.name = `Presence Guides ${aspect} (${locale})`;
  group.x = 0;
  group.y = 0;

  figma.currentPage.selection = [group];
  figma.notify(`Presence guides added (${aspect}, ${locale})`);
}

/** ---------- Command Parameters UI ---------- */
// Suggest list for aspect/locale in Quick Actions.
figma.parameters.on("input", ({ key, query, result }) => {
  const q = (query ?? "").toLowerCase();
  if (key === "aspect") {
    const sug = ASPECT_CHOICES.filter((a) => a.toLowerCase().includes(q));
    result.setSuggestions(sug.length ? sug : ASPECT_CHOICES);
  } else if (key === "locale") {
    const sug = LOCALE_CHOICES.filter((l) => l.toLowerCase().includes(q));
    result.setSuggestions(sug.length ? sug : LOCALE_CHOICES);
  } else {
    result.setSuggestions([]);
  }
});

/** ---------- Run handler (supports parameters & legacy commands) ---------- */
figma.on("run", async ({ command, parameters }) => {
  // Defaults
  let aspect: Aspect = "4:5";
  let locale: LocaleKey = LOCALE;

  // 1) Parameters from Quick Actions
  const aspParam = (parameters && (parameters as any).aspect) as string | undefined;
  const locParam = (parameters && (parameters as any).locale) as string | undefined;
  if (aspParam && (ASPECT_CHOICES as string[]).includes(aspParam)) {
    aspect = aspParam as Aspect;
  }
  if (locParam && (LOCALE_CHOICES as string[]).includes(locParam)) {
    locale = locParam as LocaleKey;
  }

  // 2) Legacy command mapping (kept for backwards compatibility)
  if (command === "add-guides-2x3") aspect = "2:3";

  await addGuides(aspect, locale);
  figma.closePlugin();
});