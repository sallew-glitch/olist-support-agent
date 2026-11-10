"""Chart for the write-up: how many q3 answers contain the word "never", per model x note condition.

Usage (needs matplotlib; if the project venv lacks it, run with a Python that has it):
    python scripts/make_visual.py
    python scripts/make_visual.py --experiment traces/experiment_2026-11-09

Every count and every circle comes from the traces through agent.experiment: says_never() is the same
function the analysis script's mechanical "never" count uses, and hint_exposed() is the same exposure logic.
Before drawing, the counts are compared with the analysis output; any difference stops the script.
The note text in the caption is read from agent.tools.DELIVERY_NOTE.

Reads only. Writes assets/never_counts_q3.png (1080 px) and assets/never_counts_q3@2x.png (2160 px).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import agent  # noqa: F401
except ImportError:  # running with a Python outside the project venv (e.g. one that has matplotlib)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.config import ROOT
from agent.experiment import (
    analyze, completed_entries, hint_exposed, load_events, read_csv, read_manifest, run_end, says_never,
    tools_called,
)
from agent.tools import DELIVERY_NOTE

QUESTION = "q3"
DELIVERY_TOOL = "get_delivery_performance"
# Rows top to bottom: grouped by model, the note-on row above the note-off row.
ROW_ORDER = [("claude-sonnet-5-5", "on"), ("claude-sonnet-5-5", "off"), ("claude-haiku-5-5", "on"), ("claude-haiku-5-5", "off")]


@dataclass(frozen=True)
class Run:
    repeat: int
    says_never: bool       # agent.experiment.says_never on the final answer
    exposed: bool          # agent.experiment.hint_exposed: a tool's raw result carried a hint field
    called_delivery: bool  # the run called get_delivery_performance
    effort: str | None     # thinking effort sent with the requests (from run_start.request_params)


def extract(folder: Path, question_id: str = QUESTION) -> dict[tuple[str, str], list[Run]]:
    """Per (model, hints) cell, the completed runs of question_id, ordered by repeat."""
    cells: dict[tuple[str, str], list[Run]] = {}
    for entry in completed_entries(read_manifest(folder)).values():
        if entry["question_id"] != question_id:
            continue
        events = load_events(folder / entry["trace_file"])
        output_config = (events[0].get("request_params") or {}).get("output_config") or {}
        cells.setdefault((entry["model"], entry["hints"]), []).append(Run(
            repeat=entry["repeat"],
            says_never=says_never(run_end(events)["final_answer"]),
            exposed=hint_exposed(events),
            called_delivery=DELIVERY_TOOL in tools_called(events),
            effort=output_config.get("effort"),
        ))
    return {cell: sorted(runs, key=lambda r: r.repeat) for cell, runs in cells.items()}


def analysis_never_counts(sheet: list[dict], key: list[dict], question_id: str = QUESTION) -> dict[tuple[str, str], tuple[int, int]]:
    """(never, answers) per (model, hints) for question_id, parsed from the analysis output."""
    lines = analyze(sheet, key)
    start = next(i for i, line in enumerate(lines) if line.startswith("Mechanical, exploratory")) + 2
    counts = {}
    for line in lines[start:]:
        parts = line.split()
        if len(parts) != 6 or not parts[0].startswith("claude-"):
            break
        model, hints, qid, answers, never, _ = parts
        if qid == question_id:
            counts[(model, hints)] = (int(never), int(answers))
    return counts


def model_label(model_id: str) -> str:
    """claude-sonnet-5-5 -> Sonnet 5.5"""
    _, family, *version = model_id.split("-")
    return f"{family.capitalize()} {'.'.join(version)}"


def check(cells: dict, expected: dict) -> list[str]:
    """Problems that must stop the script (empty list = fine)."""
    problems = []
    if set(cells) != set(ROW_ORDER):
        problems.append(f"cells found {sorted(cells)} != expected {ROW_ORDER}")
    for cell, runs in cells.items():
        mine = (sum(r.says_never for r in runs), len(runs))
        if expected.get(cell) != mine:
            problems.append(f"{cell}: chart counts {mine} != analysis counts {expected.get(cell)}")
        for r in runs:
            if r.exposed != r.called_delivery:  # for q3 the only hint-bearing tool is the delivery tool
                problems.append(f"{cell} repeat {r.repeat}: exposed={r.exposed} but called_delivery={r.called_delivery}")
    efforts = {r.effort for runs in cells.values() for r in runs}
    if len(efforts) != 1 or None in efforts:
        problems.append(f"expected one recorded thinking effort across runs, found {efforts}")
    return problems


def texts(cells: dict[tuple[str, str], list[Run]]) -> dict[str, str]:
    """All text on the image. Numbers are computed from the cells."""
    (n,) = {len(runs) for runs in cells.values()}
    (effort,) = {r.effort for runs in cells.values() for r in runs}
    haiku, sonnet = ROW_ORDER[2][0], ROW_ORDER[0][0]
    called = {hints: sum(r.called_delivery for r in cells[(haiku, hints)]) for hints in ("on", "off")}
    if called["on"] == called["off"]:
        called_text = f"{model_label(haiku).split()[0]} called the delivery tool in {called['on']} of {n} runs per condition."
    else:
        called_text = (f"{model_label(haiku).split()[0]} called the delivery tool in {called['on']} of {n} runs with the note on "
                       f"and {called['off']} of {n} with it off.")
    return {
        "title": "Answers saying \u201cnever\u201d about a canceled order",
        "subtitle": f"Delivery and carrier dates were blank in the data. {n} runs per cell.",
        "caption": f"The note: \u201c{DELIVERY_NOTE}\u201d",
        "fine_print": (f"Claude {model_label(haiku)} and {model_label(sonnet)}, {effort} thinking effort. {called_text} "
                       "Word count; the word was chosen after reading the answers. "
                       "Data: Olist (Kaggle), CC BY-NC-SA 4.0. Demo on public data, not production."),
        "legend_filled": "says \u201cnever\u201d",
        "legend_hollow": "does not",
        "legend_dashed": "delivery tool not called",
    }


# --- drawing -----------------------------------------------------------------

SIZE = 1080
MARGIN = 72
BG = "#FFFFFF"
INK = "#1F2328"        # dark gray: text, hollow and dashed outlines
MUTED = "#57606A"      # secondary text
RULE = "#D0D7DE"
ACCENT = "#0B5CAD"     # the one accent color: filled circles
FONTS = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf"}
FONT_DIR = Path("C:/Windows/Fonts")


def render(cells: dict[tuple[str, str], list[Run]], t: dict[str, str], out: Path, out_2x: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle

    missing = [f for f in FONTS.values() if not (FONT_DIR / f).exists()]
    if missing:
        sys.exit(f"Fonts not found in {FONT_DIR}: {missing}")
    font = {k: FontProperties(fname=str(FONT_DIR / f)) for k, f in FONTS.items()}

    dpi = 100
    fig = plt.figure(figsize=(SIZE / dpi, SIZE / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)  # y grows downwards, in pixels
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    pt = lambda px: px * 72 / dpi  # font size in pixels -> points
    layout = []

    def width(s, px, weight):
        tmp = ax.text(0, 0, s, fontproperties=font[weight], fontsize=pt(px))
        w = tmp.get_window_extent(renderer).width
        tmp.remove()
        return w

    def greedy(s, px, weight, max_w):
        lines, line = [], ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if line and width(trial, px, weight) > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + [line]

    def wrap(s, px, weight, max_w):
        """Balanced wrap: the fewest lines that fit, made as even as possible (no one-word last line)."""
        n = len(greedy(s, px, weight, max_w))
        lo, hi = max(width(w, px, weight) for w in s.split()), max_w
        while hi - lo > 2:
            mid = (lo + hi) / 2
            if len(greedy(s, px, weight, mid)) <= n:
                hi = mid
            else:
                lo = mid
        return greedy(s, px, weight, hi)

    def block(name, s, y, px, weight="regular", color=INK, line_height=1.22):
        lines = wrap(s, px, weight, SIZE - 2 * MARGIN)
        for i, line in enumerate(lines):
            ax.text(MARGIN, y + i * px * line_height, line, fontproperties=font[weight], fontsize=pt(px),
                    color=color, va="top", ha="left")
        h = len(lines) * px * line_height
        layout.append((name, round(y), round(y + h), len(lines)))
        return y + h

    def dot(x, y, r, filled, dashed):
        ax.add_patch(Circle((x, y), r, facecolor=ACCENT if filled else BG,
                            edgecolor=INK if (dashed or not filled) else ACCENT, linewidth=3.2,
                            linestyle=(0, (3.2, 2.6)) if dashed else "solid", zorder=3))

    y = MARGIN - 6
    y = block("title", t["title"], y, 56, "bold", line_height=1.12) + 16
    y = block("subtitle", t["subtitle"], y, 36, "regular", MUTED) + 22
    y = block("caption", t["caption"], y, 34, "semibold") + 30

    # Chart: model label spans its two rows; then "note on/off"; then 5 circles; count right-aligned.
    row_h, group_gap, r = 78, 30, 26
    groups = [ROW_ORDER[:2], ROW_ORDER[2:]]
    model_w = max(width(model_label(m), 36, "bold") for m, _ in ROW_ORDER)
    note_w = max(width(f"note {h}", 36, "regular") for _, h in ROW_ORDER)
    n_max = max(len(runs) for runs in cells.values())
    count_w = max(width(f"{n_max}/{n_max}", 64, "bold"), width("0/0", 64, "bold"))
    x_note = MARGIN + model_w + 26
    x_dot0 = x_note + note_w + 34 + r
    gap = 18
    x_count_right = SIZE - MARGIN
    dots_right = x_dot0 + (n_max - 1) * (2 * r + gap) + r
    if dots_right + 30 + count_w > x_count_right:
        sys.exit(f"Chart too wide: dots end at {dots_right:.0f}, counts need {count_w:.0f} px before {x_count_right}.")
    chart_top = y
    for gi, group in enumerate(groups):
        if gi:
            ax.plot([MARGIN, SIZE - MARGIN], [y + group_gap / 2] * 2, color=RULE, linewidth=2)
            y += group_gap
        ax.text(MARGIN, y + row_h, model_label(group[0][0]), fontproperties=font["bold"], fontsize=pt(36),
                color=INK, va="center", ha="left")
        for model, hints in group:
            cy = y + row_h / 2
            runs = cells[(model, hints)]
            ax.text(x_note, cy, f"note {hints}", fontproperties=font["regular"], fontsize=pt(36), color=INK,
                    va="center", ha="left")
            for i, run in enumerate(runs):
                dot(x_dot0 + i * (2 * r + gap), cy, r, run.says_never, not run.exposed)
            ax.text(x_count_right, cy, f"{sum(run.says_never for run in runs)}/{len(runs)}", fontproperties=font["bold"],
                    fontsize=pt(64), color=INK, va="center_baseline", ha="right")
            y += row_h
    layout.append(("chart", round(chart_top), round(y), 4))
    y += 26

    # Legend: one line if it fits, otherwise wrapped item by item.
    lx, ly, lr, lpx = MARGIN + 14, y + 22, 14, 30
    items = [(t["legend_filled"], True, False), (t["legend_hollow"], False, False), (t["legend_dashed"], False, True)]
    legend_top = y
    for label, filled, dashed in items:
        item_w = 2 * lr + 12 + width(label, lpx, "regular")
        if lx + item_w > SIZE - MARGIN:
            lx, ly = MARGIN + 14, ly + 44
        dot(lx + lr - 14, ly, lr, filled, dashed)
        ax.text(lx + 2 * lr - 2, ly, label, fontproperties=font["regular"], fontsize=pt(lpx), color=INK,
                va="center_baseline", ha="left")
        lx += item_w + 34
    y = ly + 22
    layout.append(("legend", round(legend_top), round(y), 1 + round((ly - legend_top - 22) / 44)))
    y += 24

    y = block("fine_print", t["fine_print"], y, 26, "regular", MUTED, line_height=1.25)

    if y > SIZE - MARGIN + 16:
        sys.exit(f"Content runs to y={y:.0f}, past the bottom margin ({SIZE - MARGIN}). Layout: {layout}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=BG)
    fig.savefig(out_2x, dpi=2 * dpi, facecolor=BG)
    plt.close(fig)
    return [f"{name:<11} y {top:>4}-{bottom:<4} lines {n}" for name, top, bottom, n in layout] + [f"bottom of content: {y:.0f} (limit {SIZE - MARGIN})"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, default=ROOT / "traces" / "experiment_2026-11-09")
    parser.add_argument("--scoring-dir", type=Path, default=ROOT / "scoring")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "never_counts_q3.png")
    args = parser.parse_args(argv)

    cells = extract(args.experiment)
    expected = analysis_never_counts(read_csv(args.scoring_dir / "sheet.csv"), read_csv(args.scoring_dir / "key.csv"))
    print(f"q3 answers containing \u201cnever\u201d (chart vs analysis):")
    for cell in ROW_ORDER:
        runs = cells.get(cell, [])
        circles = " ".join(("F" if r.says_never else "H") + ("" if r.exposed else "-dashed") for r in runs)
        print(f"  {model_label(cell[0]):<11} note {cell[1]:<4} chart {sum(r.says_never for r in runs)}/{len(runs)}"
              f" | analysis {expected.get(cell)} | runs by repeat: {circles}")
    problems = check(cells, expected)
    if problems:
        print("STOP: the chart data does not match:\n  " + "\n  ".join(problems))
        return 1
    t = texts(cells)
    out_2x = args.out.with_name(args.out.stem + "@2x" + args.out.suffix)
    for line in render(cells, t, args.out, out_2x):
        print("  " + line)
    print(f"Wrote {args.out} and {out_2x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
