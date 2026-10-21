"""
Plots the judge_evaluation.py rubric scores as one column (bar) chart per
"pillar" -- all 7 rubric criteria (Zone Identification, Logic Rigor, Feature
Comprehensiveness, Biological Plausibility, Specificity, Grounded Mechanistic
Synthesis, Literature Grounding). Literature Grounding is a half-weighted
bonus category in compute_final_score() (not full-weight like the other 6),
but is still plotted here on the same 0-10 scale for visibility.

Primary Zone (Zone 1) only, per model, averaged across however many of the
prompt cases have a real score for that model. Makes NO API/LLM calls --
reads only the scorecards judge_evaluation.py already wrote.

Each pillar's chart ranks models left-to-right from lowest to highest
average score for THAT pillar specifically, so a model's left-right position
can differ between charts. Model->color mapping stays consistent across all
7 charts so the same model is always the same color regardless of rank.

Output: evaluation_results/pillar_plots/<pillar>.{jpg,pdf}  (7 pillars x 2
formats = 14 files). PDFs embed TrueType text (pdf.fonttype=42) instead of
outlining glyphs, so labels stay selectable/editable in Illustrator/Inkscape.

Usage:
    python plot_evaluation_pillars.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # keep PDF text editable, not outlined
matplotlib.rcParams["svg.fonttype"] = "none"  # keep SVG text live, not outlined
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
OUT_DIR = EVAL_RESULTS_DIR / "pillar_plots"

PILLARS = [
    "Zone Identification score",
    "Logic Rigor",
    "Feature Comprehensiveness",
    "Biological Plausibility",
    "Specificity",
    "Grounded Mechanistic Synthesis",
    "Literature Grounding",
]

# Extra derived series (not a rubric field): plain sum of the pillars EXCLUDING
# Specificity, every remaining pillar at full weight (Literature Grounding is
# NOT half-weighted here, unlike judge_evaluation.py's compute_final_score()).
# Max = 6 x 10 = 60. Computed per-scorecard, then averaged across prompts,
# exactly like the real pillars.
OVERALL_EXCL_SPECIFICITY_LABEL = "Overall rank (excl. Specificity)"
OVERALL_EXCL_SPECIFICITY_WEIGHTS = {
    "Zone Identification score": 1.0,
    "Logic Rigor": 1.0,
    "Feature Comprehensiveness": 1.0,
    "Biological Plausibility": 1.0,
    "Grounded Mechanistic Synthesis": 1.0,
    "Literature Grounding": 1.0,
}

# Extra derived series: plain sum of ALL 7 pillars (Specificity included), every pillar at full weight
# (Literature Grounding is NOT half-weighted here, same convention as the series above). Max = 7 x 10 = 70.
OVERALL_ALL7_LABEL = "Overall rank (all 7 pillars)"
OVERALL_ALL7_WEIGHTS = {p: 1.0 for p in PILLARS}


def model_short_name(model_slug: str) -> str:
    return model_slug.split("__", 1)[-1]


_REGISTRY_CACHE: dict | None = None


def model_label_with_size(model_slug: str) -> str:
    """'Kimi-K3 (2.8T)': short name + total parameters from model_registry.json.
    The anchor (gemini) has no disclosed size; a model missing from the
    registry is flagged on stderr and labelled 'size ?', never guessed."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from model_registry import load_registry

        reg = load_registry()
        _REGISTRY_CACHE = {
            "anchor": reg["anchor_slug"],
            "total_b": {m["slug"]: m["total_b"] for m in reg["models"] if m["slug"]},
        }
    name = model_short_name(model_slug)
    if model_slug == _REGISTRY_CACHE["anchor"]:
        return f"{name} (size n/a)"
    b = _REGISTRY_CACHE["total_b"].get(model_slug)
    if b is None:
        print(f"WARNING: {model_slug} is not in model_registry.json; labelled 'size ?'.", file=sys.stderr)
        return f"{name} (size ?)"
    return f"{name} ({b / 1000:g}T)" if b >= 1000 else f"{name} ({b:g}B)"


def load_pillar_averages(
    exclude_prompts: list[str] | None = None,
    models: list[str] | None = None,
    eval_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """{pillar: {model_slug: mean zone1 score across available prompts}}

    exclude_prompts: substrings matched against the scorecard's prompt_file
    (case-insensitive) -- any scorecard matching one is skipped entirely.
    models: restrict to these model subfolder names (default: all).
    eval_dir: directory to read <model>/*_scorecard.json from (default: EVAL_RESULTS_DIR)."""
    exclude_prompts = exclude_prompts or []
    eval_dir = eval_dir or EVAL_RESULTS_DIR
    model_dirs = sorted(d for d in eval_dir.iterdir() if d.is_dir())
    if models is not None:
        wanted = set(models)
        model_dirs = [d for d in model_dirs if d.name in wanted]
    per_model_scores: dict[str, dict[str, list[float]]] = {}
    for model_dir in model_dirs:
        for scorecard_path in sorted(model_dir.glob("*_scorecard.json")):
            data = json.loads(scorecard_path.read_text(encoding="utf-8"))
            prompt_file = data.get("prompt_file", scorecard_path.name)
            if any(ex.lower() in prompt_file.lower() for ex in exclude_prompts):
                continue
            scores = data.get("scores") or {}
            for pillar in PILLARS:
                val = scores.get(f"{pillar}_zone1")
                if val is None:
                    continue
                per_model_scores.setdefault(model_dir.name, {}).setdefault(pillar, []).append(val)

            # Derived series: sum of the weighted pillars (excl. Specificity)
            # for THIS scorecard, only if every weighted pillar has a value.
            weighted_vals = {p: scores.get(f"{p}_zone1") for p in OVERALL_EXCL_SPECIFICITY_WEIGHTS}
            if all(v is not None for v in weighted_vals.values()):
                overall = sum(OVERALL_EXCL_SPECIFICITY_WEIGHTS[p] * v for p, v in weighted_vals.items())
                per_model_scores.setdefault(model_dir.name, {}).setdefault(
                    OVERALL_EXCL_SPECIFICITY_LABEL, []
                ).append(overall)

            # Derived series: sum of all 7 pillars for THIS scorecard (same all-present rule).
            all7_vals = {p: scores.get(f"{p}_zone1") for p in OVERALL_ALL7_WEIGHTS}
            if all(v is not None for v in all7_vals.values()):
                overall7 = sum(OVERALL_ALL7_WEIGHTS[p] * v for p, v in all7_vals.items())
                per_model_scores.setdefault(model_dir.name, {}).setdefault(OVERALL_ALL7_LABEL, []).append(overall7)

    result: dict[str, dict[str, float]] = {
        pillar: {} for pillar in PILLARS + [OVERALL_EXCL_SPECIFICITY_LABEL, OVERALL_ALL7_LABEL]
    }
    for model, pillar_map in per_model_scores.items():
        for pillar, vals in pillar_map.items():
            if vals:
                result[pillar][model] = sum(vals) / len(vals)
    return result


def plot_pillar(
    pillar: str,
    scores_by_model: dict[str, float],
    color_by_model: dict[str, tuple],
    out_stem: Path,
    subtitle_suffix: str = "",
    ylim_max: float = 10,
) -> None:
    ordered = sorted(scores_by_model.items(), key=lambda kv: kv[1])
    models = [m for m, _ in ordered]
    values = [v for _, v in ordered]
    labels = [model_label_with_size(m) for m in models]
    colors = [color_by_model[m] for m in models]

    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(labels) + 2), 6.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, ylim_max)
    ax.set_ylabel(f"Score (0-{ylim_max:g})")
    ax.set_title(
        f"{pillar} -- Primary Zone (Zone 1)\n"
        f"averaged across prompt cases, ranked low to high{subtitle_suffix}"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    label_offset = ylim_max * 0.015
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + label_offset, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".jpg"), dpi=200)
    fig.savefig(out_stem.with_suffix(".svg"))  # live text (svg.fonttype=none), editable in Illustrator
    plt.close(fig)
    print(f"Saved: {out_stem.with_suffix('.pdf')} / .jpg / .svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--exclude-prompts", nargs="+", default=None,
        help="Substrings matched against each prompt_file (case-insensitive); any matching prompt's "
             "scorecards are left out of the averages entirely, across all models.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=f"Output directory for the plots (default: {OUT_DIR}). Use a fresh folder for "
             "one-off/test runs so the default plots aren't overwritten.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model subfolder names to include (default: all models under evaluation_results).",
    )
    parser.add_argument(
        "--exclude-models", nargs="+", default=None,
        help="Model subfolder names to leave out of every plot (applied on top of --models).",
    )
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help=f"Directory to read <model>/*_scorecard.json from (default: {EVAL_RESULTS_DIR}). "
             "Point this at a judge_evaluation.py --out-dir folder to plot a rubric-revision batch.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir) if args.eval_dir else EVAL_RESULTS_DIR
    if not eval_dir.is_dir():
        raise SystemExit(f"ERROR: {eval_dir} not found -- run judge_evaluation.py first.")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    subtitle_suffix = f"\n(excluding: {', '.join(args.exclude_prompts)})" if args.exclude_prompts else ""

    selected_models = args.models
    if args.exclude_models:
        excluded = set(args.exclude_models)
        pool = args.models if args.models is not None else [d.name for d in eval_dir.iterdir() if d.is_dir()]
        selected_models = [m for m in pool if m not in excluded]

    pillar_data = load_pillar_averages(args.exclude_prompts, selected_models, eval_dir)
    all_models = sorted({m for pillar_scores in pillar_data.values() for m in pillar_scores})
    if not all_models:
        raise SystemExit(f"ERROR: no zone1 scores found in {eval_dir} (check --exclude-prompts didn't remove everything).")

    palette = matplotlib.colormaps["tab20"].colors
    color_by_model = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}

    for pillar in PILLARS:
        scores_by_model = pillar_data.get(pillar, {})
        if not scores_by_model:
            print(f"SKIP (no data): {pillar}")
            continue
        file_stem = pillar.lower().replace(" ", "_")
        plot_pillar(pillar, scores_by_model, color_by_model, out_dir / file_stem, subtitle_suffix)

    # The total-score bar plot (six criteria without zone identification, six prompts) is made by
    # 5_evaluation_results/evaluation_results_rubric_v12/total_score_6cases_excl_zone_identification/plot_total_score_barplot.py

    print(f"\nDone. Plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
