"""
Blind seven-pillar rubric judging via OpenAI, adapted from judge_prompt_thinking.py
(originally written for Gemini) to run over the multi-model trial in
trial_results/.

For each trial_results/<model_slug>/<prompt_stem>_selected_answer.json
(produced by select_best_answer.py), this script:
  1. Reconstructs the exact prompt the model was given, from
     retrain/prompt/<prompt_stem>.txt (same parser as official_trial.py).
  2. Sends ONLY the prompt + the single selected answer to the judge model --
     no model name, no other candidates, no reasoning traces -- so scoring is
     blind and cheap (one answer read per grading call).
  3. Uses the exact same rubric/scoring formula as judge_prompt_thinking.py.
  4. Saves a per-prompt scorecard into:
         evaluation_results/<model_slug>/<prompt_stem>_scorecard.json
     and appends a row to the aggregate:
         evaluation_results/all_scores.csv
     (the CSV includes the model name for our own bookkeeping -- the judge
     itself never sees it).

Usage:
    python judge_evaluation.py                 # judge every selected answer
    python judge_evaluation.py --models MODEL_SLUG [MODEL_SLUG ...]
    python judge_evaluation.py --overwrite      # redo combos already scored
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRIAL_RESULTS_DIR = SCRIPT_DIR / "trial_results"
PROMPT_DIR = SCRIPT_DIR.parent / "prompt"
EVAL_RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
AGGREGATE_CSV = EVAL_RESULTS_DIR / "all_scores.csv"

JUDGE_MODEL = "gpt-5.6-sol"

SYSTEM_HEADER = "--- SYSTEM PROMPT ---"
USER_START_HEADER = "--- EXPERIMENT CONTEXT & DATA PIPELINE ---"

# Identical rubric to judge_prompt_thinking.py (originally written for Gemini),
# just ported to a plain chat-completions call.
JUDGE_SYSTEM_PROMPT = """You are an expert computational biology reviewer grading a model's interpretation of decision-tree rules. You will receive the exact prompt given to the model, followed by its complete submission. The model's identity is hidden.

Grade carefully and consistently. Do not reward confident tone, polished formatting, length, citations, jargon, or molecular detail by themselves.

Use integer scores from 0 to 10. Only evaluate the response's primary zone -- the zone it presents as its main finding. Ignore any secondary or supplementary zones the response also discusses.

GENERAL BOUNDARIES

Construct an independent structural reference for Zone Identification and Logic Rigor. For the remaining criteria, evaluate the submitted interpretation on its own terms rather than requiring agreement with a preferred mechanism.

A decision tree establishes conditional associations, not a unique causal mechanism. A novel hypothesis may receive high scores when it reasonably extends accurate biological knowledge.

BEFORE GRADING

1. Read the FOCUSED ZONES section and use the raw tree only when needed to verify the complete path.
2. Identify the best-supported zone from its Precision and True Positive counts.
3. Record the reference node, predicted state, complete nonredundant conditions, Precision, and True Positives.
4. For the submitted zone, compare its stated conditions with the actual path leading to that node.
5. Identify the required biological items:
   - the target marker or response;
   - the focus feature;
   - each distinct contextual feature defining the zone.

This normally produces six items: one target, one focus feature, and four contextual features. Count repeated thresholds on the same feature only once.

SEVEN-CRITERION RUBRIC

1. ZONE IDENTIFICATION

Evaluate whether the submission selected the appropriate decision-tree zone. Score condition extraction under Logic Rigor.

- 10: Selects the reference node or terminal leaf and assigns the correct predicted state.
- 5: Selects a different but reasonable zone because it has unusually high Precision or unusually high True Positives.
- 0: Selects an unsupported zone, assigns the wrong predicted state, or does not identify a zone.

Use only 10, 5, or 0.

2. LOGIC RIGOR

Evaluate whether the response correctly extracts and consistently preserves the mathematical rule for the selected node.

Count each of the following as one condition error:

- reversing an inequality;
- omitting a nonredundant condition;
- adding a condition not present on the path;
- assigning a condition to the wrong feature;
- reporting a materially incorrect threshold;
- later using the opposite condition in the rationale, hypothesis, prediction, or experiment.

A weaker repeated threshold may be omitted when a stronger threshold on the same feature mathematically implies it. Descriptions such as â€œhigher,â€ â€œabove the split,â€ â€œlower,â€ or â€œbelow the splitâ€ are acceptable when they preserve the correct direction.

- 10: Correct reference zone and no condition errors.
- 8: One condition error.
- 5: Two condition errors.
- 3: Three or more condition errors, but part of the rule remains recognizable.
- 0: No meaningful rule is extracted, or the interpretation is fundamentally opposite to the tree.

If the response selects a different zone from the reference, compare its conditions with the actual path of that submitted node, but the Logic Rigor score cannot exceed 6.

Biological errors do not reduce Logic Rigor unless they cause the response to reverse or misuse a tree condition.

3. FEATURE COMPREHENSIVENESS

Evaluate whether every required biological item is addressed and assigned an intelligible role. Evaluate factual correctness separately.

For each required item, assign 0â€“2 credit:

- 2: Gives an item-specific biological or contextual role and relates it to the target association, zone, cell state, or hypothesis.
- 1.5: Gives a meaningful role, but its relevance to the zone or hypothesis is incomplete.
- 1: Names the item and describes its condition or general identity, but gives little explanation of its role.
- 0.5: Refers to the item only vaguely or indirectly.
- 0: Omits the item.

A threshold restatement, marker label, or generic statement such as â€œis involved in regulationâ€ cannot receive more than 1 point.

Calculate:

Feature Comprehensiveness =
10 Ã— [sum of item credits / (2 Ã— number of required items)]

Round the final value to the nearest integer, with 0.5 rounded upward.

4. BIOLOGICAL PLAUSIBILITY

Evaluate whether the hypothesis is biologically defensible in the stated cell type and experimental context. Evaluate biological accuracy and defensibility, not explanatory depth, feature coverage, or mechanistic resolution.

Score Components A and B independently, add them, and then apply any hallucination ceiling.

A. Accuracy and support of the biological foundation: 0â€“8

Evaluate whether the established biological roles assigned to the focus feature, target, and contextual features are accurate and context-relevant, and whether the proposed hypothesis is a biologically reasonable extension of those roles.

The complete hypothesis does not require direct literature support. A novel, indirect, or unmeasured relationship may receive high credit when it is compatible with established biology and supported by a reasonable biological rationale.

- 8â€“7: The biological foundation is strong or mostly reliable. The important feature functions and contextual assignments are generally accurate, and the proposed hypothesis is biologically defensible. Use the lower end only when there is a genuine biological simplification, questionable feature assignment, or weakness in contextual relevance. Do not lower the score merely because the exact relationship is indirect, unproven, or incompletely resolved.

- 6â€“3: The biological foundation is partially plausible to weak. Some relevant premises are correct, but one or more important feature functions, contextual assignments, or biological relationships are unsupported, incorrectly generalized, or inaccurate. Use the higher end when a substantial plausible core remains. Use the lower end when the hypothesis depends primarily on poorly grounded biological assumptions.

- 2â€“1: Most central premises are invented, incorrectly applied, or inconsistent with established biology, although limited relevant biological information remains.

- 0: The biological foundation is meaningless or incompatible with the stated biological system.

Missing intermediate steps, incomplete pathway detail, and insufficient integration must not reduce Component A unless they make the proposed relationship biologically indefensible. Evaluate those limitations under Hypothesis Integration and Specificity.

B. Overall biological coherence: 0â€“2

Evaluate whether the biological premises can coexist in a reasonable explanation of the featureâ€“target association. Do not evaluate how detailed or fully resolved the mechanism is.

- 2: The overall hypothesis is biologically coherent and defensible. Direct, indirect, common-cause, permissive-context, cell-state, and marker/proxy explanations may receive full credit. Unmeasured or unspecified intermediates are acceptable when the proposed relationship remains reasonable.

- 1: The hypothesis contains relevant biological relationships, but one central relationship is biologically strained, internally inconsistent, or poorly connected to the target.

- 0: No biologically coherent explanation of the target association is provided.

Biological Plausibility score = A + B.

Do not penalize the same limitation under both Components A and B. An indirect or incompletely specified relationship belongs under Hypothesis Integration and Specificity unless it is biologically unreasonable.

After calculating A + B, apply the hallucination ceilings below.

HALLUCINATION CEILINGS

A material hallucination is a clearly false or invented feature function, biological interaction, pathway relationship, or cell-specific fact on which the principal mechanism depends.

Do not classify a claim as a hallucination merely because it is novel, indirect, unmeasured, absent from the judgeâ€™s preferred interpretation, or not demonstrated in the exact experimental context. A ceiling applies only when the claim is clearly incompatible with established biology or has no defensible biological basis.

- One central material hallucination: maximum score 6.
- Two or more independent central material hallucinations: maximum score 3.
- A mechanism based primarily on hallucinated biology: maximum score 1.

A claim is central only when removing it causes the principal mechanism to fail. Peripheral errors, optional mechanisms, secondary extensions, and experimental suggestions do not trigger a central ceiling.

Multiple statements arising from the same underlying misconception count as one material hallucination.

A hypothesis cannot receive high plausibility merely because false premises form an internally coherent story.

Also provide "Biological Plausibility_reason": a concise 1â€“2 sentence explanation naming the main biological strength or weakness determining the score. If a ceiling was applied, identify the specific claim that triggered it.

5. HYPOTHESIS INTEGRATION AND SPECIFICITY

Evaluate the depth, resolution, and integration of the proposed biological hypothesis. Evaluate factual correctness under Biological Plausibility and Literature Grounding.

- 10: Develops a detailed, cohesive, multi-step mechanism. The focus feature, target, and most contextual features make distinct and material contributions to the same explanatory model, and the biological steps connecting them are clearly explained.
- 8: Provides a specific, connected mechanism incorporating most relevant features. One feature or mechanistic transition remains broad or incompletely integrated.
- 6: Provides a meaningful central mechanism, but only some contextual features materially contribute. Other features are explained separately or used mainly as background or state markers.
- 4: Primarily presents a feature-by-feature account or isolated pairwise relationships. Removing most contextual features would leave the central hypothesis essentially unchanged.
- 2: Provides only a broad pathway, cellular state, generic causal suggestion, or isolated featureâ€“target relationship.
- 0: Provides no substantive hypothesis beyond restating the tree or listing feature functions.

Use odd-numbered scores when the response falls between anchors.

Apply these rules:

- Count each distinct explanatory contribution once.
- Repetition, paraphrasing, citations, pathway names, and molecular terminology do not increase the score.
- Placing multiple feature functions in the same paragraph does not establish integration.
- Generic connecting phrases demonstrate integration only when the biological relationship is explained.
- Contextual features may function as regulators, mediators, permissive conditions, markers, proxies, parallel processes, or downstream readouts, but their relevance to the hypothesis must be clear.
- If removing most contextual features leaves the mechanism essentially unchanged, the score cannot exceed 4.
- If the focus feature is connected to the target but most contextual features remain labels or background descriptions, the score cannot exceed 6.
- Scores of 8â€“10 require most contextual features to materially shape, constrain, or explain the central hypothesis.
- Experiments, citations, presentation quality, and answer length do not increase this score.

6. MECHANISTIC SYNTHESIS

Evaluate how effectively the response composes its biological premises into a connected, integrated, and sufficiently detailed explanation of the focus-featureâ€“target association.

Before scoring, internally identify:

- The focus feature and target;
- The contextual features;
- Any intermediate processes or cellular states; and
- The relationships explicitly proposed among them.

Use Biological Plausibility only to identify material hallucinations. Do not transfer the Biological Plausibility score or its deductions into this criterion. A novel, indirect, unmeasured, or weakly supported relationship may contribute fully to synthesis when the response explains it as a biologically reasonable hypothesis. Exclude a relationship only when Biological Plausibility identifies it as clearly false, invented, or biologically indefensible.

A contextual feature counts as integrated when the response explains its biological role and how that role helps create, modify, constrain, or identify the condition in which the focus feature is associated with the target. A feature may contribute as a regulator, mediator, intermediate, permissive condition, state marker, proxy, parallel process, or downstream readout. It does not need to interact directly with the focus feature.

Mechanistic depth requires more than listing feature functions, pathways, or molecular terminology. Credit depth when the response explains the biological transitions connecting the features, intermediate processes, and target association. For an indirect relationship, the response should identify a relevant process, pathway, or cellular state. Do not supply a missing connection on behalf of the response.

- 10â€“8: Strong mechanistic synthesis. The response presents a cohesive, detailed, and sufficiently resolved mechanism connecting the focus feature to the target. Most relevant contextual features make explicit and meaningful contributions, and the important biological transitions or intermediate processes are explained. Use the higher end for a well-developed multi-step mechanism and the lower end when one important transition or contextual contribution remains broad or incomplete.

- 7â€“5: Partial mechanistic synthesis. A meaningful central mechanism is present, but its integration or resolution is incomplete. Some contextual features contribute clearly, while others remain loosely connected, isolated, or primarily descriptive. One or more important transitions may be asserted through broad pathway language without adequately explaining the biological bridge.

- 4â€“2: Weak mechanistic synthesis. The response mainly provides feature-by-feature descriptions, pathway names, isolated pairwise relationships, or a broad cellular-state explanation. A central idea may be present, but few features are connected through explicit biological transitions.

- 1â€“0: Minimal or absent mechanistic synthesis. The response provides little or no substantive mechanism beyond restating the tree or listing biological functions.

Use any integer within these ranges according to the overall integration, resolution, and mechanistic detail of the explanation.

Apply the following ceilings:

- No explicit mechanism connecting the focus feature to the target: maximum 3.
- Fewer than half of the relevant contextual features are explicitly integrated: maximum 5.
- Two or more central transitions are asserted without a relationship, intermediate process, or contextual bridge: maximum 6.
- The response primarily lists feature functions or pathways without explaining how they work together: maximum 5.
- Multiple mechanisms are proposed without explaining how they are compatible or related: maximum 5.
- The principal focus-featureâ€“target mechanism depends on a material hallucination identified under Biological Plausibility: maximum 3.

Apply only the lowest relevant ceiling. Do not impose multiple deductions for the same deficiency.

Count each distinct explanatory contribution once. Repetition, paraphrasing, pathway-name accumulation, citations, terminology, answer length, and multiple similar experiments do not increase the score.

Do not require agreement with the judge's preferred mechanism. A different hypothesis may receive full credit when it is explicitly connected, integrated, sufficiently resolved, and mechanistically detailed.

Also provide "Grounded Mechanistic Synthesis_reason": exactly three concise sentences stating (1) which relationships or contextual features were successfully integrated, (2) which remained isolated or insufficiently connected, and (3) which ceiling, if any, was applied and why.

7. LITERATURE GROUNDING

Evaluate the reliability of the established biological knowledge used to support the hypothesis, including feature functions, pathway relationships, cell-type associations, and claims attributed to publications.

Score the following components and add them.

A. Accuracy and relevance of supporting premises: 0â€“5

- 5: Central supporting facts are accurate, specific, relevant, and correctly applied.
- 4: The foundation is mostly accurate, with one peripheral imprecision or indirect extrapolation.
- 3: Useful knowledge is present, but one important premise is questionable or several premises are overly general.
- 2: Several important feature functions or pathway relationships are inaccurate, unsupported, or incorrectly applied.
- 1: The supporting premises are mostly generic, unreliable, or invented.
- 0: The biological support is predominantly fictional, irrelevant, or fundamentally incorrect.

B. Evidence fidelity and claim calibration: 0â€“5

- 5: Established knowledge, inference, and hypothesis are clearly distinguished. References support the claims attributed to them.
- 4: Evidence is generally well calibrated, with one peripheral mismatch, indirect citation, or mild overstatement.
- 3: The biological foundation remains recognizable, but several sources are indirect or one important speculative relationship is presented too strongly.
- 2: Multiple references or assertions are mismatched, likely inaccurate, or used to support claims they do not establish.
- 1: Citation misuse, likely fabricated references, or unsupported assertions presented as established facts are widespread.
- 0: The evidence base is materially fabricated or unrelated to the interpretation.

Literature Grounding score = A + B.

Do not award credit for citation quantity, journal prestige, detailed formatting, or authoritative wording. Evidence from a related gene family, cell type, or pathway is indirect unless presented as an analogy or extrapolation. An uncited but established biological fact may still receive credit.

OUTPUT FORMAT

Return the final grades in a markdown JSON block using exactly these keys. Scores are integers; "Biological Plausibility_reason" is a short string (1-2 sentences); "Grounded Mechanistic Synthesis_reason" is a short string (exactly 3 sentences):

~~~json
{
  "Zone Identification score_zone1": 0,
  "Logic Rigor_zone1": 0,
  "Feature Comprehensiveness_zone1": 0,
  "Biological Plausibility_zone1": 0,
  "Biological Plausibility_reason": "",
  "Specificity_zone1": 0,
  "Grounded Mechanistic Synthesis_zone1": 0,
  "Grounded Mechanistic Synthesis_reason": "",
  "Literature Grounding_zone1": 0
}
~~~
""".replace('~~~', '```')


CSV_HEADERS = [
    "model",
    "prompt_file",
    "Zone Identification score_zone1",
    "Logic Rigor_zone1",
    "Feature Comprehensiveness_zone1",
    "Biological Plausibility_zone1",
    "Biological Plausibility_reason",
    "Specificity_zone1",
    "Grounded Mechanistic Synthesis_zone1",
    "Grounded Mechanistic Synthesis_reason",
    "Literature Grounding_zone1",
    "final score_zone1",
]


def parse_prompt_file(path: Path) -> tuple[str, str]:
    """Same parsing contract as official_trial.py / pilot_run.py."""
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith(SYSTEM_HEADER):
        raise ValueError(f"'{path.name}' does not start with '{SYSTEM_HEADER}'.")
    after_header = text.lstrip()[len(SYSTEM_HEADER):]
    split_idx = after_header.find(USER_START_HEADER)
    if split_idx == -1:
        raise ValueError(f"Could not find '{USER_START_HEADER}' in '{path.name}'.")
    system_prompt = after_header[:split_idx].strip()
    user_prompt = after_header[split_idx:].strip()
    return system_prompt, user_prompt


def load_client(env_var: str = "OPENAI_API_KEY"):
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(
            f"ERROR: Environment variable {env_var} is not set.\n"
            'Set it with: [System.Environment]::SetEnvironmentVariable('
            f'"{env_var}", "your_api_key", "User")\n'
            "then restart your terminal."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=300.0)


def compute_final_score(scores: dict) -> float | None:
    if scores.get("Zone Identification score_zone1") is None:
        return None
    lit = scores.get("Literature Grounding_zone1") or 0
    return (
        (scores.get("Zone Identification score_zone1") or 0)
        + (scores.get("Logic Rigor_zone1") or 0)
        + (scores.get("Feature Comprehensiveness_zone1") or 0)
        + (scores.get("Biological Plausibility_zone1") or 0)
        + (scores.get("Specificity_zone1") or 0)
        + (scores.get("Grounded Mechanistic Synthesis_zone1") or 0)
        + (lit * 0.5)
    )


def judge_one(client, exam_paper: str, student_answer: str) -> dict:
    """Blind grading call -- reads only the prompt + one answer, no model name."""
    user_content = f"""--- INPUT 1: WHAT THE STUDENT WAS GIVEN ---
{exam_paper}

--- INPUT 2: THE STUDENT'S SUBMISSION ---
{student_answer}
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw_text = response.choices[0].message.content or ""
    json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    clean_json_str = json_match.group(1) if json_match else raw_text.strip().strip("`")
    return json.loads(clean_json_str)


def append_csv_row(row: dict) -> None:
    is_new = not AGGREGATE_CSV.exists()
    with open(AGGREGATE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def process_file(client, selected_path: Path, overwrite: bool) -> bool:
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    model = data["model"]
    prompt_file = data["prompt_file"]
    model_slug = selected_path.parent.name

    out_dir = EVAL_RESULTS_DIR / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_stem = Path(prompt_file).stem
    out_path = out_dir / f"{prompt_stem}_scorecard.json"

    if out_path.exists() and not overwrite:
        print(f"SKIP (already scored): {out_path}")
        return False

    if not data.get("answer"):
        # Selection step found no real answer to grade (e.g. all 3 raw
        # generations failed upstream -- see select_best_answer.py's
        # "all_answers_empty" method). Sending the judge a literal "None"
        # string would get scored as a real (bad) submission, indistinguishable
        # from an actually-attempted-but-poor answer, so skip grading instead
        # and record it as missing (null), not zero.
        print(
            f"SKIP (no answer to grade -- selection_method="
            f"{data.get('selection_method')!r}): {selected_path}",
            file=sys.stderr,
        )
        scorecard = {
            "model": model,
            "prompt_file": prompt_file,
            "scores": None,
            "final_score_zone1": None,
            "note": "No answer available to grade; not sent to the judge.",
        }
        out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        row = {h: None for h in CSV_HEADERS}
        row["model"] = model
        row["prompt_file"] = prompt_file
        append_csv_row(row)
        print(f"  -> no answer, recorded as missing: {out_path}")
        return True

    prompt_path = PROMPT_DIR / prompt_file
    if not prompt_path.is_file():
        print(f"ERROR: original prompt file not found: {prompt_path}", file=sys.stderr)
        raise SystemExit(1)

    system_prompt, user_prompt = parse_prompt_file(prompt_path)
    exam_paper = f"{system_prompt}\n\n{user_prompt}"

    print(f"Judging: {model} / {prompt_file} ...", flush=True)

    try:
        scores = judge_one(client, exam_paper, data["answer"])
    except Exception as exc:  # noqa: BLE001 - surface the real error, no silent fallback
        print(f"ERROR: judging failed for {selected_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    final_zone1 = compute_final_score(scores)

    scorecard = {
        "model": model,
        "prompt_file": prompt_file,
        "scores": scores,
        "final_score_zone1": final_zone1,
    }
    out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    row = {
        "model": model,
        "prompt_file": prompt_file,
        "Zone Identification score_zone1": scores.get("Zone Identification score_zone1"),
        "Logic Rigor_zone1": scores.get("Logic Rigor_zone1"),
        "Feature Comprehensiveness_zone1": scores.get("Feature Comprehensiveness_zone1"),
        "Biological Plausibility_zone1": scores.get("Biological Plausibility_zone1"),
        "Biological Plausibility_reason": scores.get("Biological Plausibility_reason"),
        "Specificity_zone1": scores.get("Specificity_zone1"),
        "Grounded Mechanistic Synthesis_zone1": scores.get("Grounded Mechanistic Synthesis_zone1"),
        "Grounded Mechanistic Synthesis_reason": scores.get("Grounded Mechanistic Synthesis_reason"),
        "Literature Grounding_zone1": scores.get("Literature Grounding_zone1"),
        "final score_zone1": final_zone1,
    }
    append_csv_row(row)

    print(f"  -> zone1={final_zone1}  Saved: {out_path}")
    return True


def main() -> None:
    global EVAL_RESULTS_DIR, AGGREGATE_CSV

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model subfolder names (e.g. 'zai-org__GLM-5.3') to process (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo combos that already have a scorecard (default: skip them).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Subset of prompt stems (no .txt) to process (default: all with a selected answer).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=f"Directory to write scorecards + aggregate CSV into (default: {EVAL_RESULTS_DIR}). "
             "Use a fresh folder for a rubric revision so old scores aren't overwritten.",
    )
    parser.add_argument(
        "--api-key-env", type=str, default="OPENAI_API_KEY",
        help="Name of the environment variable holding the OpenAI API key to use "
             "(default: OPENAI_API_KEY). Use this to switch between multiple keys, "
             "e.g. --api-key-env OPENAI_API_KEY_2.",
    )
    args = parser.parse_args()

    if not TRIAL_RESULTS_DIR.is_dir():
        raise SystemExit(f"ERROR: trial_results folder not found: {TRIAL_RESULTS_DIR}")
    if not PROMPT_DIR.is_dir():
        raise SystemExit(f"ERROR: prompt folder not found: {PROMPT_DIR}")

    if args.out_dir:
        EVAL_RESULTS_DIR = Path(args.out_dir)
        AGGREGATE_CSV = EVAL_RESULTS_DIR / "all_scores.csv"

    client = load_client(args.api_key_env)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in TRIAL_RESULTS_DIR.iterdir() if d.is_dir())
    if args.models:
        wanted = set(args.models)
        model_dirs = [d for d in model_dirs if d.name in wanted]
        missing = wanted - {d.name for d in model_dirs}
        if missing:
            raise SystemExit(f"ERROR: unknown model subfolder(s): {missing}")

    if not model_dirs:
        raise SystemExit(f"ERROR: no model subfolders found under {TRIAL_RESULTS_DIR}")

    done = 0
    skipped = 0
    for model_dir in model_dirs:
        for selected_path in sorted(model_dir.glob("*_selected_answer.json")):
            if args.prompts is not None:
                prompt_file = json.loads(selected_path.read_text(encoding="utf-8"))["prompt_file"]
                if Path(prompt_file).stem not in set(args.prompts):
                    continue
            if process_file(client, selected_path, args.overwrite):
                done += 1
            else:
                skipped += 1

    print(f"\nDone. {done} scorecard(s) written, {skipped} skipped.")
    print(f"Aggregate CSV: {AGGREGATE_CSV}")


if __name__ == "__main__":
    main()
