"""Blind six-pillar rubric judging of the 12 primary-only DT-setting
hypotheses via OpenAI (gpt-5.6-sol), adapted from
ClassifierEvaluation/judge_evaluation.py (a sibling multi-LLM-model trial's
7-pillar rubric).

Kept from that rubric: Feature Comprehensiveness, Logic Rigor, Biological
Plausibility, Specificity (its "Hypothesis Integration and Specificity"),
Mechanistic Synthesis (its "Grounded Mechanistic Synthesis"), Literature
Grounding. Dropped: Zone Identification. (An earlier run judged the same
hypotheses on five pillars, without Specificity; its scorecards remain in
results/judge_evaluation_openai/.)

Adaptations to fit this project's prompts (the source rubric assumed a
prompt with a "FOCUSED ZONES" section and one reference node):
  - Logic Rigor: no reference-zone comparison and no "different zone caps at
    6" rule (zone choice is not graded here, and our prompts legitimately
    offer many candidate nodes ranked by support/lift/IG). The rule the
    response states is checked against the actual path of the node IT chose.
    Because the prompt tells the model to anchor its hypothesis on the ~4
    most important conditions of a path (per the feature-confidence ranking),
    not mentioning the remaining conditions is NOT counted as an omission.
  - Feature Comprehensiveness: required items are the target, the focus
    feature, and the distinct features among the selected node's top-4
    conditions by the prompt's feature-confidence ranking.
Rubric text for Biological Plausibility, Specificity, Mechanistic Synthesis
and Literature Grounding is unchanged from the source.

Each (dataset, setting) is judged BLIND: the judge sees only the exact
*_primaryonly_prompt.txt that setting was given plus its single hypothesis
(run 1 of *_primaryonly_hypothesis.jsonl) -- never the dataset/setting name.

Requires an OpenAI key in an environment variable (default OPENAI_API_KEY;
use --api-key-env to name a different one). Nothing is hardcoded.

Usage:
    python judge_evaluation.py [--overwrite] [--api-key-env NAME]
"""
import argparse
import json
import os
import re
import sys

import config

JUDGE_MODEL = "gpt-5.6-sol"
OUT_DIR = config.RESULTS_DIR / "judge_evaluation_openai_6pillar"
AGGREGATE_JSON = OUT_DIR / "all_scores.json"

PILLARS = [
    "Feature Comprehensiveness",
    "Logic Rigor",
    "Biological Plausibility",
    "Specificity",
    "Mechanistic Synthesis",
    "Literature Grounding",
]

JUDGE_SYSTEM_PROMPT = """You are an expert computational biology reviewer grading a model's interpretation of decision-tree rules. You will receive the exact prompt given to the model, followed by its complete submission. The model's identity is hidden.

Grade carefully and consistently. Do not reward confident tone, polished formatting, length, citations, jargon, or molecular detail by themselves.

Use integer scores from 0 to 10 (Feature Comprehensiveness is rounded to the nearest integer). The submission discusses only one zone (its primary zone) -- evaluate that.

A decision tree establishes conditional associations, not a unique causal mechanism. A novel hypothesis may receive high scores when it reasonably extends accurate biological knowledge.

BEFORE GRADING

1. Read the CANDIDATE NODES section of the prompt, and use the raw decision tree only when needed to verify a complete path.
2. Identify the node the submission presents as its primary zone. Record that node's actual conditions, exactly as listed in the prompt.
3. Identify the required biological items:
   - the target marker or response;
   - the focus feature;
   - each distinct feature among that node's top 4 conditions by the prompt's "Feature confidence ranking".

Count repeated thresholds on the same feature only once. This normally produces up to six items: one target, one focus feature, and up to four contextual features.

SIX-CRITERION RUBRIC

1. FEATURE COMPREHENSIVENESS

Evaluate whether every required biological item is addressed and assigned an intelligible role. Evaluate factual correctness separately.

For each required item, assign 0-2 credit:

- 2: Gives an item-specific biological or contextual role and relates it to the target association, zone, cell state, or hypothesis.
- 1.5: Gives a meaningful role, but its relevance to the zone or hypothesis is incomplete.
- 1: Names the item and describes its condition or general identity, but gives little explanation of its role.
- 0.5: Refers to the item only vaguely or indirectly.
- 0: Omits the item.

A threshold restatement, marker label, or generic statement such as "is involved in regulation" cannot receive more than 1 point.

Calculate:

Feature Comprehensiveness =
10 x [sum of item credits / (2 x number of required items)]

Round the final value to the nearest integer, with 0.5 rounded upward.

2. LOGIC RIGOR

Evaluate whether the response correctly extracts and consistently preserves the mathematical rule for the node it selected.

Count each of the following as one condition error:

- reversing an inequality;
- omitting a condition the response claims to be reporting as part of the rule (see note below);
- adding a condition not present on the path;
- assigning a condition to the wrong feature;
- reporting a materially incorrect threshold;
- later using the opposite condition in the rationale, hypothesis, prediction, or experiment.

The prompt instructs the model to anchor its hypothesis on the most important (about four) conditions of a path, ranked by the feature-confidence annotation. Not mentioning the remaining lower-ranked conditions is therefore NOT an omission error, as long as the rule the response states does not misrepresent the path. A weaker repeated threshold may also be omitted when a stronger threshold on the same feature mathematically implies it. Descriptions such as "higher," "above the split," "lower," or "below the split" are acceptable when they preserve the correct direction.

- 10: The selected node's rule is extracted correctly, with no condition errors.
- 8: One condition error.
- 5: Two condition errors.
- 3: Three or more condition errors, but part of the rule remains recognizable.
- 0: No meaningful rule is extracted, or the interpretation is fundamentally opposite to the tree.

Biological errors do not reduce Logic Rigor unless they cause the response to reverse or misuse a tree condition.

3. BIOLOGICAL PLAUSIBILITY

Evaluate whether the hypothesis is biologically defensible in the stated cell type and experimental context. Evaluate biological accuracy and defensibility, not explanatory depth, feature coverage, or mechanistic resolution.

Score Components A and B independently, add them, and then apply any hallucination ceiling.

A. Accuracy and support of the biological foundation: 0-8

Evaluate whether the established biological roles assigned to the focus feature, target, and contextual features are accurate and context-relevant, and whether the proposed hypothesis is a biologically reasonable extension of those roles.

The complete hypothesis does not require direct literature support. A novel, indirect, or unmeasured relationship may receive high credit when it is compatible with established biology and supported by a reasonable biological rationale.

- 8-7: The biological foundation is strong or mostly reliable. The important feature functions and contextual assignments are generally accurate, and the proposed hypothesis is biologically defensible. Use the lower end only when there is a genuine biological simplification, questionable feature assignment, or weakness in contextual relevance. Do not lower the score merely because the exact relationship is indirect, unproven, or incompletely resolved.

- 6-3: The biological foundation is partially plausible to weak. Some relevant premises are correct, but one or more important feature functions, contextual assignments, or biological relationships are unsupported, incorrectly generalized, or inaccurate. Use the higher end when a substantial plausible core remains. Use the lower end when the hypothesis depends primarily on poorly grounded biological assumptions.

- 2-1: Most central premises are invented, incorrectly applied, or inconsistent with established biology, although limited relevant biological information remains.

- 0: The biological foundation is meaningless or incompatible with the stated biological system.

Missing intermediate steps, incomplete pathway detail, and insufficient integration must not reduce Component A unless they make the proposed relationship biologically indefensible.

B. Overall biological coherence: 0-2

Evaluate whether the biological premises can coexist in a reasonable explanation of the feature-target association. Do not evaluate how detailed or fully resolved the mechanism is.

- 2: The overall hypothesis is biologically coherent and defensible. Direct, indirect, common-cause, permissive-context, cell-state, and marker/proxy explanations may receive full credit. Unmeasured or unspecified intermediates are acceptable when the proposed relationship remains reasonable.

- 1: The hypothesis contains relevant biological relationships, but one central relationship is biologically strained, internally inconsistent, or poorly connected to the target.

- 0: No biologically coherent explanation of the target association is provided.

Biological Plausibility score = A + B.

After calculating A + B, apply the hallucination ceilings below.

HALLUCINATION CEILINGS

A material hallucination is a clearly false or invented feature function, biological interaction, pathway relationship, or cell-specific fact on which the principal mechanism depends.

Do not classify a claim as a hallucination merely because it is novel, indirect, unmeasured, absent from the judge's preferred interpretation, or not demonstrated in the exact experimental context. A ceiling applies only when the claim is clearly incompatible with established biology or has no defensible biological basis.

- One central material hallucination: maximum score 6.
- Two or more independent central material hallucinations: maximum score 3.
- A mechanism based primarily on hallucinated biology: maximum score 1.

A claim is central only when removing it causes the principal mechanism to fail. Peripheral errors, optional mechanisms, secondary extensions, and experimental suggestions do not trigger a central ceiling.

A hypothesis cannot receive high plausibility merely because false premises form an internally coherent story.

Also provide "Biological Plausibility_reason": a concise 1-2 sentence explanation naming the main biological strength or weakness determining the score. If a ceiling was applied, identify the specific claim that triggered it.

4. HYPOTHESIS INTEGRATION AND SPECIFICITY

Evaluate the depth, resolution, and integration of the proposed biological hypothesis. Evaluate factual correctness under Biological Plausibility and Literature Grounding.

- 10: Develops a detailed, cohesive, multi-step mechanism. The focus feature, target, and most contextual features make distinct and material contributions to the same explanatory model, and the biological steps connecting them are clearly explained.
- 8: Provides a specific, connected mechanism incorporating most relevant features. One feature or mechanistic transition remains broad or incompletely integrated.
- 6: Provides a meaningful central mechanism, but only some contextual features materially contribute. Other features are explained separately or used mainly as background or state markers.
- 4: Primarily presents a feature-by-feature account or isolated pairwise relationships. Removing most contextual features would leave the central hypothesis essentially unchanged.
- 2: Provides only a broad pathway, cellular state, generic causal suggestion, or isolated feature-target relationship.
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
- Scores of 8-10 require most contextual features to materially shape, constrain, or explain the central hypothesis.
- Experiments, citations, presentation quality, and answer length do not increase this score.

5. MECHANISTIC SYNTHESIS

Evaluate how effectively the response composes its biological premises into a connected, integrated, and sufficiently detailed explanation of the focus-feature-target association.

Before scoring, internally identify:
- The focus feature and target;
- The contextual features;
- Any intermediate processes or cellular states; and
- The relationships explicitly proposed among them.

Use Biological Plausibility only to identify material hallucinations. Do not transfer the Biological Plausibility score or its deductions into this criterion. A novel, indirect, unmeasured, or weakly supported relationship may contribute fully to synthesis when the response explains it as a biologically reasonable hypothesis. Exclude a relationship only when Biological Plausibility identifies it as clearly false, invented, or biologically indefensible.

A contextual feature counts as integrated when the response explains its biological role and how that role helps create, modify, constrain, or identify the condition in which the focus feature is associated with the target. A feature may contribute as a regulator, mediator, intermediate, permissive condition, state marker, proxy, parallel process, or downstream readout. It does not need to interact directly with the focus feature.

Mechanistic depth requires more than listing feature functions, pathways, or molecular terminology. Credit depth when the response explains the biological transitions connecting the features, intermediate processes, and target association.

- 10-8: Strong mechanistic synthesis. The response presents a cohesive, detailed, and sufficiently resolved mechanism connecting the focus feature to the target. Most relevant contextual features make explicit and meaningful contributions, and the important biological transitions or intermediate processes are explained. Use the higher end for a well-developed multi-step mechanism and the lower end when one important transition or contextual contribution remains broad or incomplete.

- 7-5: Partial mechanistic synthesis. A meaningful central mechanism is present, but its integration or resolution is incomplete. Some contextual features contribute clearly, while others remain loosely connected, isolated, or primarily descriptive.

- 4-2: Weak mechanistic synthesis. The response mainly provides feature-by-feature descriptions, pathway names, isolated pairwise relationships, or a broad cellular-state explanation.

- 1-0: Minimal or absent mechanistic synthesis. The response provides little or no substantive mechanism beyond restating the tree or listing biological functions.

Apply the following ceilings:
- No explicit mechanism connecting the focus feature to the target: maximum 3.
- Fewer than half of the relevant contextual features are explicitly integrated: maximum 5.
- Two or more central transitions are asserted without a relationship, intermediate process, or contextual bridge: maximum 6.
- The response primarily lists feature functions or pathways without explaining how they work together: maximum 5.
- Multiple mechanisms are proposed without explaining how they are compatible or related: maximum 5.
- The principal focus-feature-target mechanism depends on a material hallucination identified under Biological Plausibility: maximum 3.

Apply only the lowest relevant ceiling. Do not require agreement with the judge's preferred mechanism.

Also provide "Mechanistic Synthesis_reason": exactly three concise sentences stating (1) which relationships or contextual features were successfully integrated, (2) which remained isolated or insufficiently connected, and (3) which ceiling, if any, was applied and why.

6. LITERATURE GROUNDING

Evaluate the reliability of the established biological knowledge used to support the hypothesis, including feature functions, pathway relationships, cell-type associations, and claims attributed to publications.

Score the following components and add them.

A. Accuracy and relevance of supporting premises: 0-5

- 5: Central supporting facts are accurate, specific, relevant, and correctly applied.
- 4: The foundation is mostly accurate, with one peripheral imprecision or indirect extrapolation.
- 3: Useful knowledge is present, but one important premise is questionable or several premises are overly general.
- 2: Several important feature functions or pathway relationships are inaccurate, unsupported, or incorrectly applied.
- 1: The supporting premises are mostly generic, unreliable, or invented.
- 0: The biological support is predominantly fictional, irrelevant, or fundamentally incorrect.

B. Evidence fidelity and claim calibration: 0-5

- 5: Established knowledge, inference, and hypothesis are clearly distinguished. References support the claims attributed to them.
- 4: Evidence is generally well calibrated, with one peripheral mismatch, indirect citation, or mild overstatement.
- 3: The biological foundation remains recognizable, but several sources are indirect or one important speculative relationship is presented too strongly.
- 2: Multiple references or assertions are mismatched, likely inaccurate, or used to support claims they do not establish.
- 1: Citation misuse, likely fabricated references, or unsupported assertions presented as established facts are widespread.
- 0: The evidence base is materially fabricated or unrelated to the interpretation.

Literature Grounding score = A + B.

Do not award credit for citation quantity, journal prestige, detailed formatting, or authoritative wording. Evidence from a related gene family, cell type, or pathway is indirect unless presented as an analogy or extrapolation. An uncited but established biological fact may still receive credit.

OUTPUT FORMAT

Return the final grades in a markdown JSON block using exactly these keys. Scores are integers; the two "_reason" values are short strings:

~~~json
{
  "Feature Comprehensiveness": 0,
  "Logic Rigor": 0,
  "Biological Plausibility": 0,
  "Biological Plausibility_reason": "",
  "Specificity": 0,
  "Mechanistic Synthesis": 0,
  "Mechanistic Synthesis_reason": "",
  "Literature Grounding": 0
}
~~~
""".replace("~~~", "```")


def parse_prompt_file(content):
    """Same section split used by generate_hypotheses.py's parse_prompt_file."""
    match = re.search(r'--- SYSTEM PROMPT ---\s*\n(.*?)\n--- ', content, re.DOTALL)
    if match:
        system_prompt = match.group(1).strip()
        end_of_sys_prompt = content.find(system_prompt) + len(system_prompt)
        user_content = content[end_of_sys_prompt:].strip()
    else:
        system_prompt = "You are an expert computational biologist tasked with interpreting decision tree classification rules."
        user_content = content
    return system_prompt, user_content


def load_client(env_var):
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(
            f"ERROR: environment variable {env_var} is not set.\n"
            f'Set it with: [System.Environment]::SetEnvironmentVariable("{env_var}", "<your key>", "User")\n'
            "then restart the terminal (or pass --api-key-env NAME for a differently named variable)."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=300.0)


def parse_judge_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = match.group(1) if match else text.strip().strip("`")
    return json.loads(raw)


def judge_one(client, exam_paper, student_answer):
    user_content = (
        f"--- INPUT 1: WHAT THE MODEL WAS GIVEN ---\n{exam_paper}\n\n"
        f"--- INPUT 2: THE MODEL'S SUBMISSION ---\n{student_answer}\n"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return parse_judge_json(response.choices[0].message.content or "")


def main(overwrite, api_key_env):
    client = load_client(api_key_env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, failed = [], []
    for ds in config.DATASETS:
        for setting in config.DT_SETTINGS:
            out_dir = config.RESULTS_DIR / ds["key"]
            prompt_path = out_dir / f"{ds['key']}_{setting['name']}_primaryonly_prompt.txt"
            hyp_path = out_dir / f"{ds['key']}_{setting['name']}_primaryonly_hypothesis.jsonl"
            scorecard_path = OUT_DIR / f"{ds['key']}_{setting['name']}_scorecard.json"

            if not prompt_path.exists() or not hyp_path.exists():
                print(f"SKIP {ds['key']}/{setting['name']}: missing prompt or hypothesis file")
                failed.append((ds["key"], setting["name"]))
                continue
            if scorecard_path.exists() and not overwrite:
                print(f"SKIP (cached): {ds['key']}/{setting['name']}")
                scores = json.loads(scorecard_path.read_text(encoding="utf-8"))["scores"]
                all_rows.append({"dataset": ds["key"], "setting": setting["name"], **scores})
                continue

            system_prompt, user_prompt = parse_prompt_file(prompt_path.read_text(encoding="utf-8"))
            exam_paper = f"{system_prompt}\n\n{user_prompt}"
            with open(hyp_path, "r", encoding="utf-8") as f:
                answer = json.loads(f.readline())["final_answer"]

            print(f"Judging {ds['key']} / {setting['name']} ...", end=" ", flush=True)
            try:
                scores = judge_one(client, exam_paper, answer)
            except Exception as e:  # surface the real error; the run summary lists what's missing
                print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                failed.append((ds["key"], setting["name"]))
                continue
            print("  ".join(f"{p.split()[0][:4]}={scores.get(p)}" for p in PILLARS))

            scorecard_path.write_text(json.dumps(
                {"dataset": ds["key"], "setting": setting["name"], "judge_model": JUDGE_MODEL, "scores": scores},
                indent=2), encoding="utf-8")
            all_rows.append({"dataset": ds["key"], "setting": setting["name"], **scores})

    AGGREGATE_JSON.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nDone. {len(all_rows)} scorecard(s), {len(failed)} missing/failed: {failed or 'none'}")
    print(f"Aggregate: {AGGREGATE_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Redo (dataset, setting) pairs that already have a scorecard.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Name of the env var holding the OpenAI key.")
    args = parser.parse_args()
    main(args.overwrite, args.api_key_env)
