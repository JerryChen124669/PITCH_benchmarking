"""Builds the two ablation prompts per dataset. Both share the same overall
biological question and JSON-footer schema as the full-pipeline studies
(DTEvaluation, ClassifierEvaluation) so hypotheses stay comparable, but each
strips away one more computational stage:

  Condition 1 (ig_only):  keeps IG ranking, removes the classifier entirely.
  Condition 2 (raw_data): removes IG too -- raw per-cell numbers only.

Both prompts explicitly instruct the model to be honest about uncertainty
and to self-report a numeric confidence score, per the study's design for
telling a genuinely-uninformed answer apart from a confidently-worded but
unsupported one (see disqualify.py for how that gets checked afterward).
"""
import config

HONESTY_INSTRUCTION = (
    "HONESTY INSTRUCTION -- READ CAREFULLY: If the evidence provided below is not sufficient to form a "
    "specific, well-grounded biological hypothesis, you MUST say so explicitly, explain what information "
    "would be needed, and report a LOW confidence score. Do NOT fabricate specific mechanisms, statistics, "
    "node identifiers, p-values, or literature citations to make your answer sound more certain or complete "
    "than it actually is. It is significantly better to honestly say 'I cannot determine this with confidence "
    "from the given evidence' than to produce a fluent but unsupported answer. Your confidence score will be "
    "checked against how well-grounded your stated evidence actually is -- a mismatch between a high "
    "confidence score and a vague or fabricated answer is the single worst outcome for this evaluation."
)

JSON_FOOTER_SCHEMA = (
    '```json\n'
    '{\n'
    '  "confidence": <integer 1-10, your genuine confidence in this specific hypothesis>,\n'
    '  "sufficient_information": <true or false -- was the evidence you were given actually enough to form '
    'a specific, testable hypothesis, as opposed to a generic guess?>,\n'
    '  "top_4_contexts": ["<feature>", "<feature>", "<feature>", "<feature>"]\n'
    '}\n'
    '```\n'
    "Feature names in top_4_contexts must be copied verbatim from the evidence given to you above "
    "(the exact column header / feature name), not the focused feature itself."
)


def _shared_task_block(target_prot, target_name):
    return (
        f"--- TASK ---\n"
        f"Form a rationale and hypothesis explaining HOW features relate to {target_prot} and together "
        f"associate with {target_name}. State your single best hypothesis and the 4 features/values that "
        f"most drove your reasoning. Identify whether your conclusion (a) directly confirms established "
        f"literature knowledge, or (b) suggests a new testable mechanism -- and cite supporting publications "
        f"if you can. If you genuinely cannot form a specific hypothesis from the evidence given, say so per "
        f"the HONESTY INSTRUCTION above instead of inventing one.\n\n"
        f"FINAL OUTPUT REQUIREMENT: after the narrative, append exactly one fenced json code block with this "
        f"structure:\n{JSON_FOOTER_SCHEMA}"
    )


RAW_DATA_JSON_FOOTER_SCHEMA = (
    '```json\n'
    '{\n'
    '  "confidence": <integer 1-10, your genuine confidence in this specific hypothesis>,\n'
    '  "sufficient_information": <true or false -- was the evidence you were given actually enough to form '
    'a specific, testable hypothesis, as opposed to a generic guess?>,\n'
    '  "top_4_contexts": ["<feature>", "<feature>", "<feature>", "<feature>"]\n'
    '}\n'
    '```\n'
    "Feature names in top_4_contexts must be copied verbatim from the column headers given to you above, "
    "not the target response column itself."
)


def _raw_data_task_block(target_name):
    """Unlike _shared_task_block, this does NOT name a target_prot -- cond2
    is supposed to test whether raw data alone can find which feature(s)
    matter, not just explain a feature the pipeline already told it about.
    Also uses its own JSON footer wording (RAW_DATA_JSON_FOOTER_SCHEMA)
    instead of JSON_FOOTER_SCHEMA, since the latter's closing line ("not the
    focused feature itself") presupposes a pre-named focused feature that
    this condition deliberately withholds."""
    return (
        f"--- TASK ---\n"
        f"Identify which feature(s) among those shown, if any, appear most associated with {target_name}, "
        f"and form a rationale and hypothesis for the underlying biological mechanism. State your single "
        f"best hypothesis and the 4 features/values that most drove your reasoning -- these must be features "
        f"YOU identify from inspecting the table yourself, not features named anywhere else in this prompt "
        f"(no feature has been flagged as the one of interest). Identify whether your conclusion (a) directly "
        f"confirms established literature knowledge, or (b) suggests a new testable mechanism -- and cite "
        f"supporting publications if you can. If you genuinely cannot form a specific hypothesis from the "
        f"evidence given, say so per the HONESTY INSTRUCTION above instead of inventing one.\n\n"
        f"FINAL OUTPUT REQUIREMENT: after the narrative, append exactly one fenced json code block with this "
        f"structure:\n{RAW_DATA_JSON_FOOTER_SCHEMA}"
    )


def build_ig_only_prompt(*, dataset_meta, target_name, target_prot, corr_label,
                          n_features_requested, ig_ranking, cv_note=None):
    """Condition 1: IG ranking only, no classifier of any kind."""
    ranking_lines = "\n".join(
        f"  {i}. {feat}  (mean |IG attribution|: {val:.4f})"
        for i, (feat, val) in enumerate(ig_ranking, start=1)
    )
    text = (
        f"--- SYSTEM PROMPT ---\n"
        f"Data Domain Context: {dataset_meta['domain']}\n"
        f"Cell Type / Study Tissue: {dataset_meta['cell_type']}\n"
        f"Ablation Condition: IG-ONLY (Integrated Gradients ranking given; NO Decision Tree / Random Forest / "
        f"Logistic Regression classifier was fit for this condition -- this tests whether the classifier "
        f"stage of the pipeline adds anything beyond the raw importance ranking alone).\n\n"

        f"You are an expert computational biologist. Your objective is to translate a ranked list of feature "
        f"importances into a cohesive biological narrative and a concrete, testable hypothesis. Unlike a full "
        f"analysis, you are NOT given any conditional logic, decision rules, or per-subgroup statistics -- "
        f"only which features matter most overall, on average across all cells.\n\n"
        f"{HONESTY_INSTRUCTION}\n\n"

        f"--- EXPERIMENT CONTEXT & DATA PIPELINE ---\n"
        f"1. Goal: explain why {target_prot} attributes to the prediction of {target_name}. Focused feature: "
        f"'{target_prot}'. Target response: '{target_name}'.\n"
        f"2. Upstream Model: a PyTorch nonlinear Multi-layer Perceptron was trained on the raw dataset "
        f"(for the pERK/EGF dataset, an upstream multi-concentration model), and Integrated Gradients (IG) "
        f"attribution scores were computed from it.\n"
        f"3. Feature Selection: NO classifier was fit. The top {n_features_requested} features by mean |IG| "
        f"attribution are listed below, in ranked order -- this is the entire evidence you are given.\n"
        f"4. Target response and Focus feature Correlation: {corr_label} correlation between {target_prot} "
        f"and {target_name}.\n\n"

        f"--- IG RANKING (top {n_features_requested} features by mean |IG| attribution, no classifier applied) ---\n"
        f"{ranking_lines}\n\n"

        f"{_shared_task_block(target_prot, target_name)}"
    )
    return text


def build_raw_data_prompt(*, dataset_meta, target_name, target_prot, n_total_cells, n_sample_cells,
                           sample_tokens, csv_table_text):
    """Condition 2: raw pre-MLP data only, no IG, no classifier, no derived
    statistics of any kind (not even a correlation coefficient).

    target_prot is accepted for bookkeeping/scoring only (so disqualify.py
    and downstream analysis can check whether the model's own answer landed
    on it) -- it is NEVER placed in the prompt text itself. This condition's
    own stated purpose is to test whether raw data can replace the ENTIRE
    pipeline, including which feature is worth investigating in the first
    place; naming target_prot up front would just hand over that answer."""
    text = (
        f"--- SYSTEM PROMPT ---\n"
        f"Data Domain Context: {dataset_meta['domain']}\n"
        f"Cell Type / Study Tissue: {dataset_meta['cell_type']}\n"
        f"Ablation Condition: RAW DATA ONLY (no Integrated Gradients, no classifier of any kind, no "
        f"precomputed statistics -- this tests whether your own prior biological knowledge plus direct "
        f"inspection of raw numbers can replace the entire computational pipeline, INCLUDING deciding which "
        f"feature is even worth investigating).\n\n"

        f"You are an expert computational biologist. You are given a raw, unprocessed sample of single-cell "
        f"measurements as a CSV table below: each row is one cell, each column (except the last) is one "
        f"measured feature, and the last column is the target response. NOTHING has been computed for you -- "
        f"no feature ranking, no correlation, no classifier, no binary labeling, and no feature has been "
        f"pre-selected as important. You must inspect the raw numbers yourself (to whatever extent you "
        f"actually can) and determine what, if anything, appears associated with the target.\n\n"
        f"{HONESTY_INSTRUCTION} In particular: performing genuine multivariate statistical inference over a "
        f"CSV table by reading it is a real limitation of this format -- if you are pattern-matching from "
        f"general biological knowledge about these feature/gene names rather than actually deriving a "
        f"relationship from the numbers in front of you, SAY SO explicitly, and reflect that in your "
        f"confidence and sufficient_information fields.\n\n"

        f"--- EXPERIMENT CONTEXT & DATA PIPELINE ---\n"
        f"1. Goal: identify what, if anything, regulates {target_name} abundance/expression, based only on "
        f"the raw feature table below. No feature has been flagged as the one of interest -- you must "
        f"determine that yourself from the data.\n"
        f"2. NO upstream model, IG attribution, or classifier was used to prepare this evidence.\n"
        f"3. Sampling: the full dataset has {n_total_cells} cells. Because Gemini's input context cannot fit "
        f"the entire raw table for a dataset this wide, {n_sample_cells} cells ({n_sample_cells / n_total_cells * 100:.2f}% "
        f"of the full dataset, {sample_tokens:,} tokens) were sampled below, stratified across deciles of the "
        f"target value so the full low-to-high range is represented -- NOT a random or convenience sample, "
        f"but still only a small fraction of the full dataset. Treat any conclusion accordingly.\n\n"

        f"--- RAW DATA (CSV, {n_sample_cells} cells x {csv_table_text.split(chr(10))[0].count(',') + 1} columns; last column = {target_name}) ---\n"
        f"{csv_table_text}\n"

        f"{_raw_data_task_block(target_name)}"
    )
    return text
