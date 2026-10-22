"""Builds the LLM interpretation prompt text, extending the template used in
app_v4.py's Step 3 "Export LLM Interpretation Prompt" section with:
  - an added system-prompt line asking the LLM to identify the single best
    node and the 4 most important contexts to anchor hypothesis generation;
  - a richer per-node evidence table (7-point spec: conditions, depth,
    support, Gini impurity, Gini reduction, target-cell enrichment, IG
    enrichment) instead of just TP/FP/precision/Gini on positive leaves;
  - low-support nodes flagged/excluded from primary consideration rather
    than silently dropped, since a pure-but-tiny node likely reflects
    overfitting rather than a reproducible biological context.
"""
from sklearn.tree import export_text

import config


def _format_feature_confidence(rec):
    lines = [
        f"      {i}. {step['condition']}  (local Gini-reduction contribution: {step['local_importance']:.4f})"
        for i, step in enumerate(rec["feature_confidence"], start=1)
    ]
    return "\n".join(lines)


def _format_node_block(rec, target_prot):
    flag = "  *** LOW SUPPORT -- EXCLUDED FROM PRIMARY CONSIDERATION (possible overfitting) ***\n" if rec["flag_low_support"] else ""
    note = "  (path has >4 conditions -- use the ranking below to pick the top ~4)\n" if rec["depth"] > 4 else ""
    return (
        f"Node {rec['node_id']} (depth {rec['depth']})\n"
        f"{flag}"
        f"  (1) Split conditions: {rec['rule']}\n"
        f"{note}"
        f"      Feature confidence ranking within this rule (highest local Gini-reduction contribution first):\n"
        f"{_format_feature_confidence(rec)}\n"
        f"  (2) Node depth: {rec['depth']}\n"
        f"  (3) Support: {rec['n_cells']} cells ({rec['pop_fraction'] * 100:.2f}% of total population)\n"
        f"  (4) Node Gini impurity: {rec['gini_impurity']:.4f}\n"
        f"  (5) Gini reduction from originating split: {rec['gini_reduction']:.4f}\n"
        f"  (6) Target-cell fraction: {rec['target_fraction'] * 100:.1f}% "
        f"(population rate {rec['overall_target_rate'] * 100:.1f}%, enrichment/lift = {rec['target_lift']:.2f}x)\n"
        f"  (7) Mean IG score of '{target_prot}' in node: {rec['mean_ig']:.4f} "
        f"(population mean {rec['overall_mean_ig']:.4f}, SD {rec['overall_std_ig']:.4f}; "
        f"enrichment = {rec['ig_enrichment_sd']:+.2f} population-SD units above/below the population mean)\n"
    )


def build_prompt(*, dataset_meta, target_name, target_prot, corr_label, attr_direction,
                  ig_percentile, n_features_requested, attr_features, max_depth,
                  class_1_weight, clf, cv_f1, node_records, min_support,
                  include_secondary=True, tree_rules_override=None):
    """include_secondary=False builds the "primary-zone-only" variant used to
    test whether dropping the secondary-zone ask changes cross-setting
    robustness -- see DTEvaluation/regenerate_primary_only.py.
    tree_rules_override lets that script reuse the tree-rules text already
    saved in the original prompt .txt instead of needing the fitted `clf`
    (which isn't persisted to disk), so no retraining is required."""

    primary = sorted(
        [r for r in node_records if not r["flag_low_support"]],
        key=lambda r: (-r["target_lift"], -r["pop_fraction"]),
    )
    flagged = [r for r in node_records if r["flag_low_support"]]

    primary_block = "\n".join(_format_node_block(r, target_prot) for r in primary) if primary else \
        "No leaf node met the minimum support threshold; all nodes are flagged below."
    flagged_block = "\n".join(
        f"Node {r['node_id']} (depth {r['depth']}): {r['rule']} "
        f"[{r['n_cells']} cells, target lift {r['target_lift']:.2f}x] -- below min support ({min_support} cells)"
        for r in flagged
    ) if flagged else "None."

    tree_rules = tree_rules_override if tree_rules_override is not None else export_text(clf, feature_names=attr_features)

    additional_instruction = (
        f"ADDITIONAL INSTRUCTION: Explicitly identify (a) the single BEST node overall for hypothesis generation, "
        f"stating why it beats the alternatives on the criteria above, and (b) the 4 most important contexts "
        f"(i.e. the most decision-relevant split conditions/features) that should anchor your hypothesis. When a "
        f"node's rule has more than 4 conditions (e.g. a depth-6 path), use that node's own 'Feature confidence "
        f"ranking' (its per-condition local Gini-reduction contribution) to select the top ~4 conditions -- do not "
        f"just take the first 4 in path order. State both (a) and (b) explicitly before building the narrative.\n\n"
    ) if not include_secondary else (
        f"ADDITIONAL INSTRUCTION: Explicitly identify (a) the single BEST node overall for hypothesis generation, "
        f"stating why it beats the alternatives on the criteria above, and (b) the 4 most important contexts "
        f"(i.e. the most decision-relevant split conditions/features) that should anchor your hypothesis. When a "
        f"node's rule has more than 4 conditions (e.g. a depth-6 path), use that node's own 'Feature confidence "
        f"ranking' (its per-condition local Gini-reduction contribution) to select the top ~4 conditions -- do not "
        f"just take the first 4 in path order. State both (a) and (b) explicitly before building the narrative. If multiple "
        f"biologically distinct zones are plausible, select the most plausible as the PRIMARY zone and a "
        f"highly plausible but biologically distinct zone as the SECONDARY zone.\n\n"
    )

    task_section = (
        f"--- TASK ---\n"
        f"Based on the CANDIDATE NODES above, first explicitly state (a) the single BEST node for hypothesis "
        f"generation and why, and (b) the 4 most important contexts (features/conditions) driving that "
        f"conclusion. Then form rationale and hypotheses of the feature relationships (pathways) to explain HOW "
        f"features and {target_prot} together associate with {target_name}. Highlight which zones are the most "
        f"biologically robust based on support, enrichment, and impurity reduction (not Gini alone). Identify "
        f"findings into two categories: 1. findings that directly confirm literature knowledge; 2. findings "
        f"suggesting new testable mechanisms supported by literature. List the publications supporting the "
        f"conclusions."
    ) if not include_secondary else (
        f"--- TASK ---\n"
        f"Based on the CANDIDATE NODES above, first explicitly state (a) the single BEST node for hypothesis "
        f"generation and why, and (b) the 4 most important contexts (features/conditions) driving that "
        f"conclusion. Then form rationale and hypotheses of the feature relationships (pathways) to explain HOW "
        f"features and {target_prot} together associate with {target_name}. Highlight which zones are the most "
        f"biologically robust based on support, enrichment, and impurity reduction (not Gini alone). Identify "
        f"findings into two categories: 1. findings that directly confirm literature knowledge; 2. findings "
        f"suggesting new testable mechanisms supported by literature. List the publications supporting the "
        f"conclusions. Next, look into the second-best plausible zone and repeat the analysis (present the "
        f"conclusions for the two highlighted zones separately)."
    )

    footer_secondary_line = '  "secondary_zone_node_id": <integer node id, or null if none>,\n' if include_secondary else ""

    text = (
        f"--- SYSTEM PROMPT ---\n"
        f"Data Domain Context: {dataset_meta['domain']}\n"
        f"Cell Type / Study Tissue: {dataset_meta['cell_type']}\n\n"

        f"You are an expert computational biologist tasked with interpreting decision tree classification rules "
        f"that predict specific biomarkers based on RNA gene expression and gene attributes. Your objective is "
        f"to translate the mathematical logic of the tree, including feature hierarchy, split thresholds, and "
        f"conditional dependencies, into a cohesive biological narrative that explains the underlying regulatory "
        f"mechanisms. Using the specific biological context provided, you must synthesize these findings into "
        f"concrete, experimentally testable hypotheses.\n"
        f"CRITICAL INSTRUCTION: Primarily analyze the CANDIDATE NODES section below. Prioritize nodes that combine "
        f"substantial population support, strong IG enrichment, high target-cell enrichment, meaningful Gini "
        f"impurity reduction, and relatively low tree depth -- do NOT select a node solely because it has the "
        f"lowest Gini impurity or highest precision if it is only supported by a handful of cells. Nodes flagged "
        f"as low support are listed separately and should be treated as overfitting risks, not primary evidence. "
        f"Use the full raw decision tree only as supplementary context; do not attempt to parse every individual "
        f"branch.\n"
        f"{additional_instruction}"

        f"--- EXPERIMENT CONTEXT & DATA PIPELINE ---\n"
        f"1. Goal: Forming biological rationale and hypothesis based on the decision tree results to explain why "
        f"{target_prot} heavily attributes to the prediction of {target_name}. Focused feature: '{target_prot}'. "
        f"Target response: '{target_name}'.\n"
        f"2. Upstream Model: A PyTorch nonlinear Multi-layer Perceptron model was trained on the raw dataset.\n"
        f"3. Feature Selection: Integrated Gradients (IG) was applied. The top {n_features_requested} features "
        f"with the highest Mean Absolute IG Attribution were selected as candidate attributing features "
        f"(the focused feature itself is excluded from this attributing set), yielding {len(attr_features)} "
        f"attributing features actually used by the tree.\n"
        f"4. Target response and Focus feature Correlation: {corr_label} correlation between {target_prot} and {target_name}.\n"
        f"5. Label Generation: Samples were labeled as Class 1 ('High focused Attribution') if their IG score for "
        f"{target_prot} fell in the extreme {100 - ig_percentile}% according to the analysis type. All other "
        f"samples are Class 0. Analysis type: the {attr_direction} Attribution Analysis focusing on conditions of "
        f"{target_prot}'s {attr_direction} attribution effect.\n"
        f"6. Tree Parameters and classification: max_depth={max_depth}, min_samples_leaf="
        f"{config.DT_FIXED_PARAMS['min_samples_leaf']}, no max_leaf_nodes cap (leaf count is bounded only by "
        f"max_depth and min_samples_leaf, so depth is free to add structure rather than being capped at a shared "
        f"leaf budget), class_weight={{0: 1, 1: {class_1_weight}}}. 5-fold CV F1 (macro): {cv_f1:.1%}. The decision tree aims to "
        f"find the conditions that contain most Class 1 data points and minimal Class 0 data points.\n"
        f"7. Node exclusion rule: leaf nodes with fewer than {min_support} cells "
        f"(max({config.MIN_SUPPORT_ABS} cells, {config.MIN_SUPPORT_FRACTION * 100:.1f}% of the dataset)) are "
        f"flagged and excluded from primary consideration, since a highly pure node with very few cells is more "
        f"likely to reflect overfitting than a reproducible biological context.\n\n"

        f"--- CANDIDATE NODES: QUANTITATIVE EVIDENCE (all leaf nodes, sorted by target-cell lift) ---\n"
        f"{primary_block}\n\n"

        f"--- LOW-SUPPORT / FLAGGED NODES (for reference only, not primary evidence) ---\n"
        f"{flagged_block}\n\n"

        f"--- FULL DECISION TREE RAW RULES ---\n"
        f"{tree_rules}\n"

        f"{task_section}\n\n"
        f"FINAL OUTPUT REQUIREMENT: After the narrative, append exactly one fenced json code block with this "
        f"structure, so your selections can be checked against the quantitative evidence above:\n"
        f"```json\n"
        f"{{\n"
        f'  "primary_zone_node_id": <integer node id from the CANDIDATE NODES section>,\n'
        f"{footer_secondary_line}"
        f'  "top_4_contexts": ["<feature name>", "<feature name>", "<feature name>", "<feature name>"]\n'
        f"}}\n"
        f"```\n"
        f"Feature names in top_4_contexts must be copied verbatim from the split conditions of the primary zone "
        f"(e.g. the feature name in \"{attr_features[0] if attr_features else 'FEATURE'} <= 1.23\" is "
        f"\"{attr_features[0] if attr_features else 'FEATURE'}\"), not the focused feature '{target_prot}' itself."
    )
    return text
