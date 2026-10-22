"""Builds the LLM interpretation prompt text for one (dataset, model_type)
run. PRIMARY ZONE ONLY -- secondary-zone analysis was dropped from the study
(the "secondary hypothesis" had no grounded candidate pool for LogReg/RF
anyway; see below).

Model-type-aware "MODEL-SPECIFIC EVIDENCE" section:
  - Decision Tree: leaf-node quantitative evidence table -- the ONLY model
    type with a real, verifiable pool of context-dependent candidates (each
    leaf node is an actual conditional subpopulation with its own support/
    lift/Gini stats).
  - Random Forest: forest-wide feature_importances_ ranking ONLY. An earlier
    version of this prompt also included an "illustrative representative
    tree" (a second DecisionTreeClassifier, fit fresh with DT's exact
    recipe on the same data) to give RF something zone-shaped to reason
    with -- but that tree is not part of the actual 200-tree ensemble and
    does not reflect how the forest really decides (200 different trees
    voting, no single unified rule). It was removed: RF's authoritative
    evidence (feature_importances_) is a pure aggregate, exactly as
    "zone-less" as Logistic Regression's coefficients, and giving it a
    borrowed tree was manufacturing conditional structure the ensemble
    doesn't actually expose in readable form.
  - Logistic Regression: standardized-coefficient ranking (direction +
    magnitude of each feature's effect on the log-odds of Class 1). No
    zones -- global linear model, structurally incapable of representing
    conditional/context-dependent mechanisms.

The required JSON footer is kept schema-compatible: "primary_zone_node_id"
is populated only for DT (null for RF/LogReg, which have no nodes);
"top_4_contexts" is always populated (top-4 features driving the
hypothesis, by whatever ranking that model type produces).
"""
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


def _model_section_dt(evidence, target_prot):
    primary = sorted(
        [r for r in evidence["node_records"] if not r["flag_low_support"]],
        key=lambda r: (-r["target_lift"], -r["pop_fraction"]),
    )
    flagged = [r for r in evidence["node_records"] if r["flag_low_support"]]

    primary_block = "\n".join(_format_node_block(r, target_prot) for r in primary) if primary else \
        "No leaf node met the minimum support threshold; all nodes are flagged below."
    flagged_block = "\n".join(
        f"Node {r['node_id']} (depth {r['depth']}): {r['rule']} "
        f"[{r['n_cells']} cells, target lift {r['target_lift']:.2f}x] -- below min support ({evidence['min_support']} cells)"
        for r in flagged
    ) if flagged else "None."

    from sklearn.tree import export_text
    tree_rules = export_text(evidence["clf"], feature_names=evidence["attr_features"])
    return (
        f"--- CANDIDATE NODES: QUANTITATIVE EVIDENCE (all leaf nodes, sorted by target-cell lift) ---\n"
        f"{primary_block}\n\n"
        f"--- LOW-SUPPORT / FLAGGED NODES (for reference only, not primary evidence) ---\n"
        f"{flagged_block}\n\n"
        f"--- FULL DECISION TREE RAW RULES ---\n"
        f"{tree_rules}\n"
    )


def _model_section_rf(evidence, target_prot):
    forest_lines = "\n".join(
        f"  {i}. {feat}  (mean decrease in impurity, averaged over {config.RF_N_ESTIMATORS} trees: {imp:.4f})"
        for i, (feat, imp) in enumerate(evidence["forest_ranking"], start=1)
    )
    return (
        f"--- FOREST-WIDE FEATURE IMPORTANCE (AUTHORITATIVE -- averaged across all {config.RF_N_ESTIMATORS} trees) ---\n"
        f"{forest_lines}\n\n"
        f"NOTE: This is the ENTIRE evidence for this model. A Random Forest is {config.RF_N_ESTIMATORS} "
        f"independently-trained trees voting together; there is no single unified 'if condition A and B then Y' "
        f"rule for the ensemble the way there is for one tree. Do not invent conditional/branching logic that "
        f"was not given to you -- this ranking is a global, aggregate measure (mean impurity decrease per "
        f"feature, averaged over all trees), structurally similar to Logistic Regression's coefficient ranking, "
        f"not to a Decision Tree's leaf-node zones.\n"
    )


def _model_section_logreg(evidence, target_prot):
    coef_lines = "\n".join(
        f"  {i}. {feat}  (standardized coefficient: {coef:+.4f}, "
        f"{'increases' if coef > 0 else 'decreases'} log-odds of high '{target_prot}' attribution per +1 SD)"
        for i, (feat, coef) in enumerate(evidence["coef_ranking"], start=1)
    )
    return (
        f"--- STANDARDIZED LOGISTIC REGRESSION COEFFICIENTS (AUTHORITATIVE, ranked by |coefficient|) ---\n"
        f"All attributing features were z-scored (mean 0, SD 1) before fitting, so coefficients are directly "
        f"comparable in magnitude across features of different raw units. A positive coefficient means higher "
        f"values of that feature push the sample toward Class 1 ('High {target_prot} Attribution'); a negative "
        f"coefficient means the opposite.\n\n"
        f"{coef_lines}\n\n"
        f"NOTE: Logistic Regression is a GLOBAL LINEAR model -- there are no rule-based 'zones' or leaf nodes. "
        f"Its hypothesis should be framed in terms of a monotonic, additive relationship between the top-weighted "
        f"features and {target_prot}'s attribution, not conditional/branching logic.\n"
    )


def build_prompt(*, model_key, model_label, dataset_meta, target_name, target_prot, corr_label,
                  attr_direction, ig_percentile, n_features_requested, attr_features,
                  class_1_weight, cv_f1, evidence):

    has_zones = (model_key == "dt")  # ONLY Decision Tree has a real, verifiable candidate pool

    if model_key == "dt":
        model_section = _model_section_dt(evidence, target_prot)
        classifier_desc = (
            f"A single Decision Tree Classifier (max_depth={config.DT_MAX_DEPTH}, "
            f"min_samples_leaf={config.DT_FIXED_PARAMS['min_samples_leaf']}, no max_leaf_nodes cap, "
            f"class_weight={{0: 1, 1: {class_1_weight}}})."
        )
        critical_instruction = (
            "CRITICAL INSTRUCTION: Primarily analyze the CANDIDATE NODES section below. Prioritize nodes that "
            "combine substantial population support, strong IG enrichment, high target-cell enrichment, "
            "meaningful Gini impurity reduction, and relatively low tree depth -- do NOT select a node solely "
            "because it has the lowest Gini impurity or highest precision if it is only supported by a handful "
            "of cells. Nodes flagged as low support are listed separately and should be treated as overfitting "
            "risks, not primary evidence."
        )
        anchor_instruction = (
            "ADDITIONAL INSTRUCTION: Explicitly identify (a) the single BEST node for hypothesis generation, "
            "stating why it beats the alternatives on the criteria above, and (b) the 4 most important contexts "
            "(i.e. the most decision-relevant split conditions/features) that should anchor your hypothesis. "
            "When a node's rule has more than 4 conditions, use that node's own 'Feature confidence ranking' to "
            "select the top ~4 conditions -- do not just take the first 4 in path order."
        )
    elif model_key == "rf":
        model_section = _model_section_rf(evidence, target_prot)
        classifier_desc = (
            f"A Random Forest Classifier ({config.RF_N_ESTIMATORS} trees, max_depth={config.RF_MAX_DEPTH} per "
            f"tree, min_samples_leaf={config.RF_FIXED_PARAMS['min_samples_leaf']}, "
            f"class_weight={{0: 1, 1: {class_1_weight}}})."
        )
        critical_instruction = (
            "CRITICAL INSTRUCTION: Anchor your hypothesis on the FOREST-WIDE FEATURE IMPORTANCE ranking below -- "
            "this is the entire evidence available for this model. There are no tree nodes/zones for this "
            "model -- describe the relationship as a global, additive association between the top-ranked "
            "features and the outcome, not as conditional/branching logic. Do not describe or imply specific "
            "split thresholds, since none were given to you."
        )
        anchor_instruction = (
            "ADDITIONAL INSTRUCTION: Explicitly identify the 4 most important contexts (the top-4 features by "
            "forest-wide importance) that should anchor your hypothesis."
        )
    else:  # logreg
        model_section = _model_section_logreg(evidence, target_prot)
        classifier_desc = (
            f"A Logistic Regression classifier fit on z-scored features (class_weight="
            f"{{0: 1, 1: {class_1_weight}}}, L2-regularized, max_iter={config.LOGREG_FIXED_PARAMS['max_iter']})."
        )
        critical_instruction = (
            "CRITICAL INSTRUCTION: Primarily anchor your hypothesis on the STANDARDIZED LOGISTIC REGRESSION "
            "COEFFICIENTS section below, in ranked order of |coefficient|. There are no tree nodes/zones for "
            "this model -- describe the relationship as a global, additive, monotonic association, not as "
            "conditional branching logic."
        )
        anchor_instruction = (
            "ADDITIONAL INSTRUCTION: Explicitly identify the 4 most important contexts (the top-4 features by "
            "|standardized coefficient|) that should anchor your hypothesis, and state the direction (positive "
            "or negative) of each one's association with high attribution of the focused feature."
        )

    text = (
        f"--- SYSTEM PROMPT ---\n"
        f"Data Domain Context: {dataset_meta['domain']}\n"
        f"Cell Type / Study Tissue: {dataset_meta['cell_type']}\n"
        f"Model Type Under Evaluation: {model_label}\n\n"

        f"You are an expert computational biologist tasked with interpreting a {model_label} classifier's "
        f"predictions of specific biomarkers based on cellular feature/gene expression data. Your objective is "
        f"to translate the model's logic, {'including feature hierarchy, split thresholds, and conditional dependencies, ' if has_zones else 'including the direction and relative magnitude of each feature effect, '}"
        f"into a cohesive biological narrative that explains the underlying regulatory mechanisms. Using the "
        f"specific biological context provided, you must synthesize these findings into concrete, "
        f"experimentally testable hypotheses.\n"
        f"{critical_instruction}\n"
        f"{anchor_instruction} State this explicitly before building the narrative.\n\n"

        f"--- EXPERIMENT CONTEXT & DATA PIPELINE ---\n"
        f"1. Goal: Forming biological rationale and hypothesis based on the {model_label} results to explain why "
        f"{target_prot} heavily attributes to the prediction of {target_name}. Focused feature: '{target_prot}'. "
        f"Target response: '{target_name}'.\n"
        f"2. Upstream Model: A PyTorch nonlinear Multi-layer Perceptron model was trained on the raw dataset "
        f"(for the pERK/EGF dataset, this MLP was trained upstream on the multi-concentration signaling panel; "
        f"for all other datasets, it was trained directly on this dataset's raw features).\n"
        f"3. Feature Selection: Integrated Gradients (IG) attribution scores were computed from that upstream "
        f"model. {n_features_requested} attributing candidate features "
        f"({'the top-ranked by Mean Absolute IG Attribution' if dataset_meta.get('mode') != 'manual' else 'a manually curated feature panel, not IG-ranked'}) "
        f"were selected (the focused feature itself is excluded from this attributing set), yielding "
        f"{len(attr_features)} attributing features actually used by the model.\n"
        f"4. Target response and Focus feature Correlation: {corr_label} correlation between {target_prot} and {target_name}.\n"
        f"5. Label Generation: Samples were labeled as Class 1 ('High focused Attribution') if their IG score for "
        f"{target_prot} fell in the extreme {100 - ig_percentile}% according to the analysis type. All other "
        f"samples are Class 0. Analysis type: the {attr_direction} Attribution Analysis focusing on conditions of "
        f"{target_prot}'s {attr_direction} attribution effect.\n"
        f"6. Classifier: {classifier_desc} 5-fold CV F1 (macro): {cv_f1:.1%}. The classifier aims to find the "
        f"conditions/features that best separate Class 1 from Class 0 data points.\n\n"

        f"{model_section}\n"

        f"--- TASK ---\n"
        f"Based on the evidence above, first explicitly state (a) the single BEST {'node' if has_zones else 'set of features'} "
        f"for hypothesis generation and why, and (b) the 4 most important contexts (features/conditions) driving "
        f"that conclusion. Then form rationale and a hypothesis of the feature relationships (pathways) to "
        f"explain HOW features and {target_prot} together associate with {target_name}. Base this on the "
        f"evidence above (not any single metric alone). Identify findings into two categories: 1. findings "
        f"that directly confirm literature knowledge; 2. findings suggesting new testable mechanisms supported "
        f"by literature. List the publications supporting the conclusions.\n\n"

        f"FINAL OUTPUT REQUIREMENT: After the narrative, append exactly one fenced json code block with this "
        f"structure, so your selections can be checked against the quantitative evidence above:\n"
        f"```json\n"
        f"{{\n"
        f'  "primary_zone_node_id": <integer node id from the evidence section above, or null if this model '
        f'type has no nodes (Random Forest and Logistic Regression both have none)>,\n'
        f'  "top_4_contexts": ["<feature name>", "<feature name>", "<feature name>", "<feature name>"]\n'
        f"}}\n"
        f"```\n"
        f"Feature names in top_4_contexts must be copied verbatim from the evidence section above "
        f"(e.g. \"{attr_features[0] if attr_features else 'FEATURE'}\"), not the focused feature '{target_prot}' itself."
    )
    return text
