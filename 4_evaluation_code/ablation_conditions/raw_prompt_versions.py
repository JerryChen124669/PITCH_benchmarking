"""Single source of truth for the MEASUREMENT-ONLY prompt wording.

FINAL (newest) measurement-only prompt = the v2 measurement-only prompt + ONE sentence, inserted right before ANCHOR:
    "Approach: rank the features by how strongly they correlate with {target} across the cells in the table (positively or
     negatively), then use the top-ranked features to formulate your hypothesis for how {target} is regulated."
This is what build_prompts_v2.py writes into prompts/ and what run_repeats_v2.py sends by default, for every model.

Earlier measurement-only wordings differ from the final one ONLY in that one sentence, so they are fully reconstructable from the final
prompt (nothing else changed; the data table is byte-identical in all versions). Answers generated before the final wording keep
the version they were generated under in their `prompt_variant` field:
    "standard"                          = final prompt with the Approach sentence removed
    "guide_line" / "deepseek_guide_line"= final prompt with the Approach sentence replaced by the "guide" sentence below
    "raw_corr_rank" / "final_prompt_set"= the final prompt
reconstruct() rebuilds the exact text for any of these (the judge uses it so it always shows what the model actually saw).
"""
import re

ANCHOR = "State your single best hypothesis based on the given data. Any features you rely on"


def target_of(text):
    return re.search(r"appear most associated with (.+?), and form a rationale", text).group(1)


def corr_sentence(target):
    return (f"Approach: rank the features by how strongly they correlate with {target} across the cells in the table "
            f"(positively or negatively), then use the top-ranked features to formulate your hypothesis for how {target} "
            f"is regulated. ")


def guide_sentence(target):
    return (f"Approach: first pick out the features that look most important for {target}, then use those features to "
            f"formulate your hypothesis (an exhaustive scan of every column is not required). ")


def with_final_sentence(v2_raw_text):
    """v2 measurement-only prompt (no Approach sentence) -> FINAL measurement-only prompt."""
    assert v2_raw_text.count(ANCHOR) == 1, "anchor not found exactly once"
    return v2_raw_text.replace(ANCHOR, corr_sentence(target_of(v2_raw_text)) + ANCHOR)


def reconstruct(final_text, variant):
    """Exact measurement-only prompt text that was used for an answer generated under `variant`."""
    t = target_of(final_text)
    final_s = corr_sentence(t)
    assert final_text.count(final_s) == 1, "final prompt does not contain the final Approach sentence"
    if variant in (None, "raw_corr_rank", "final_prompt_set"):
        return final_text
    if variant == "standard":
        return final_text.replace(final_s, "")
    if variant in ("guide_line", "deepseek_guide_line"):
        return final_text.replace(final_s, guide_sentence(t))
    raise ValueError(f"unknown measurement-only prompt variant: {variant}")
