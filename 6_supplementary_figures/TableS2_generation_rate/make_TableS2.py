"""Table S2 - hypothesis generation rates across four LLMs under input conditions.

Reads data/TableS2_attempts.csv (one row per completed attempt; no API calls) and writes TableS2_generation_rates.{csv,md,xlsx}.
Value = percentage of attempts that produced a specific hypothesis (outcome "committed": the answer commits to one specific hypothesis; caveats and low confidence allowed;
"tentative", "declined" and "no text" count as not generated), with the 95% Wilson interval in parentheses, pooled over the CITE-seq CD45RA, CosMx Ki-67 and Patch-seq input-resistance prompts.
All completed attempts, including retries and empty responses, are counted. The csv also gives successes and attempts.
Usage: python make_TableS2.py
"""
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONDITIONS = ["PITCH", "MLP + IG", "Measurement-only", "Context-only"]
LLMS = ["Gemini 3.1 Pro Preview", "Qwen3.8 2.4T-A95B", "GLM-5.3", "DeepSeek V4.1 Flash"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def main():
    a = pd.read_csv(HERE / "data" / "TableS2_attempts.csv")
    long, shown = [], []
    for c in CONDITIONS:
        row = {"Input condition": c}
        for m in LLMS:
            s = a[(a["Input condition"] == c) & (a.LLM == m)]
            k, n = int((s.Outcome == "committed").sum()), len(s)
            lo, hi = wilson(k, n)
            long.append({"Input condition": c, "LLM": m, "Successes": k, "Attempts": n, "Generation rate %": round(100 * k / n, 2), "CI low %": round(100 * lo, 2), "CI high %": round(100 * hi, 2)})
            row[m] = f"{round(100 * k / n):.0f} ({round(100 * lo):.0f}–{round(100 * hi):.0f})"
        shown.append(row)
    pd.DataFrame(long).to_csv(HERE / "TableS2_generation_rates.csv", index=False)
    sh = pd.DataFrame(shown)
    head = {m: f"{m}\nGeneration rate % (95% CI)" for m in LLMS}
    cols = ["Input condition"] + LLMS
    md = ["| " + " | ".join(["Input condition"] + [head[m].replace("\n", " ") for m in LLMS]) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in sh.iterrows()]
    foot = ("Percentage of attempts that produced a specific hypothesis, with 95% Wilson confidence intervals in parentheses, pooled across the CITE-seq CD45RA, CosMx Ki-67 and Patch-seq input-resistance tasks. "
            "All completed attempts, including retries and empty responses, were counted (3 attempts per prompt and LLM; Gemini with MLP + IG: 41 attempts because it was retried).")
    (HERE / "TableS2_generation_rates.md").write_text("\n".join(md) + "\n\n" + foot + "\n", encoding="utf-8")
    with pd.ExcelWriter(HERE / "TableS2_generation_rates.xlsx", engine="openpyxl") as w:
        sh.rename(columns=head).to_excel(w, sheet_name="Table S2", index=False)
        pd.DataFrame(long).to_excel(w, sheet_name="Counts", index=False)
        pd.DataFrame({"note": [foot]}).to_excel(w, sheet_name="Notes", index=False)
        for s in w.sheets.values():
            for col in s.columns:
                s.column_dimensions[col[0].column_letter].width = 26
    print(sh.to_string(index=False))
    print(pd.DataFrame(long)[["Input condition", "LLM", "Successes", "Attempts"]].to_string(index=False))


if __name__ == "__main__":
    main()
