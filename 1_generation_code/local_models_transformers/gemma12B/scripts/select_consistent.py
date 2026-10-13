"""
For each results/<prompt_stem>.json (5 independent responses per prompt),
pick the single response that is most representative of the other 4 --
i.e. the "consensus" response -- and save it alone into
results/most_consistent/<prompt_stem>.json.

Method: embed each of the 5 responses with a sentence-embedding model
(all-MiniLM-L6-v2), compute the 5x5 pairwise cosine similarity matrix, and
pick the response with the highest average similarity to the other four
(the medoid). This is a semantic-similarity measure, not exact text
matching -- it picks the response whose content most other runs agree with,
even if wording differs.

Usage:
    python select_consistent.py
    python select_consistent.py --results-dir ../results --out-dir ../results/most_consistent
"""

import argparse
import glob
import json
import os

# Keep the embedding model off the near-full root filesystem, consistent
# with how the main LLM weights are stored on /mnt/data.
os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "most_consistent")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def pick_most_consistent(embeddings):
    """Return (index, avg_similarity_to_others, full_similarity_matrix)."""
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = norms @ norms.T
    n = len(embeddings)
    avg_sim_excl_self = (sim_matrix.sum(axis=1) - 1.0) / (n - 1)
    best_idx = int(np.argmax(avg_sim_excl_self))
    return best_idx, float(avg_sim_excl_self[best_idx]), sim_matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description="Select the most-consistent response per prompt.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    result_paths = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not result_paths:
        raise SystemExit(f"No result files found in {args.results_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] loading embedding model {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"{'prompt':<55} {'selected_run':<13} {'seed':<7} {'avg_similarity':<15}")
    for path in result_paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)

        responses = record["responses"]
        texts = [r["response"] for r in responses]
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        best_idx, avg_sim, sim_matrix = pick_most_consistent(embeddings)
        chosen = responses[best_idx]

        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"{stem:<55} {chosen['run_index']:<13} {chosen['seed']:<7} {avg_sim:<15.4f}")

        # Single-answer output: just which model produced it and the exact,
        # unmodified answer text of the response the other 4 runs agree with
        # most (the semantic medoid) -- no prompt content, generation
        # settings, or scoring metadata included.
        out_record = {
            "model": "gemma-4-12B-it",
            "answer": chosen["response"],
        }

        out_path = os.path.join(args.out_dir, f"{stem}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_record, f, ensure_ascii=False, indent=2)

    print(f"[info] done -- wrote {len(result_paths)} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
