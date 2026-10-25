"""Runs the FINAL prompt set (hypothesis_repeats_v2/prompts/: full pipeline, IG-only, and the newest raw-data prompt, identical for every model) through 4 models x 3 conditions x 4 datasets x N repeats (default N=3).

Models   : gemini-3.1-pro-preview (Google API) + 3 Together models (GLM-5.3, DeepSeek V4.1 Flash, Qwen3.8-2.4T-A95B).
Settings : temperature 0.5 for every call (Gemini via GenerateContentConfig, Together via the `temperature` arg);
           Together reasoning effort as in the earlier runs (GLM-5.3 xhigh, Qwen xhigh, DeepSeek V4.1 Flash max).
           If a Together model rejects `temperature`, the call is retried without it and the record says so
           (`temperature_applied: false`) -- nothing is silently changed.
Output   : hypothesis_repeats_v2/answers/<model_slug>/<dataset>_<cond>_run<k>.json, one file per call
           ({model, dataset, condition, run, temperature, temperature_applied, reasoning_effort, final_answer,
             finish_reason, usage}). Existing non-empty answers are skipped, so the run is resumable.
Usage    :
    python run_repeats_v2.py --dry-run                       # plan + input-token estimate, no API calls
    python run_repeats_v2.py                                 # everything (needs TOGETHER_API_KEY and GEMINI_API_KEY)
    python run_repeats_v2.py --models zai-org/GLM-5.3 --repeats 1 --conds cond1_ig_only --datasets pERK_EGF10   # smoke test
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
ANSWERS = HERE / "answers"
sys.path.insert(0, str(HERE))   # together_helpers.py (retry / max_tokens-escalation helpers) sits next to this script
import together_helpers as rt  # noqa: E402

VARIANT_TAG = "final_prompt_set"   # set from --variant-tag; recorded in every answer file (default = the final prompt set in prompts/, newest raw-data wording, all models)
rt.TRANSIENT_RETRY_DELAYS = [30, 60, 120, 240, 480]   # ride out Together 503 "no healthy relay" outages (~15 min) instead of failing after ~100 s

MAX_OUT_TOKENS = {"deepseek-ai/DeepSeek-V4.1-Flash": 130000}   # all other Together models: 64000 (default in call_together)
TEMPERATURE = 0.5
DATASETS = ["CITEseq_CD45RA_naive", "CosMx_CancerKi67", "Gouwens_GABAergic", "pERK_EGF10"]
CONDS = ["cond0_full_pipeline", "cond1_ig_only", "cond2_raw_data"]  # cheapest prompt first
GEMINI = "gemini-3.1-pro-preview"
TOGETHER_EFFORT = {"zai-org/GLM-5.3": "xhigh", "deepseek-ai/DeepSeek-V4.1-Flash": "max", "Qwen/Qwen3.8-2.4T-A95B": "xhigh"}
ALL_MODELS = [GEMINI] + list(TOGETHER_EFFORT)
WORKERS = {GEMINI: 4, "zai-org/GLM-5.3": 4, "deepseek-ai/DeepSeek-V4.1-Flash": 4, "Qwen/Qwen3.8-2.4T-A95B": 4}

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def slug(model):
    return model.replace("/", "__")


def _split(text):
    """Same system/user split as the earlier Gemini + Together runs (system = text before the EXPERIMENT CONTEXT header)."""
    head = "--- SYSTEM PROMPT ---"
    user_hdr = "--- EXPERIMENT CONTEXT & DATA PIPELINE ---"
    assert text.lstrip().startswith(head)
    after = text.lstrip()[len(head):]
    i = after.find(user_hdr)
    assert i != -1
    return after[:i].strip(), after[i:].strip()


def gemini_key():
    """Gemini API key from the environment only (nothing is stored in this repository)."""
    return os.environ.get("GEMINI_API_KEY")


def call_gemini(client, system_prompt, user_prompt):
    from google.genai import types
    last = None
    for delay in [0, 15, 45, 90]:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.generate_content(
                model=GEMINI, contents=user_prompt,
                config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=TEMPERATURE))
            u = r.usage_metadata
            fin = str(r.candidates[0].finish_reason) if r.candidates else None
            return {"final_answer": r.text, "finish_reason": fin, "temperature_applied": True, "reasoning_effort": None,
                    "usage": {"prompt": getattr(u, "prompt_token_count", None), "output": getattr(u, "candidates_token_count", None),
                              "thinking": getattr(u, "thoughts_token_count", None)}}
        except Exception as e:  # noqa: BLE001
            last = e
            txt = str(e)
            if not any(s in txt for s in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")):
                raise
    raise last


def call_together(client, model, system_prompt, user_prompt):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    # Start with a generous output CAP (not a target; only generated tokens are billed). The earlier runs started at the API
    # default and re-sent the whole ~500K-token prompt 2-3 times when a reasoning model exhausted it on hidden reasoning.
    # DeepSeek V4.1 Flash repeatedly used its whole 100,000-token cap on hidden reasoning (5 empty raw-data answers): cap raised 30% to
    # 130,000 and used from the first attempt (the escalation ladder tops out at 100,000, so it is skipped).
    base = {"model": model, "messages": messages, "reasoning_effort": TOGETHER_EFFORT[model],
            "max_tokens": MAX_OUT_TOKENS.get(model, 64000)}
    applied = True
    try:
        resp = rt._create_with_escalation(client, {**base, "temperature": TEMPERATURE})
    except Exception as e:  # noqa: BLE001
        if "temperature" in rt._error_text(e).lower():
            applied = False
            resp = rt._create_with_escalation(client, base)
        else:
            raise
    content, fin = rt._content_and_finish(resp)
    u = (resp.model_dump().get("usage") or {})
    return {"final_answer": content, "finish_reason": fin, "temperature_applied": applied,
            "reasoning_effort": TOGETHER_EFFORT[model],
            "usage": {"prompt": u.get("prompt_tokens"), "output": u.get("completion_tokens")}}


def one_task(model, ds, cond, run, clients):
    out = ANSWERS / slug(model) / f"{ds}_{cond}_run{run}.json"
    if out.exists():
        try:
            if (json.loads(out.read_text(encoding="utf-8")).get("final_answer") or "").strip():
                return "skip"
        except Exception:  # noqa: BLE001
            pass
    system_prompt, user_prompt = _split((PROMPTS / ds / f"{ds}_{cond}_prompt.txt").read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        rec = call_gemini(clients["gemini"], system_prompt, user_prompt) if model == GEMINI else \
            call_together(clients["together"], model, system_prompt, user_prompt)
    except Exception as e:  # noqa: BLE001
        log(f"  ERROR {model} {ds}/{cond} run{run}: {type(e).__name__}: {str(e)[:200]}")
        return "error"
    rec.update({"model": model, "dataset": ds, "condition": cond, "run": run, "temperature": TEMPERATURE,
                "prompt_variant": VARIANT_TAG, "prompts_dir": PROMPTS.name, "elapsed_s": round(time.time() - t0, 1)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    status = "ok" if (rec["final_answer"] or "").strip() else "empty"
    log(f"  {status:5s} {model} {ds}/{cond} run{run} ({rec['elapsed_s']}s, finish={rec['finish_reason']}, temp_applied={rec['temperature_applied']})")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prompts-dir", default=None, help="Folder with <dataset>/<dataset>_<cond>_prompt.txt (default: prompts/).")
    ap.add_argument("--variant-tag", default="final_prompt_set", help="Label stored in each answer file; the default marks the FINAL prompt set (see raw_prompt_versions.py).")
    ap.add_argument("--workers", type=int, default=None, help="Parallel calls PER MODEL (default 4). Use 1 on a small Together balance: "
                    "many large requests in flight at once can trip the credit check even when the balance is positive.")
    ap.add_argument("--answers-dir", default=None, help="Write answers here instead of answers/ (use for trials so the main set stays untouched).")
    args = ap.parse_args()
    if args.workers:
        for m in WORKERS:
            WORKERS[m] = args.workers
    global PROMPTS, VARIANT_TAG, ANSWERS
    if args.prompts_dir:
        PROMPTS = Path(args.prompts_dir)
    if args.answers_dir:
        ANSWERS = Path(args.answers_dir)
    VARIANT_TAG = args.variant_tag

    tasks = [(m, ds, c, r) for r in range(1, args.repeats + 1) for c in args.conds for ds in args.datasets for m in args.models]
    if args.dry_run:
        by_model = {}
        for m, ds, c, r in tasks:
            n = len((PROMPTS / ds / f"{ds}_{c}_prompt.txt").read_text(encoding="utf-8"))
            by_model.setdefault(m, [0, 0])
            by_model[m][0] += 1
            by_model[m][1] += n
        print(f"{len(tasks)} calls planned (temperature {TEMPERATURE}, {args.repeats} repeat(s)):")
        for m, (n, ch) in by_model.items():
            print(f"  {m:36s} {n:3d} calls, ~{ch/1e6:.1f}M prompt characters")
        return

    clients = {}
    if GEMINI in args.models:
        from google import genai
        key = gemini_key()
        assert key, "no Gemini key available"
        clients["gemini"] = genai.Client(api_key=key)
    if any(m != GEMINI for m in args.models):
        key = os.environ.get("TOGETHER_API_KEY")
        assert key, "TOGETHER_API_KEY is not set"
        from together import Together
        clients["together"] = Together(timeout=1800.0, api_key=key)

    counts = {}
    pools = {m: ThreadPoolExecutor(max_workers=WORKERS[m]) for m in args.models}
    futs = [pools[m].submit(one_task, m, ds, c, r, clients) for (m, ds, c, r) in tasks]
    for f in as_completed(futs):
        s = f.result()
        counts[s] = counts.get(s, 0) + 1
    for p in pools.values():
        p.shutdown()
    print("Done.", counts, "->", ANSWERS)


if __name__ == "__main__":
    main()
