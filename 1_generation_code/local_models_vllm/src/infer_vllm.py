#!/usr/bin/env python
"""Qwen3.5-9B inference on decision-tree interpretation prompts, data-parallel with vLLM.

One TP1 engine per GPU; the parent re-invokes this file per GPU with --shard, forwarding
its own argv so no flag can be dropped. Each shard keeps --max-num-seqs requests running,
submits the next prompt (or a retry) as soon as one finishes, and appends each row the
moment it is final, so an interrupted run resumes where it stopped. The resolved settings are saved to <out-dir>/run_config.json.

Prompt and output settings (independent)
  --thinking            Qwen chat template with enable_thinking=True. The text before
                        </think> is stored as `reasoning`; a row with no </think> hit
                        max_tokens mid-reasoning and fails validation.
  --prompt-format json  = ../prompt.py: system prompt + JSON_OUTPUT_INSTRUCTION; the
                        content after the reasoning must parse to {thinking, answer}.
  --prompt-format plain the prompt as given. With --thinking, thinking = reasoning and answer =
                        the text after </think>; without it (models that have no thinking mode),
                        thinking is null and the whole output is the answer.
  --guided              schema-constrained JSON (json format without --thinking only).

Decode defaults: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5,
repetition_penalty 1.0. Rows failing validation are retried with seed+attempt, up to
--max-format-retries attempts; at temperature 0 there is a single attempt.

Inputs (--input, .jsonl or .json)
  * rows with system_prompt / user_prompt (or the four *_text parts), e.g.
    ../synthetic_prompt_dataset_v3_clean_refined_prompt_0828.jsonl or
    ../data/case_studies.jsonl; key = row["key"], else request-<id>
  * SFT rows {"key", "messages": [system, user, assistant]}; the assistant turn is kept
    as `reference`

Outputs in --out-dir
  shard_<i>.jsonl / shard_<i>.log    per-GPU raw rows and engine logs
  raw_responses.jsonl                every row, input order, with attempts_log
  sft_ready_results.jsonl            rows that passed validation, {"key", "messages"}

    conda activate vllm-nfs       # see env/ for the build
    export HF_HOME=/path/to/hf-cache
    python infer_vllm.py --input ../synthetic_prompt_dataset_v3_clean_refined_prompt_0828.jsonl \
        --thinking --prompt-format json --out-dir runs/synthetic_think_json --gpus 4,5,6,7
    python infer_vllm.py --input ../data/case_studies.jsonl --thinking --prompt-format plain \
        --temperature 0 --out-dir runs/case_studies_think_t0 --gpus 4,5,6,7
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import interpret_prompt as P  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tokenizer-mode", default="auto", help="'mistral' for models with no HF chat template")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--prompt-format", choices=["json", "plain"], required=True)
    ap.add_argument("--guided", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N rows")
    ap.add_argument("--sample-size", type=int, default=None, help="random N rows (after --limit)")
    ap.add_argument("--sample-seed", type=int, default=0)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-format-retries", type=int, default=3)

    # prompts are <=2.5k tokens; 32768 output is Qwen3.5's recommended budget for general tasks
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=36000)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64, help="requests running at once per GPU")
    ap.add_argument("--eager", action="store_true", help="enforce_eager (no CUDA graphs)")
    ap.add_argument("--bench-minutes", type=float, default=None, help="stop after N minutes and print throughput")
    ap.add_argument("--bench-warmup", type=float, default=4.0)

    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel size. 1 = one engine per GPU (data parallel, fastest when the "
                         "model fits on one card); >1 = a single engine across --gpus, for models that do not")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--gpu", default="0", help="GPU index for this shard, or a comma list when --tp > 1")
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    if args.guided and (args.thinking or args.prompt_format != "json"):
        ap.error("--guided only works with --prompt-format json and no --thinking")
    return args


def load_rows(args):
    """[{id, key, system, user, reference?}] in input order, after --limit / --sample-size."""
    if args.input.endswith(".json"):
        obj = json.load(open(args.input))
        raw = obj if isinstance(obj, list) else [obj]
    else:
        raw = [json.loads(line) for line in open(args.input) if line.strip()]
    if args.limit is not None:
        raw = raw[:args.limit]

    rows = []
    for idx, data in enumerate(raw):
        if "messages" in data:
            by_role = {m["role"]: m["content"] for m in data["messages"]}
            row = {"key": data.get("key", f"request-{idx}"),
                   "system": by_role.get("system") or P.FALLBACK_SYSTEM_PROMPT,
                   "user": by_role["user"]}
            if "assistant" in by_role:
                row["reference"] = by_role["assistant"]
        else:
            system, user = P.build_prompts(data)
            row = {"key": data.get("key") or f"request-{data.get('id', idx)}", "system": system, "user": user}
        row["id"] = idx
        rows.append(row)

    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate keys in input; resume and merge rely on unique keys")
    if args.sample_size is not None:
        rows = random.Random(args.sample_seed).sample(rows, min(args.sample_size, len(rows)))
        rows.sort(key=lambda r: r["id"])
    return rows


def _validated(reasoning, answer):
    """(thinking, answer, reason) for an already-split pair."""
    ok, reason = P.validate_response({"thinking": reasoning, "answer": answer})
    return (reasoning, answer, reason) if ok else (None, None, reason)


def parse_output(text, args):
    """(reasoning, thinking, answer, reason); thinking/answer are None when the text is unusable."""
    reasoning, content = None, text
    if args.thinking:
        # Qwen/Nemotron close the block with </think>; Gemma 4 uses <|channel>thought … <channel|>,
        # whose markers vLLM strips as special tokens -- the text then opens with "thought\n" and the
        # answer begins at the first markdown header glued to the end of the reasoning ("…Zone.### ").
        if "</think>" not in text and "<channel|>" not in text and text.lstrip().startswith("thought"):
            body = text.lstrip()[len("thought"):]
            m = re.search(r"(?<![\n#])#{1,4} |^#{1,4} ", body, re.M)  # inline or line-start header
            if m:
                head = body[:m.start()].rstrip().rstrip("-").rstrip()
                return (head,) + _validated(head, body[m.start():].strip())
        for tag in ("</think>", "<channel|>"):
            if tag in text:
                reasoning, content = text.split(tag, 1)
                reasoning = reasoning.replace("<think>", "").replace("<|channel>thought", "").strip()
                break
        else:  # the model wrote no reasoning block; judge the answer alone
            ok = len(text.strip()) >= 50
            return None, "", text.strip(), "ok" if ok else "no reasoning block and no usable answer"
    if args.prompt_format == "json":
        obj = P.extract_json_lenient(content)
        if not obj:
            return reasoning, None, None, "invalid JSON"
    else:
        obj = {"thinking": reasoning, "answer": content.strip()}
        if not args.thinking:  # no reasoning block exists; judge the answer alone
            ok = isinstance(obj["answer"], str) and len(obj["answer"].strip()) >= 50
            return None, "", obj["answer"], "ok" if ok else "missing or suspiciously short answer"
    ok, reason = P.validate_response(obj)
    return (reasoning, obj["thinking"], obj["answer"], reason) if ok else (reasoning, None, None, reason)


def render_prompt(tok, system, user, thinking):
    """Chat-template string, or {"prompt_token_ids": …} for tokenizers that only tokenize (Mistral)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)
    except TypeError:  # template takes no enable_thinking
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return {"prompt_token_ids": tok.apply_chat_template(msgs)}


def run_shard(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # a comma list when --tp > 1
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

    mine = load_rows(args)[args.shard::args.n_shards]
    out = os.path.join(args.out_dir, f"shard_{args.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {json.loads(line)["key"] for line in fh if line.strip()}
    todo = [r for r in mine if r["key"] not in done]
    print(f"[shard {args.shard}] {len(mine)} rows, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              tokenizer_mode=args.tokenizer_mode,
              **({"config_format": "mistral", "load_format": "mistral"} if args.tokenizer_mode == "mistral" else {}),
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=args.max_num_seqs, enforce_eager=args.eager, trust_remote_code=True,
              seed=args.seed, disable_log_stats=args.bench_minutes is None,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok, engine = llm.get_tokenizer(), llm.llm_engine
    suffix = P.JSON_OUTPUT_INSTRUCTION if args.prompt_format == "json" else ""
    attempts = args.max_format_retries if args.temperature > 0 else 1
    for r in todo:
        r["system_sent"] = r["system"] + suffix
        r["prompt"] = render_prompt(tok, r["system_sent"], r["user"], args.thinking)
        r["attempts_log"] = []

    def params(attempt):
        sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
            presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens, seed=args.seed + attempt - 1,
            structured_outputs=StructuredOutputsParams(json=P.RESPONSE_SCHEMA) if args.guided else None)
        sp.output_kind = RequestOutputKind.FINAL_ONLY
        return sp

    # Continuous feeding: keep max_num_seqs requests in the engine, submit the next prompt (or a
    # retry) as soon as one finishes, and save each row the moment it is final.
    queue = deque((r, 1) for r in todo)
    inflight = {}
    t0 = last_report = time.time()
    n_written, n_tokens = len(done), 0
    bench = BenchMeter(engine, args) if args.bench_minutes is not None else None
    with open(out, "a") as fh:
        while queue or inflight:
            while queue and len(inflight) < args.max_num_seqs:
                r, attempt = queue.popleft()
                rid = f"{r['key']}#{attempt}"
                engine.add_request(rid, r["prompt"], params(attempt))
                inflight[rid] = (r, attempt)
            for res in engine.step():
                if not res.finished:
                    continue
                r, attempt = inflight.pop(res.request_id)
                o = res.outputs[0]
                n_tokens += len(o.token_ids)
                reasoning, thinking, answer, reason = parse_output(o.text, args)
                log = {"attempt": attempt, "valid": thinking is not None, "reason": reason,
                       "finish_reason": o.finish_reason, "output_tokens": len(o.token_ids)}
                if thinking is None and attempt < attempts:  # the final attempt's text is raw_response
                    log["text"] = o.text
                r["attempts_log"].append(log)
                r.update(raw_response=o.text, reasoning=reasoning, thinking=thinking, answer=answer)
                if thinking is not None or attempt == attempts:
                    fh.write(json.dumps(_record(r, attempt), ensure_ascii=False) + "\n")
                    fh.flush()
                    n_written += 1
                else:
                    queue.appendleft((r, attempt + 1))
            now = time.time()
            if bench is not None and bench.tick(now):
                break
            if now - last_report > 300:
                last_report = now
                print(f"[shard {args.shard}] {n_written}/{len(mine)} rows written, {len(inflight)} running, "
                      f"{n_tokens / (now - t0):.0f} output tok/s (finished requests), {(now - t0) / 60:.1f} min",
                      flush=True)
    if bench is None:
        print(f"[shard {args.shard}] done: {n_written}/{len(mine)} rows, {(time.time() - t0) / 60:.1f} min", flush=True)


class BenchMeter:
    """--bench-minutes: steady-state generation throughput from engine counters, measured
    between --bench-warmup minutes and --bench-minutes after the first request."""

    def __init__(self, engine, args):
        self.engine, self.args = engine, args
        self.t0 = time.time()
        self.start = None

    def _counters(self):
        vals = {}
        for m in self.engine.get_metrics():
            if m.name in ("vllm:generation_tokens", "vllm:num_preemptions", "vllm:prompt_tokens"):
                vals[m.name] = vals.get(m.name, 0) + getattr(m, "value", 0)
        return vals

    def tick(self, now):
        if self.start is None and now - self.t0 >= 60 * self.args.bench_warmup:
            self.start = (now, self._counters())
        if now - self.t0 < 60 * self.args.bench_minutes:
            return False
        t1, c1 = now, self._counters()
        t_s, c_s = self.start
        dt = t1 - t_s
        gen = c1.get("vllm:generation_tokens", 0) - c_s.get("vllm:generation_tokens", 0)
        pre = c1.get("vllm:num_preemptions", 0) - c_s.get("vllm:num_preemptions", 0)
        print(f"[bench gpu {self.args.gpu}] eager={self.args.eager} max_num_seqs={self.args.max_num_seqs} "
              f"gpu_mem={self.args.gpu_mem}: {gen / dt:.0f} output tok/s over {dt / 60:.1f} min, "
              f"{pre:.0f} preemptions", flush=True)
        return True


def _record(r, attempt):
    ok = r["thinking"] is not None
    rec = {"key": r["key"], "id": r["id"],
           "format_validation": "passed" if ok else f"failed_after_{attempt}_attempts",
           "attempts_log": r["attempts_log"],
           "raw_system_sent": r["system_sent"], "raw_prompt_sent": r["user"],
           "raw_response": r["raw_response"], "reasoning": r["reasoning"],
           "thinking": r["thinking"], "answer": r["answer"], "system": r["system"]}
    if "reference" in r:
        rec["reference"] = r["reference"]
    return rec


def merge(args, n_shards, expected):
    rows = []
    for i in range(n_shards):
        path = os.path.join(args.out_dir, f"shard_{i}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        rows += [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) != expected:
        raise SystemExit(f"merged {len(rows)} rows, expected {expected}")
    rows.sort(key=lambda r: r["id"])

    raw_path = os.path.join(args.out_dir, "raw_responses.jsonl")
    sft_path = os.path.join(args.out_dir, "sft_ready_results.jsonl")
    passed = 0
    with open(raw_path, "w") as raw_fh, open(sft_path, "w") as sft_fh:
        for r in rows:
            raw_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["format_validation"] == "passed":
                passed += 1
                sft = P.sft_row(r["key"], r["system"], r["raw_prompt_sent"], r["thinking"], r["answer"])
                sft_fh.write(json.dumps(sft, ensure_ascii=False) + "\n")

    last = [r["attempts_log"][-1] for r in rows]
    truncated = sum(a["finish_reason"] == "length" for a in last)
    tokens = sorted(a["output_tokens"] for a in last)
    print(f"{len(rows)} rows: {passed} passed, {len(rows) - passed} failed, "
          f"{truncated} hit max_tokens on the last attempt, output tokens "
          f"mean {sum(tokens) / len(tokens):.0f} / max {tokens[-1]}")
    for r in rows:
        if r["format_validation"] != "passed":
            print(f"  - {r['key']}: {r['attempts_log'][-1]['reason']}")
    print(f"-> {raw_path}\n-> {sft_path}")


def busy_gpus(gpus):
    used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout
    mem = {int(i): int(m) for i, m in (line.split(",") for line in used.strip().splitlines())}
    return [g for g in gpus if mem[int(g)] > 2000]


def main():
    args = parse_args()
    if args.shard >= 0:
        run_shard(args)
        return

    gpus = [g for g in args.gpus.split(",") if g]
    if args.tp > 1 and args.tp != len(gpus):
        raise SystemExit(f"--tp {args.tp} needs exactly {args.tp} GPUs in --gpus (got {len(gpus)})")
    busy = busy_gpus(gpus)
    if busy:
        raise SystemExit(f"GPUs {busy} have >2 GB in use; check owners, run ./gpu_reclaim.sh <gpu> "
                         f"for our own orphaned engines, or pick other --gpus")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args)
    n = 1 if args.tp > 1 else min(len(gpus), len(rows))
    config = {k: v for k, v in vars(args).items() if k not in ("shard", "gpu", "n_shards")}
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as fh:
        json.dump(dict(config, n_rows=len(rows), started=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    how = f"TP={args.tp} across GPUs {gpus}" if args.tp > 1 else f"one engine per GPU {gpus[:n]}"
    print(f"{len(rows)} rows, thinking={args.thinking}, format={args.prompt_format}, "
          f"T={args.temperature}, {how} -> {args.out_dir}", flush=True)

    procs = []
    for i, gpu in enumerate([",".join(gpus)] if args.tp > 1 else gpus[:n]):
        log = open(os.path.join(args.out_dir, f"shard_{i}.log"), "a")
        cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + [
            "--shard", str(i), "--gpu", gpu, "--n-shards", str(n)]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        raise SystemExit(f"shard failures: {codes} (see {args.out_dir}/shard_<i>.log)")
    merge(args, n, len(rows))


if __name__ == "__main__":
    main()
