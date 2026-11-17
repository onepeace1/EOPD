"""Difficulty-stratified DAPO-Math-17k training set for the 4B OPD/EOPD runs.

DAPO-Math-17k has no difficulty labels, so difficulty is measured: the untrained student
(Qwen3-4B-Base) answers each pool problem K times with the training sampler (T=1.0, top-p 1.0,
4096 tokens, same chat template) and its pass rate sets the bin:
    hard   = 0/K correct      medium = 1..K/2 correct      easy = > K/2 correct

  python data/prep_dapo_4b.py full                    # PAPER SETUP: train_full.parquet + 100-question probe
  python data/prep_dapo_4b.py score --gpus 0,2,5,6     # ~15 min, writes data/dapo17k/pool_scores.jsonl
  python data/prep_dapo_4b.py build --steps 50         # writes train.parquet / probe.parquet / summary.json

train.parquet  steps x 128 prompts, ordered so EVERY batch of 128 has the same hard/medium/easy quota
               (pool proportions) -> hard problems appear in every one of the 50 steps.
               Train with data.shuffle=False so this order is kept.
probe.parquet  60 held-out problems (20 per bin), never trained on; used as a validation set so the
               same questions are re-answered every TEST_FREQ steps (see scripts/behavior_report.py).
extra_info of both carries {index, base_pass, base_len, bin} for per-difficulty analysis.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.expanduser("~/Rethinking-OPD/datasets/test_data/DAPO/train.parquet")
OUT = os.path.join(ROOT, "data", "dapo17k")
MODEL = "Qwen/Qwen3-4B-Base"
BATCH = 128


def load_src():
    df = pd.read_parquet(SRC).reset_index(drop=True)
    df["uid"] = df["id"].astype(str)
    return df


def pool_ids(df, pool, seed):
    return np.random.RandomState(seed).permutation(len(df))[:pool]


def cmd_score(a):
    os.makedirs(OUT, exist_ok=True)
    gpus = a.gpus.split(",")
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        cmd = [sys.executable, __file__, "shard", "--shard", str(i), "--nshards", str(len(gpus)),
               "--pool", str(a.pool), "--k", str(a.k), "--seed", str(a.seed)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"{OUT}/score_shard{i}.log", "w"),
                                      stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        sys.exit(f"shard failed: rcs={rcs} (see {OUT}/score_shard*.log)")
    with open(f"{OUT}/pool_scores.jsonl", "w") as f:
        for i in range(len(gpus)):
            f.write(open(f"{OUT}/pool_scores_shard{i}.jsonl").read())
    print("wrote", f"{OUT}/pool_scores.jsonl")


def cmd_shard(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, ROOT)
    from data.reward import compute_score

    df = load_src()
    ids = pool_ids(df, a.pool, a.seed)[a.shard::a.nshards]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [tok.apply_chat_template(list(df.prompt[i]), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for i in ids]
    llm = LLM(MODEL, gpu_memory_utilization=0.85, max_model_len=2048 + 4096, seed=a.shard)
    outs = llm.generate(prompts, SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=4096, seed=a.shard))
    with open(f"{OUT}/pool_scores_shard{a.shard}.jsonl", "w") as f:
        for i, o in zip(ids, outs):
            gt = df.reward_model[i]["ground_truth"]
            s = [compute_score("DAPO", c.text, gt) for c in o.outputs]
            f.write(json.dumps({"row": int(i), "uid": df.uid[i], "scores": s,
                                "lens": [len(c.token_ids) for c in o.outputs],
                                "truncated": [c.finish_reason == "length" for c in o.outputs]}) + "\n")


def bin_of(p, k):
    c = round(p * k)
    return "hard" if c == 0 else ("medium" if c <= k // 2 else "easy")


def cmd_build(a):
    df = load_src()
    sc = pd.read_json(f"{OUT}/pool_scores.jsonl", lines=True)
    k = len(sc.scores[0])
    sc["base_pass"] = sc.scores.map(np.mean)
    sc["base_len"] = sc.lens.map(np.mean)
    sc["bin"] = sc.base_pass.map(lambda p: bin_of(p, k))
    # drop prompts verl would filter (> MAX_PROMPT tokens) so it cannot shift the per-batch quotas
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    plen = sc.row.map(lambda i: len(tok.apply_chat_template(list(df.prompt[int(i)]), add_generation_prompt=True,
                                                            tokenize=True, enable_thinking=False)))
    n_long = int((plen > a.max_prompt).sum())
    sc = sc[plen <= a.max_prompt]
    rng = np.random.RandomState(a.seed)
    sc = sc.iloc[rng.permutation(len(sc))].reset_index(drop=True)

    bins = ["hard", "medium", "easy"]
    probe = pd.concat([sc[sc.bin == b].head(a.probe_per_bin) for b in bins])
    rest = sc.drop(probe.index)
    need = a.steps * BATCH
    assert len(rest) >= need, f"pool too small: {len(rest)} < {need}"
    frac = rest.bin.value_counts(normalize=True)
    quota = {b: int(round(BATCH * frac.get(b, 0))) for b in bins}
    quota["medium"] += BATCH - sum(quota.values())               # absorb rounding
    queues = {b: list(rest[rest.bin == b].index) for b in bins}
    order = []
    for _ in range(a.steps):
        batch = []
        for b in bins:
            take = min(quota[b], len(queues[b]))
            batch += queues[b][:take]; queues[b] = queues[b][take:]
        while len(batch) < BATCH:                                 # a bin ran dry: top up from the largest
            b = max(bins, key=lambda x: len(queues[x])); batch.append(queues[b].pop(0))
        rng.shuffle(batch); order += batch

    def rows(sel, split):
        out = []
        for _, r in sel.iterrows():
            src = df.iloc[int(r.row)]
            out.append({"data_source": "DAPO-probe" if split == "probe" else "DAPO",
                        "prompt": list(src.prompt), "ability": "math",
                        "reward_model": {"style": "rule", "ground_truth": str(src.reward_model["ground_truth"])},
                        "extra_info": {"split": split, "index": r.uid, "base_pass": float(r.base_pass),
                                       "base_len": float(r.base_len), "bin": r.bin}})
        return pd.DataFrame(out)

    train = rows(sc.loc[order], "train")
    pr = rows(probe, "probe")
    train.to_parquet(f"{OUT}/train.parquet"); pr.to_parquet(f"{OUT}/probe.parquet")
    per_batch = train.extra_info.map(lambda e: e["bin"]).values.reshape(a.steps, BATCH)
    summary = {"k": k, "pool": len(sc), "dropped_long_prompts": n_long, "pool_bins": sc.bin.value_counts().to_dict(),
               "pool_mean_pass": float(sc.base_pass.mean()),
               "pool_trunc_rate": float(np.mean([t for ts in sc.truncated for t in ts])),
               "train_prompts": len(train), "steps": a.steps, "batch": BATCH, "quota_per_batch": quota,
               "min_hard_in_any_batch": int((per_batch == "hard").sum(1).min()),
               "train_bins": train.extra_info.map(lambda e: e["bin"]).value_counts().to_dict(),
               "probe_bins": pr.extra_info.map(lambda e: e["bin"]).value_counts().to_dict()}
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


def cmd_full(a):
    """Paper setup (EOPD Table 9, 4B student: DAPO, 2 epochs, shuffled): train on all of DAPO-Math-17k except a
    random held-out probe set. No scoring needed - probe difficulty is read off the step-0 validation instead."""
    os.makedirs(OUT, exist_ok=True)
    df = load_src()
    perm = np.random.RandomState(a.seed).permutation(len(df))
    probe_ids, train_ids = perm[:a.probe], perm[a.probe:]

    def rows(ids, split):
        return pd.DataFrame([{"data_source": "DAPO-probe" if split == "probe" else "DAPO",
                              "prompt": list(df.prompt[i]), "ability": "math",
                              "reward_model": {"style": "rule", "ground_truth": str(df.reward_model[i]["ground_truth"])},
                              "extra_info": {"split": split, "index": df.uid[i]}} for i in ids])
    rows(train_ids, "train").to_parquet(f"{OUT}/train_full.parquet")
    # verl concatenates val sets with HF datasets: the probe must match MATH-500/AIME24's schema exactly
    import pyarrow as pa, pyarrow.parquet as pq
    schema = pq.read_schema(os.path.expanduser("~/Rethinking-OPD/datasets/test_data/MATH-500/test.parquet"))
    pr = rows(probe_ids, "probe")
    pr["source"], pr["id"] = "dapo-math", [df.uid[i] for i in probe_ids]
    pq.write_table(pa.Table.from_pandas(pr[schema.names], schema=schema.remove_metadata(), preserve_index=False),
                   f"{OUT}/probe.parquet")
    print(json.dumps({"train": len(train_ids), "probe": len(probe_ids),
                      "steps_per_epoch": len(train_ids) // BATCH}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["score", "shard", "build", "full"])
    p.add_argument("--gpus", default="0,2,5,6")
    p.add_argument("--pool", type=int, default=8192)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--probe_per_bin", type=int, default=20)
    p.add_argument("--max_prompt", type=int, default=2048)
    p.add_argument("--probe", type=int, default=100)
    a = p.parse_args()
    {"score": cmd_score, "shard": cmd_shard, "build": cmd_build, "full": cmd_full}[a.cmd](a)
