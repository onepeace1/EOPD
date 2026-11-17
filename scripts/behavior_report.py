"""How does the model's behaviour on the SAME questions change during training?

    python scripts/behavior_report.py runs/qwen3_4b_dapo17k_opd [runs/qwen3_4b_dapo17k_eopd ...] \
        --out runs/report_4b.md

Reads, per run directory:
  val_generations/<step>.jsonl  fixed val questions (MATH-500, AIME24, DAPO-probe) x 4 samples, every TEST_FREQ
  rollouts/<step>.jsonl         every training step (128 prompts, score, early-token student/teacher probs)
Writes a markdown report plus CSVs next to it:
  1. val accuracy / pass@4 / length / no-\\boxed rate / truncation per step and benchmark (+ per difficulty bin
     for DAPO-probe)
  2. per-question transitions first -> last val step: gained, lost, stable (per benchmark), with examples showing
     the first-step and last-step answer to the same question
  3. training rollouts per step and difficulty bin: score, length, truncation, early-token p_student / p_teacher /
     teacher entropy (how closely the student tracks the teacher)
  4. with >1 run: per-question head-to-head at the last common val step
CSVs: <out>_val_per_step.csv, <out>_val_per_question.csv (question x step accuracy), <out>_train_per_step.csv
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.expanduser("~/Rethinking-OPD/datasets/test_data")
VAL_SETS = {"MATH-500": f"{TB}/MATH-500/test.parquet", "AIME24": f"{TB}/AIME24/test.parquet",
            "DAPO-probe": f"{ROOT}/data/dapo17k/probe.parquet"}
TRAIN = f"{ROOT}/data/dapo17k/train_full.parquet"
MAX_RESP = 4096


def load_questions(path, source):
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    out = []
    for i, r in df.iterrows():
        content = list(r.prompt)[-1]["content"]
        ei = r.extra_info if isinstance(r.extra_info, dict) else {}
        out.append({"qid": f"{source}:{ei.get('index', i)}", "source": source, "key": content.strip()[:300],
                    "bin": ei.get("bin", "-"), "base_pass": ei.get("base_pass", np.nan)})
    return out


class Matcher:
    """dump rows only carry the rendered prompt text: map it back to the question by its content prefix."""
    def __init__(self, qs):
        self.qs = qs
        self.by_prefix = defaultdict(list)
        for q in qs:
            self.by_prefix[q["key"][:60]].append(q)
        self.cache = {}

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        hit = None
        for pre, cands in self.by_prefix.items():
            if pre in text:
                hit = next((q for q in cands if q["key"] in text), cands[0])
                break
        self.cache[text] = hit
        return hit


def has_boxed(s):
    return "\\boxed" in s


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def steps_in(d):
    return sorted(int(os.path.basename(p)[:-6]) for p in glob.glob(f"{d}/*.jsonl"))


def val_rows(run, match, tok):
    rows = []
    for st in steps_in(f"{run}/val_generations"):
        for r in read_jsonl(f"{run}/val_generations/{st}.jsonl"):
            q = match(r["input"])
            if q is None:
                continue
            n_tok = len(tok(r["output"], add_special_tokens=False).input_ids) if tok else len(r["output"]) / 3.5
            rows.append({"step": st, "qid": q["qid"], "source": q["source"], "bin": q["bin"],
                         "score": float(r["score"]), "len": n_tok, "boxed": has_boxed(r["output"]),
                         "trunc": n_tok >= MAX_RESP - 1, "output": r["output"], "gts": r["gts"],
                         "question": q["key"]})
    return pd.DataFrame(rows)


def train_rows(run, match, tok):
    rows = []
    for st in steps_in(f"{run}/rollouts"):
        for r in read_jsonl(f"{run}/rollouts/{st}.jsonl"):
            q = match(r["input"])
            n_tok = len(tok(r["output"], add_special_tokens=False).input_ids) if tok else len(r["output"]) / 3.5
            et = r.get("early_tokens") or []
            rows.append({"step": st, "qid": q["qid"] if q else None, "bin": q["bin"] if q else "?",
                         "score": float(r["score"]), "len": n_tok,
                         "boxed": has_boxed(r["output"]), "trunc": n_tok >= MAX_RESP - 1,
                         "p_student": np.mean([t["p_student"] for t in et]) if et else np.nan,
                         "p_teacher": np.mean([t["p_teacher"] for t in et]) if et else np.nan,
                         "H_teacher": np.mean([t["H_teacher"] for t in et]) if et else np.nan})
    return pd.DataFrame(rows)


def agg(df, keys):
    g = df.groupby(keys)
    out = g.agg(acc=("score", "mean"), len=("len", "mean"), no_boxed=("boxed", lambda x: 1 - x.mean()),
                trunc=("trunc", "mean"), n=("score", "size"))
    if "qid" in df:
        out["pass@k"] = df.groupby(keys + ["qid"]).score.max().groupby(keys).mean()
    return out.reset_index()


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def clip(s, n=700):
    s = s.strip()
    return s if len(s) <= n else s[: n // 2] + "\n  [...]\n" + s[-n // 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    out = a.out or os.path.join(a.runs[0], "behavior_report.md")
    base = os.path.splitext(out)[0]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    except Exception:
        tok = None
    vq = sum((load_questions(p, s) for s, p in VAL_SETS.items()), [])
    vmatch, tmatch = Matcher(vq), Matcher(load_questions(TRAIN, "train"))

    L = ["# Behaviour report", ""]
    all_val, per_q_last = [], {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        L += [f"## {name}", ""]
        v = val_rows(run, vmatch, tok)
        if len(v):
            v["run"] = name
            if (v.bin == "-").all():   # no precomputed difficulty: bin by the base model's step-0 mean@4
                s0 = v[v.step == v.step.min()].groupby("qid").score.mean()
                b0 = s0.map(lambda x: "hard" if x == 0 else ("medium" if x <= 0.5 else "easy"))
                v["bin"] = v.qid.map(b0).fillna("-")
            all_val.append(v)
            L += ["### 1. Fixed validation questions, per step", "", md(agg(v, ["source", "step"])), ""]
            p = v[v.source == "DAPO-probe"]
            if len(p):
                L += ["DAPO-probe by difficulty bin (bin = pass rate before training: hard 0/4, medium <=2/4, easy >2/4):", "",
                      md(agg(p, ["bin", "step"])), ""]
            qa = v.groupby(["qid", "source", "step"]).score.mean().unstack("step")
            first, last = qa.columns.min(), qa.columns.max()
            per_q_last[name] = (qa[last], last)
            d = qa[last] - qa[first]
            L += [f"### 2. Same question, step {first} -> {last} (mean@4 per question)", ""]
            tr = pd.DataFrame({"gained (>= +0.5)": (d >= 0.5).groupby(level="source").sum(),
                               "lost (<= -0.5)": (d <= -0.5).groupby(level="source").sum(),
                               "newly solved (0 -> >0)": ((qa[first] == 0) & (qa[last] > 0)).groupby(level="source").sum(),
                               "newly broken (>0 -> 0)": ((qa[first] > 0) & (qa[last] == 0)).groupby(level="source").sum(),
                               "questions": d.groupby(level="source").size()}).reset_index()
            L += [md(tr, ".0f"), ""]
            for label, idx in [("Lost", d.sort_values().index[: a.examples]),
                               ("Gained", d.sort_values(ascending=False).index[: a.examples])]:
                for qid, src in idx:
                    if abs(d[(qid, src)]) < 0.5:
                        continue
                    ex = v[v.qid == qid]
                    f0, f1 = ex[ex.step == first].iloc[0], ex[ex.step == last].iloc[0]
                    L += [f"<details><summary>{label}: {qid} (acc {qa.loc[(qid, src), first]:.2f} -> "
                          f"{qa.loc[(qid, src), last]:.2f}, gt {f0.gts})</summary>", "",
                          "**Question:** " + clip(f0.question, 400), "",
                          f"**Step {first} answer** (score {f0.score:.0f}, {f0.len:.0f} tok):", "```", clip(f0.output), "```",
                          f"**Step {last} answer** (score {f1.score:.0f}, {f1.len:.0f} tok):", "```", clip(f1.output), "```",
                          "</details>", ""]
            qa.to_csv(f"{base}_{name}_val_per_question.csv")
        t = train_rows(run, tmatch, tok)
        if len(t):
            t["run"] = name
            g = t.groupby("step").agg(score=("score", "mean"), len=("len", "mean"),
                                      no_boxed=("boxed", lambda x: 1 - x.mean()), trunc=("trunc", "mean"),
                                      p_student=("p_student", "mean"), p_teacher=("p_teacher", "mean"),
                                      H_teacher=("H_teacher", "mean")).reset_index()
            show = g if len(g) <= 12 else g.iloc[np.unique(np.linspace(0, len(g) - 1, 12).astype(int))]
            L += ["### 3. Training rollouts per step", "", md(show), ""]
            tb = t.assign(stage=pd.cut(t.step, 3, labels=["early", "mid", "late"])).groupby(["bin", "stage"], observed=True)
            L += ["Training score / length by difficulty bin and training stage:", "",
                  md(tb.agg(score=("score", "mean"), len=("len", "mean"), trunc=("trunc", "mean"),
                            n=("score", "size")).reset_index()), ""]
            g.to_csv(f"{base}_{name}_train_per_step.csv", index=False)
            # every training problem is seen once per epoch: same question, epoch 1 vs epoch 2
            t = t.dropna(subset=["qid"]).sort_values("step")
            t["visit"] = t.groupby("qid").cumcount() + 1
            w = t[t.visit <= 2].pivot_table(index="qid", columns="visit", values=["score", "len"])
            if ("score", 2) in w.columns:
                w = w.dropna()
                s1, s2 = w[("score", 1)], w[("score", 2)]
                L += ["### 3b. Training problems seen twice (epoch 1 -> epoch 2, same question)", "",
                      md(pd.DataFrame([{"problems": len(w), "acc epoch1": s1.mean(), "acc epoch2": s2.mean(),
                                        "wrong->right": int(((s1 == 0) & (s2 > 0)).sum()),
                                        "right->wrong": int(((s1 > 0) & (s2 == 0)).sum()),
                                        "len epoch1": w[("len", 1)].mean(), "len epoch2": w[("len", 2)].mean()}])), ""]

    if all_val:
        V = pd.concat(all_val)
        agg(V, ["run", "source", "step"]).to_csv(f"{base}_val_per_step.csv", index=False)
    if len(per_q_last) > 1:
        names = list(per_q_last)
        step = min(s for _, s in per_q_last.values())
        L += [f"## 4. Head-to-head on the same questions (last common val step {step})", ""]
        V = pd.concat(all_val)
        qa = V[V.step == step].groupby(["qid", "source", "run"]).score.mean().unstack("run")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = qa[names[i]], qa[names[j]]
                h = pd.DataFrame({f"{names[i]} better": (x - y >= 0.5).groupby(level="source").sum(),
                                  f"{names[j]} better": (y - x >= 0.5).groupby(level="source").sum(),
                                  "tie": ((x - y).abs() < 0.5).groupby(level="source").sum()}).reset_index()
                L += [md(h, ".0f"), ""]
    open(out, "w").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
