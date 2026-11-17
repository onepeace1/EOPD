"""Paper-style evaluation: 6 math benchmarks, 8 samples/question, T=1.0, top-p=0.8, max 8192 tokens.

    python eval_math.py --model <hf_dir> --out <results.json>

Reports per benchmark: avg@8 (mean accuracy over samples) and pass@8 (any of 8 correct).
"""
import argparse, json, os, sys, time

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.reward import compute_score  # noqa: E402

BENCH_DIR = os.path.expanduser("~/Rethinking-OPD/datasets/test_data")
BENCHES = ["MATH-500", "AIME24", "AIME25", "AMC23", "Minerva", "Olympiad-Bench"]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=8)
p.add_argument("--max_tokens", type=int, default=8192)
p.add_argument("--gpu_mem", type=float, default=0.85)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
llm = LLM(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_tokens + 1024, seed=0)
sp = SamplingParams(n=args.n, temperature=1.0, top_p=0.8, max_tokens=args.max_tokens, seed=0)

results, t0 = {}, time.time()
for b in BENCHES:
    df = pd.read_parquet(f"{BENCH_DIR}/{b}/test.parquet")
    prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for m in df["prompt"]]
    outs = llm.generate(prompts, sp)
    avg, pas, lens = [], [], []
    gen_path = os.path.splitext(args.out)[0] + f"_{b}.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(gen_path)), exist_ok=True)
    gen_f = open(gen_path, "w")
    for o, rm, pr in zip(outs, df["reward_model"], prompts):
        s = [compute_score(b, c.text, rm["ground_truth"]) for c in o.outputs]
        gen_f.write(json.dumps({"prompt": pr, "gt": str(rm["ground_truth"]), "responses": [c.text for c in o.outputs],
                                "scores": s}, ensure_ascii=False) + "\n")
        avg.append(sum(s) / len(s)); pas.append(float(max(s) > 0))
        lens += [len(c.token_ids) for c in o.outputs]
    gen_f.close()
    results[b] = {"avg@%d" % args.n: 100 * sum(avg) / len(avg), "pass@%d" % args.n: 100 * sum(pas) / len(pas),
                  "mean_len": sum(lens) / len(lens), "n_questions": len(df)}
    print(b, results[b], flush=True)

for k in ["avg@%d" % args.n, "pass@%d" % args.n]:
    results.setdefault("Average", {})[k] = sum(results[b][k] for b in BENCHES) / len(BENCHES)
results["_meta"] = {"model": args.model, "minutes": (time.time() - t0) / 60, **vars(args)}
print("Average", results["Average"])
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=2)
