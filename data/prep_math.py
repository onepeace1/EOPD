"""Same records as examples/data_preprocess/math_dataset.py, written with pandas
(the shared venv's datasets/pyarrow pair cannot write parquet)."""
import sys, datasets, pandas as pd
sys.path.insert(0, sys.argv[2])
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
out = sys.argv[1]
ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval")
inst = "Let's think step by step and output the final answer within \\boxed{}."
for split in ["train", "test"]:
    rows = []
    for i, ex in enumerate(ds[split]):
        rows.append({
            "data_source": "DigitalLearningGmbH/MATH-lighteval",
            "prompt": [{"role": "user", "content": ex["problem"] + " " + inst}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": remove_boxed(last_boxed_only_string(ex["solution"]))},
            "extra_info": {"split": split, "index": i},
        })
    pd.DataFrame(rows).to_parquet(f"{out}/{split}.parquet")
    print(split, len(rows))
