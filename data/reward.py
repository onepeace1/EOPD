"""Answer checker for validation (MATH-500, AIME24/25, AMC23, Minerva, OlympiadBench, MATH).

Distillation itself needs no reward; this only scores val generations (accuracy -> pass@k).
Takes the last \\boxed{} answer; math_verify first, verl's string-normalised match as fallback.
"""
from math_verify import parse, verify
from verl.utils.reward_score.math_reward import is_equiv, last_boxed_only_string, remove_boxed


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    boxed = last_boxed_only_string(solution_str[-4000:])
    if boxed is None:
        return 0.0
    try:
        pred = remove_boxed(boxed)
    except Exception:
        return 0.0
    gt = str(ground_truth)
    try:
        if verify(parse(f"\\boxed{{{gt}}}"), parse(boxed)):
            return 1.0
    except Exception:
        pass
    try:
        if is_equiv(pred, gt):
            return 1.0
        # AMC23 ground truths are stored as floats ("142.0")
        return float(abs(float(pred) - float(gt)) < 1e-6)
    except Exception:
        return 0.0
