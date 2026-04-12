"""Shared one-line GRPO training metrics for stdout / log files."""

from __future__ import annotations


def format_grpo_train_metrics_line(logs: dict[str, float], global_step: int) -> str:
    """Single-line summary: loss, lr, per-reward, KL, etc."""
    parts = [f"[GRPO] step={global_step}"]
    if "loss" in logs and logs["loss"] is not None:
        parts.append(f"loss={float(logs['loss']):.6f}")
    lr = logs.get("learning_rate")
    if lr is not None:
        parts.append(f"lr={float(lr):.2e}")
    acc_key = "rewards/answer_accuracy_reward"
    fmt_key = "rewards/answer_format_reward"
    if acc_key in logs:
        parts.append(f"accuracy_reward={float(logs[acc_key]):.6f}")
    if fmt_key in logs:
        parts.append(f"format_reward={float(logs[fmt_key]):.6f}")
    for key in sorted(logs):
        if (
            key.startswith("rewards/")
            and key not in (acc_key, fmt_key)
            and not key.startswith("rewards_weight/")
        ):
            short = key.split("/", 1)[-1]
            parts.append(f"{short}={float(logs[key]):.6f}")
    if "reward" in logs:
        parts.append(f"reward={float(logs['reward']):.6f}")
    if "reward_std" in logs:
        parts.append(f"reward_std={float(logs['reward_std']):.6f}")
    if "kl" in logs:
        parts.append(f"kl={float(logs['kl']):.6f}")
    if "completion_length" in logs:
        parts.append(f"completion_length={float(logs['completion_length']):.2f}")
    gn = logs.get("grad_norm")
    if gn is not None:
        parts.append(f"grad_norm={float(gn):.6f}")
    return " | ".join(parts)
