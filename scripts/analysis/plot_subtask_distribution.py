#!/usr/bin/env python3
"""
Plot the distribution of final subtask index (stend) across episodes in a task folder.

Each bar represents a subtask index; bars are split into success / failure counts.
Episode data is parsed from video filenames or companion JSON files.

Usage:
    python plot_subtask_distribution.py <task_folder>
    python plot_subtask_distribution.py <task_folder> --output plot.png
    python plot_subtask_distribution.py <task_folder> --title "StirVegetables (target)"
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FILENAME_PATTERN = re.compile(
    r"_s(?P<success>\d)_stend(?P<stend>\d+)_stmax(?P<stmax>\d+)_n(?P<n>\d+)"
)


def _parse_filename(path: Path) -> dict | None:
    m = _FILENAME_PATTERN.search(path.stem)
    if not m:
        return None
    return {
        "success": bool(int(m.group("success"))),
        "stend": int(m.group("stend")),
        "stmax": int(m.group("stmax")),
        "n_subtasks": int(m.group("n")),
    }


def load_episodes(task_folder: Path) -> list[dict]:
    """Load episode records from JSON files; fall back to filename parsing."""
    episodes = []

    json_files = sorted(task_folder.glob("*.json"))
    if json_files:
        for jf in json_files:
            # Companion mp4 has the stend/stmax/n info in its name
            parsed = _parse_filename(jf)
            if parsed is None:
                continue
            with open(jf) as f:
                data = json.load(f)
            episodes.append({**parsed, **data})
    else:
        # No JSONs — parse mp4 filenames only
        for mp4 in sorted(task_folder.glob("*.mp4")):
            parsed = _parse_filename(mp4)
            if parsed:
                episodes.append(parsed)

    return episodes


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_subtask_distribution(
    episodes: list[dict],
    task_name: str,
    metric: str = "stend",
    output_path: Path | None = None,
) -> None:
    """
    metric: "stend" (subtask at episode end) or "stmax" (max subtask reached).
    """
    if not episodes:
        print("No episodes found.")
        return

    n_subtasks = episodes[0]["n_subtasks"]
    # indices range from 0 to n_subtasks (terminal index)
    all_indices = list(range(n_subtasks + 1))

    # Count success / failure per metric value
    success_counts = defaultdict(int)
    failure_counts = defaultdict(int)
    for ep in episodes:
        idx = ep[metric]
        if ep["success"]:
            success_counts[idx] += 1
        else:
            failure_counts[idx] += 1

    total_episodes = len(episodes)
    overall_sr = sum(ep["success"] for ep in episodes) / total_episodes

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, len(all_indices) * 1.4), 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    x = np.arange(len(all_indices))
    bar_width = 0.55

    colors_fail = "#e07070"
    colors_succ = "#6aab6a"
    edge_color = "white"

    fail_vals = [failure_counts[i] for i in all_indices]
    succ_vals = [success_counts[i] for i in all_indices]

    bars_fail = ax.bar(
        x, fail_vals,
        width=bar_width,
        color=colors_fail,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Failed",
        zorder=3,
    )
    bars_succ = ax.bar(
        x, succ_vals,
        width=bar_width,
        bottom=fail_vals,
        color=colors_succ,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Succeeded",
        zorder=3,
    )

    # -----------------------------------------------------------------------
    # Count labels on top of each bar
    # -----------------------------------------------------------------------
    for i, (f, s) in enumerate(zip(fail_vals, succ_vals)):
        total = f + s
        if total == 0:
            continue
        ax.text(
            x[i], total + 0.3,
            str(total),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#333333",
        )

    # Success rate label inside each success segment
    for i, (f, s) in enumerate(zip(fail_vals, succ_vals)):
        total = f + s
        if s == 0 or total == 0:
            continue
        sr = s / total
        ax.text(
            x[i], f + s / 2,
            f"{sr:.0%}",
            ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold",
        )

    # -----------------------------------------------------------------------
    # X-axis tick labels
    # -----------------------------------------------------------------------
    def subtask_label(idx: int) -> str:
        if idx == n_subtasks:
            return f"Terminal\n(st={idx})"
        return f"Subtask {idx}\n(st={idx})"

    ax.set_xticks(x)
    ax.set_xticklabels(
        [subtask_label(i) for i in all_indices],
        fontsize=9.5,
    )

    # -----------------------------------------------------------------------
    # Axes styling
    # -----------------------------------------------------------------------
    metric_label = "final subtask index at episode end (stend)" if metric == "stend" else "max subtask index reached (stmax)"
    ax.set_xlabel(f"Subtask index — {metric_label}", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of episodes", fontsize=11, labelpad=8)
    ax.set_title(
        f"{task_name}\n"
        f"n={total_episodes} episodes  |  overall SR = {overall_sr:.1%}",
        fontsize=13, fontweight="bold", pad=14,
    )

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(f + s for f, s in zip(fail_vals, succ_vals)) * 1.18 + 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        framealpha=0.85,
        edgecolor="#cccccc",
        fontsize=10,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot stend distribution for a task eval folder.")
    parser.add_argument("task_folder", type=Path, help="Path to task folder containing .mp4 / .json files")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save plot to this path instead of showing")
    parser.add_argument("--title", "-t", type=str, default=None, help="Override plot title (defaults to folder name)")
    parser.add_argument("--metric", "-m", choices=["stend", "stmax"], default="stmax",
                        help="Subtask metric to plot: 'stend' (at episode end) or 'stmax' (max reached). Default: stend")
    args = parser.parse_args()

    task_folder = args.task_folder.resolve()
    if not task_folder.is_dir():
        raise SystemExit(f"Not a directory: {task_folder}")

    task_name = args.title or task_folder.name
    episodes = load_episodes(task_folder)

    if not episodes:
        raise SystemExit(f"No parseable episode files found in {task_folder}")

    print(f"Loaded {len(episodes)} episodes for '{task_name}' (metric: {args.metric})")
    plot_subtask_distribution(episodes, task_name, metric=args.metric, output_path=args.output)


if __name__ == "__main__":
    main()
