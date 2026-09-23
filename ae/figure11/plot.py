#!/usr/bin/env python3
"""Plot paper-shape kernel speedups from the Markdown latency table."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.ticker import MultipleLocator


DEFAULT_INPUT = Path(__file__).resolve().parent / "data/latencies.md"
DEFAULT_OUTPUT = Path("ae/figure11/results/figure11.png")

BASELINE_DISPLAY = [
    ("Torch BF16", "cuBLAS BF16", "#ffffff", "///"),
    ("Torch FP8", "cuBLAS FP8", "#d9eaf2", "xx"),
    ("vLLM CUTLASS W4A8", "vLLM CUTLASS W4A8", "#ead8bd", None),
    ("QServe W4A8", "QServe W4A8", "#d8a167", None),
    ("TRT-LLM FP8", "TRT-LLM FP8", "#b7d7e6", None),
    ("TRT-LLM W4A8", "TRT-LLM W4A8", "#6f9fbe", None),
    ("TRT-LLM W4A16", "TRT-LLM W4A16", "#bd5a4a", None),
    ("NVFP4A16 Marlin", "NVFP4A16 Marlin", "#756388", None),
    ("NVFP4A8 (ours)", "NVFP4A8 (ours)", "#0b2f50", None),
]

M_ORDER = [16, 128, 512, 4096]
MODEL_ORDER = ["Qwen3-8B", "Llama3-70B", "DeepSeek-V3"]
MODULE_ORDER = {
    "Qwen3-8B": ["qkv_fused", "mlp_gate_up_fused", "mlp_down"],
    "Llama3-70B": ["qkv_fused", "mlp_gate_up_fused", "mlp_down"],
    "DeepSeek-V3": [
        "dense_mlp_gate_up_fused",
        "dense_mlp_down",
        "moe_mlp_gate_up_fused",
        "moe_mlp_down",
    ],
}
MODULE_SHORT = {
    "qkv_fused": "QKV",
    "mlp_gate_up_fused": "MLP up+gate",
    "mlp_down": "MLP down",
    "dense_mlp_gate_up_fused": "Dense up+gate",
    "dense_mlp_down": "Dense down",
    "moe_mlp_gate_up_fused": "MoE up+gate",
    "moe_mlp_down": "MoE down",
}


def parse_markdown_latency_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| shape_id | model | module |"):
            start = i
            break
    if start is None:
        raise ValueError(f"Latency pivot table not found in {path}")

    table_lines = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    if len(table_lines) < 3:
        raise ValueError("Latency pivot table is too short")

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def to_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def build_records(rows: list[dict[str, str]]) -> dict[int, list[dict[str, object]]]:
    by_m_module: dict[int, dict[tuple[str, str], dict[str, object]]] = defaultdict(dict)
    for row in rows:
        m = int(row["M"])
        item: dict[str, object] = {
            "shape_id": row["shape_id"],
            "model": row["model"],
            "module": row["module"],
            "N": int(row["N"]),
            "K": int(row["K"]),
            "latencies": {raw: to_float(row[raw]) for raw, _, _, _ in BASELINE_DISPLAY},
        }
        by_m_module[m][(row["model"], row["module"])] = item

    ordered: dict[int, list[dict[str, object]]] = {}
    for m in M_ORDER:
        items = []
        for model in MODEL_ORDER:
            for module in MODULE_ORDER[model]:
                items.append(by_m_module[m][(model, module)])
        ordered[m] = items
    return ordered


def geo_mean(values: list[float]) -> float:
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def compact_dim(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return f"{value / 1024:.1f}K"


def shape_label(item: dict[str, object]) -> str:
    n = int(item["N"])
    k = int(item["K"])
    return f"{MODULE_SHORT[str(item['module'])]}\n[{n},{k}]"


def plot(records: dict[int, list[dict[str, object]]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],  # Font embedded in paper 313 Figure 11.
            "font.size": 14.0,
            "axes.labelsize": 14.0,
            "axes.linewidth": 0.82,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 19.0,
            "legend.fontsize": 15.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )

    n_cfg = len(records[M_ORDER[0]])
    x_step = 0.88
    x = np.arange(n_cfg + 1) * x_step  # final column is geometric mean.
    n_bars = len(BASELINE_DISPLAY)
    bar_w = 0.082
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_w

    fig, axes = plt.subplots(
        len(M_ORDER),
        1,
        figsize=(21.0, 8.65),
        sharex=True,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")

    all_speedups = []
    for m in M_ORDER:
        for item in records[m]:
            lat = item["latencies"]  # type: ignore[assignment]
            base = float(lat["Torch BF16"])  # type: ignore[index]
            for raw, _, _, _ in BASELINE_DISPLAY:
                all_speedups.append(base / float(lat[raw]))  # type: ignore[index]
    ymax = max(3.0, math.ceil((max(all_speedups) + 0.18) * 2) / 2)
    cluster_half = 0.42
    model_regions = [
        (0, 2, "Qwen3-8B", "#edf4f8", 0.78),
        (3, 5, "Llama3-70B", "#fbf0dc", 0.78),
        (6, 9, "DeepSeek-V3", "#f7eaea", 0.78),
        (10, 10, "GeoMean", "#ececec", 0.90),
    ]
    separators = [(x[2] + x[3]) / 2, (x[5] + x[6]) / 2, (x[9] + x[10]) / 2]

    for ax, m in zip(axes, M_ORDER):
        ax.set_facecolor("#ffffff")
        for start, end, label, color, alpha in model_regions:
            left = x[start] - cluster_half
            right = x[end] + cluster_half
            if label == "DeepSeek-V3":
                right = separators[-1]
            elif label == "GeoMean":
                left = separators[-1]
            ax.axvspan(
                left,
                right,
                facecolor=color,
                alpha=alpha,
                lw=0,
                zorder=0,
            )
        items = records[m]
        speedup_by_baseline: dict[str, list[float]] = {}
        for raw, _, _, _ in BASELINE_DISPLAY:
            vals = []
            for item in items:
                lat = item["latencies"]  # type: ignore[assignment]
                vals.append(float(lat["Torch BF16"]) / float(lat[raw]))  # type: ignore[index]
            vals.append(geo_mean(vals))
            speedup_by_baseline[raw] = vals

        for idx, (raw, label, color, hatch) in enumerate(BASELINE_DISPLAY):
            is_ours = raw == "NVFP4A8 (ours)"
            bars = ax.bar(
                x + offsets[idx],
                speedup_by_baseline[raw],
                width=bar_w,
                label=label,
                color=color,
                edgecolor="#111111" if is_ours else "#4a4a4a",
                linewidth=0.95 if is_ours else 0.38,
                hatch=hatch,
                zorder=3,
            )
            for j, b in enumerate(bars):
                if j == len(x) - 1:
                    b.set_edgecolor("#111111")
                    b.set_linewidth(0.85)
                if is_ours:
                    b.set_linewidth(1.10)
                    height = b.get_height()
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        height + 0.075,
                        f"{height:.1f}x",
                        ha="center",
                        va="bottom",
                        fontsize=12.0,
                        fontweight="bold",
                        color="#0b2f50",
                        zorder=6,
                    )

        for sep in separators:
            ax.axvline(sep, color="#5f5f5f", lw=0.75, alpha=0.65, zorder=2)
        ax.axhline(1.0, color="#303030", lw=0.70, ls=(0, (4, 3)), alpha=0.75, zorder=1)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(axis="y", which="major", color="#e2e2e2", lw=0.55, alpha=0.75, zorder=1)
        ax.grid(axis="y", which="minor", color="#f0f0f0", lw=0.35, alpha=0.65, zorder=1)
        row_max = max(float(item["latencies"]["Torch BF16"]) / float(item["latencies"][raw])  # type: ignore[index]
                      for item in records[m] for raw, _, _, _ in BASELINE_DISPLAY)
        ax.set_ylim(0, max(ymax, row_max + 0.75))  # room for the value label
        ax.set_xlim(x[0] - cluster_half, x[-1] + cluster_half)
        ax.tick_params(axis="both", length=3.2, width=0.65, color="#2d2d2d")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#303030")
            spine.set_linewidth(0.82)

        ax.text(
            0.000,
            0.84,
            f"M = {m}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=17.0,
            fontweight="bold",
            fontstyle="normal",
            color="white",
            bbox=dict(
                facecolor="#111111",
                edgecolor="#111111",
                boxstyle="square,pad=0.18",
            ),
            zorder=5,
        )

        if ax is not axes[-1]:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    xlabels = [shape_label(item) for item in records[M_ORDER[0]]] + ["Geo\nmean"]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, rotation=0, ha="center")
    axes[-1].tick_params(axis="x", pad=4)
    for tick in axes[-1].get_xticklabels()[-1:]:
        tick.set_fontweight("bold")
        tick.set_color("#0b2f50")

    # Model labels under the bottom axis.
    bottom = axes[-1]
    trans = transforms.blended_transform_factory(bottom.transData, bottom.transAxes)
    for left, right, label, _, _ in model_regions:
        center = (x[left] + x[right]) / 2
        bottom.text(
            center,
            -0.75,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=23.0,
            fontweight="bold",
            color="#0b2f50" if label == "GeoMean" else "#171717",
            clip_on=False,
        )
        bottom.plot(
            [x[left] - cluster_half, x[right] + cluster_half],
            [-0.58, -0.58],
            transform=trans,
            color="#171717",
            lw=1.15 if label == "GeoMean" else 0.95,
            clip_on=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(BASELINE_DISPLAY),
        frameon=False,
        bbox_to_anchor=(0.52, 0.92),
        columnspacing=0.58,
        handlelength=0.95,
        handletextpad=0.25,
    )
    for text in legend.get_texts():
        if "ours" in text.get_text():
            text.set_fontweight("bold")
            text.set_color("#0b2f50")

    fig.supylabel(
        "Speedup over cuBLAS",
        x=0.042,
        y=0.61,
        fontsize=25.0,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.858, bottom=0.380, left=0.074, right=0.988, hspace=0.070)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = parse_markdown_latency_table(args.input)
    # Plot the baselines present in the input (a new measurement may lack some).
    BASELINE_DISPLAY[:] = [b for b in BASELINE_DISPLAY if b[0] in rows[0]]
    records = build_records(rows)
    plot(records, args.output)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
