#!/usr/bin/env python3
"""Plot Figures 12 and 13 (serving throughput) from a measurement CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


FIGURES = Path(__file__).resolve().parents[1]
PAGE_WIDTH = 736.56
PAGE_HEIGHT = 385.92
PANEL_WIDTH = 302.4
PANEL_HEIGHT = 126.0
PANEL_CORNERS = ((75.6, 51.8), (425.5, 51.8), (75.6, 212.4), (425.5, 212.4))
MODELS = (
    ("llama3_8b", "Llama-3-8B", 12000, 2000, 100, 25, 167, 20),
    ("qwen3_8b", "Qwen3-8B", 12000, 2000, 100, 20, 100, 12),
    ("qwen3_14b", "Qwen3-14B", 8000, 2000, 60, 20, 62, 10),
    ("llama33_70b", "Llama-3.3-70B", 4000, 1000, 30, 10, 27, 5),
)
METHODS = (
    ("bf16", "BF16", "#4a4a4a", (0, (1.2, 1.8)), "o", "white", 2.05, 5.0),
    ("fp8", "FP8", "#4f88ab", (0, (5.0, 2.4)), "s", "#d9eaf2", 2.25, 5.0),
    (
        "w4a8", "W4A8 CUTLASS", "#c7833f", (0, (4.2, 1.8, 1.2, 1.8)),
        "^", "#ead8bd", 2.25, 5.3,
    ),
    ("nvfp4a16", "NVFP4A16 Marlin", "#5f4f76", (0, (7.0, 2.2)), "D", "#c5b9d2", 2.25, 4.8),
    ("tensorbridge", "NVFP4A8 (ours)", "#0b2f50", "-", "o", "#0b2f50", 2.75, 6.2),
)


def format_thousands(value: float, position: float) -> str:
    if value == 0:
        return "0"
    else:
        return f"{value / 1000:g}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(12, 13), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, help="Figure 12 or Figure 13 measurement CSV"
    )
    args = parser.parse_args()

    if args.figure == 12:
        source = args.input or FIGURES / "figure12/data/measurements.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        x_field, y_field = "num_prompts", "output_tok_per_s"
    else:
        source = args.input or FIGURES / "figure13/data/measurements.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        x_field, y_field = "request_rate", "request_throughput"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.use14corefonts": True,
            "axes.linewidth": 0.55,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(PAGE_WIDTH / 72, PAGE_HEIGHT / 72))
    figure.patch.set_facecolor("white")
    legend_handles = []
    legend_labels = []

    for model_index, (model, model_label, offline_limit, offline_step,
                      online_limit, online_step, rate_limit, rate_step) in enumerate(MODELS):
        left, top = PANEL_CORNERS[model_index]
        axis = figure.add_axes(
            [left / PAGE_WIDTH, (PAGE_HEIGHT - top - PANEL_HEIGHT) / PAGE_HEIGHT,
             PANEL_WIDTH / PAGE_WIDTH, PANEL_HEIGHT / PAGE_HEIGHT]
        )
        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        axis.grid(axis="y", which="major", color="#d7d7d7", linewidth=0.43)
        axis.grid(axis="y", which="minor", color="#ececec", linewidth=0.3)
        axis.tick_params(axis="both", which="major", direction="out", length=2.6,
                         width=0.5, pad=2.4, labelsize=17)
        axis.tick_params(axis="y", which="minor", length=1.5, width=0.4)
        for spine in axis.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(0.55)

        if args.figure == 12:
            axis.set_xlim(-0.18, 4.18)
            axis.set_ylim(0, offline_limit)
            axis.set_xticks(range(5), ("1", "4", "16", "64", "256"))
            axis.yaxis.set_major_locator(MultipleLocator(offline_step))
            axis.yaxis.set_minor_locator(MultipleLocator(offline_step / 2))
            axis.yaxis.set_major_formatter(FuncFormatter(format_thousands))
            axis.axvline(2.5, color="#aaaaaa", linewidth=0.5, linestyle=(0, (4, 3)))
            if model_index < 2:
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xlabel("Batch size", fontsize=18, fontweight="bold", labelpad=4)
        else:
            axis.set_xlim(0, rate_limit)
            axis.set_ylim(0, online_limit)
            axis.xaxis.set_major_locator(MultipleLocator(rate_step))
            axis.yaxis.set_major_locator(MultipleLocator(online_step))
            axis.yaxis.set_minor_locator(MultipleLocator(online_step / 2))
            if model_index < 2:
                axis.set_xlabel("")
            else:
                axis.set_xlabel("Request rate (req/s)", fontsize=18, fontweight="bold",
                                labelpad=4)

        series_by_method: dict[str, list[tuple[float, float]]] = {}
        for (method, method_label, color, line_style, marker,
             marker_fill, width, size) in METHODS:
            points = sorted(
                (float(row[x_field]), float(row[y_field])) for row in rows
                if row["model"] == model and row["method"] == method
            )
            series_by_method[method] = points
            if args.figure == 12:
                plot_x = [(1, 4, 16, 64, 256).index(int(point[0])) for point in points]
            else:
                plot_x = [point[0] for point in points]
            line = axis.plot(
                plot_x, [point[1] for point in points],
                color=color, linestyle=line_style, linewidth=width, marker=marker,
                markersize=size, markerfacecolor=marker_fill, markeredgecolor="#111111",
                markeredgewidth=0.45, label=method_label, zorder=4,
            )[0]
            legend_handles.append(line)
            legend_labels.append(method_label)

        baseline = dict(series_by_method["bf16"])
        best_x, best_ratio = max(
            ((x, throughput / baseline[x]) for x, throughput in
             series_by_method["tensorbridge"]),
            key=lambda point: point[1],
        )
        if args.figure == 12:
            annotation = f"Max vs BF16\n{best_ratio:.2f}x @ B={best_x:g}"
        else:
            annotation = f"Max vs BF16\n{best_ratio:.2f}x @ R={best_x:g}"
        axis.text(0.024, 0.91, model_label, transform=axis.transAxes, ha="left", va="top",
                  fontsize=26, fontweight="bold", color="#111111")
        if args.figure == 13 and model_index == 0:
            axis.text(0.97, 0.35, annotation, transform=axis.transAxes,
                      ha="right", va="top", fontsize=18, fontweight="bold",
                      linespacing=0.85, color="#0b2f50")
        else:
            axis.text(0.034, 0.69, annotation, transform=axis.transAxes,
                      ha="left", va="top", fontsize=18, fontweight="bold",
                      linespacing=0.85, color="#0b2f50")

    legend = figure.legend(
        legend_handles[:5], legend_labels[:5], loc="upper center", ncol=5,
        bbox_to_anchor=(0.52, 0.985), frameon=True, fancybox=False,
        edgecolor="#999999", facecolor="white", fontsize=15.5,
        borderpad=0.25, handlelength=2.15, handletextpad=0.38, columnspacing=0.94,
    )
    legend.get_frame().set_linewidth(0.55)
    legend.get_texts()[-1].set_color("#0b2f50")
    if args.figure == 12:
        figure.supylabel("Output throughput (token/s)", x=0.019, y=0.5,
                        fontsize=18, fontweight="bold")
    else:
        figure.supylabel("Request throughput (req/s)", x=0.024, y=0.5,
                        fontsize=18, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
