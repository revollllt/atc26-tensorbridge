#!/usr/bin/env python3
"""Plot Figures 12 and 13 (serving throughput).

The data comes from `figure1{2,3}/data/measurements.csv` (our runs), from another
CSV with `--input`, or from the JSON files that `measure.sh` writes with
`--results`: `<results>/<model>/<method>/batch-<B>.json` (Figure 12) or
`rate-<R>.json` (Figure 13). `--results` also saves the collected rows to
`<results>/measurements.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


FIGURES = Path(__file__).resolve().parents[1]
BATCH_SIZES = (1, 4, 16, 64, 256)  # Figure 12 x axis
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


def collect(results: Path, figure: int) -> list[dict[str, str]]:
    """Rows in the measurements.csv format from the JSON files under `results`."""
    rows = []
    pattern = r"batch-(\d+)\.json" if figure == 12 else r"rate-([\d.]+|inf)\.json"
    for path in sorted(results.glob("*/*/*.json")):
        match = re.fullmatch(pattern, path.name)
        if not match:
            continue
        model, method = path.parts[-3], path.parts[-2]
        data = json.loads(path.read_text())
        if figure == 12:
            rows.append({"model": model, "method": method, "num_prompts": match[1],
                         "output_tok_per_s": data["output_tok_per_s"]})
        else:
            rows.append({"model": model, "method": method, "request_rate": match[1],
                         "request_throughput": data["request_throughput"]})
    if not rows:
        raise SystemExit(f"No measurements under {results}/<model>/<method>/")
    with (results / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [{key: str(value) for key, value in row.items()} for row in rows]


def format_thousands(value: float, position: float) -> str:
    if value == 0:
        return "0"
    else:
        return f"{value / 1000:g}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, choices=(12, 13), required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="measurement CSV")
    source.add_argument("--results", type=Path, help="folder written by measure.sh")
    args = parser.parse_args()

    if args.results:
        rows = collect(args.results, args.figure)
    else:
        path = args.input or FIGURES / f"figure{args.figure}/data/measurements.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    if args.figure == 12:
        x_field, y_field = "num_prompts", "output_tok_per_s"
    else:
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
            axis.set_xticks(range(len(BATCH_SIZES)), map(str, BATCH_SIZES))
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
                and (args.figure == 13 or int(float(row[x_field])) in BATCH_SIZES)
            )
            series_by_method[method] = points
            if not points:
                continue
            if args.figure == 12:
                plot_x = [BATCH_SIZES.index(int(point[0])) for point in points]
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

        axis.text(0.024, 0.91, model_label, transform=axis.transAxes, ha="left", va="top",
                  fontsize=26, fontweight="bold", color="#111111")
        if not any(series_by_method.values()):
            axis.text(0.5, 0.45, "not measured", transform=axis.transAxes, ha="center",
                      va="center", fontsize=18, color="#888888")
            continue
        baseline = dict(series_by_method["bf16"])
        ratios = [(x, throughput / baseline[x]) for x, throughput in
                  series_by_method["tensorbridge"] if x in baseline]
        if not ratios:
            continue
        best_x, best_ratio = max(ratios, key=lambda point: point[1])
        if args.figure == 12:
            annotation = f"Max vs BF16\n{best_ratio:.2f}x @ B={best_x:g}"
        else:
            annotation = f"Max vs BF16\n{best_ratio:.2f}x @ R={best_x:g}"
        if args.figure == 13 and model_index == 0:
            axis.text(0.97, 0.35, annotation, transform=axis.transAxes,
                      ha="right", va="top", fontsize=18, fontweight="bold",
                      linespacing=0.85, color="#0b2f50")
        else:
            axis.text(0.034, 0.69, annotation, transform=axis.transAxes,
                      ha="left", va="top", fontsize=18, fontweight="bold",
                      linespacing=0.85, color="#0b2f50")

    handle_by_label = dict(zip(legend_labels, legend_handles))
    labels = [method[1] for method in METHODS if method[1] in handle_by_label]
    legend = figure.legend(
        [handle_by_label[label] for label in labels], labels, loc="upper center",
        ncol=max(1, len(labels)),
        bbox_to_anchor=(0.52, 0.985), frameon=True, fancybox=False,
        edgecolor="#999999", facecolor="white", fontsize=15.5,
        borderpad=0.25, handlelength=2.15, handletextpad=0.38, columnspacing=0.94,
    )
    legend.get_frame().set_linewidth(0.55)
    for text in legend.get_texts():
        if "ours" in text.get_text():
            text.set_color("#0b2f50")
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
