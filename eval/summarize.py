"""Print Table 3 (perplexity) and/or Table 4 (mean zero-shot accuracy) from result JSONs.

Each cell shows the new measurement and, in parentheses, the paper's value from
`ae/table{3,4}/data/paper.csv`. With `--table`, the table is also written to
`ae/table<N>/results/table<N>.md`.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["llama2_7b", "llama3_8b", "qwen3_8b", "qwen3_14b", "qwen36_35b_a3b"]
METHODS = ["bf16", "w4a8", "nvfp4a16", "nvfp4", "mxfp4", "nvfp4a8", "ours_nosnc", "ours",
           "ours_kernel"]
TABLES = {3: ("ppl", "ppl", "Table 3: WikiText2 perplexity (lower is better)"),
          4: ("tasks", "avg", "Table 4: mean zero-shot accuracy, % (higher is better)")}


def table(number, results=None):
    results = results or ROOT / f"ae/table{number}/results"
    metric, key, title = TABLES[number]
    paper = {row["method"]: row for row in csv.DictReader(
        (ROOT / f"ae/table{number}/data/paper.csv").open())}
    lines = [title, "", "| Method | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for method in METHODS:
        reference = paper.get("ours" if method == "ours_kernel" else method, {})
        cells = []
        for model in MODELS:
            path = results / model / f"{method}-{metric}.json"
            result = json.loads(path.read_text()) if path.exists() else None
            value = f"{result[key]:.2f}" if result else "-"
            if result and result.get("limit"):
                value += "*"
            cells.append(f"{value} ({reference.get(model, '-')})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "(paper value); * short run with LIMIT; ours_kernel is compared with Ours."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="?", type=Path, help="default: ae/table<N>/results")
    p.add_argument("--table", type=int, choices=TABLES)
    args = p.parse_args()
    for number in [args.table] if args.table else TABLES:
        text = table(number, args.results)
        print(text)
        if args.table:
            out = ROOT / f"ae/table{number}/results/table{number}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)


if __name__ == "__main__":
    main()
