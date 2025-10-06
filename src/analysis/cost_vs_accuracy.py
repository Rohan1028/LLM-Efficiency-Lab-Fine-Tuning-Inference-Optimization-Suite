from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from ..utils.logger import get_logger, setup_logger
from ..utils.visualizer import plot_cost_vs_accuracy


logger = get_logger("analysis.cost_vs_accuracy")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cost vs accuracy metrics.")
    parser.add_argument(
        "--reports",
        type=str,
        nargs="+",
        required=True,
        help="List of JSON metric files generated during experiments.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis/cost_vs_accuracy.html",
        help="Path to save the HTML visualization.",
    )
    return parser.parse_args(args=args)


def load_reports(report_paths: List[str]) -> pd.DataFrame:
    records = []
    for path in report_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "results" in data:
            for entry in data["results"]:
                records.append(
                    {
                        "method": entry.get("method") or data.get("method") or Path(path).stem,
                        "accuracy": entry.get("accuracy", entry.get("eval_accuracy", 0.0)),
                        "cost_usd": entry.get("cost_usd", 0.0),
                        "gpu_memory_gb": entry.get("gpu_memory_gb", 0.0),
                        "runtime_min": entry.get("runtime_min", 0.0),
                    }
                )
        elif isinstance(data, list):
            records.extend(data)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No records found in provided reports.")
    return df


def main(cli_args: argparse.Namespace | None = None) -> None:
    args = cli_args or parse_args()
    setup_logger()
    df = load_reports(args.reports)
    output_path = Path(args.output)
    artifact = plot_cost_vs_accuracy(df.to_dict(orient="records"), output_path=output_path)
    logger.info("Cost vs accuracy visualization saved to %s", artifact.path)
    print(f"Saved visualization to {artifact.path}")


if __name__ == "__main__":
    main()
