from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from .logger import get_logger


logger = get_logger("visualizer")


@dataclass
class PlotArtifact:
    """
    Encapsulates information about generated plots for reuse in reports.
    """

    path: Path
    description: str


def plot_cost_vs_accuracy(
    records: Iterable[Dict[str, float]],
    output_path: Path,
    title: str = "Cost vs. Accuracy",
) -> PlotArtifact:
    """
    Create a scatter plot showing cost versus accuracy.
    """

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Records must contain at least one row to plot.")

    fig = px.scatter(
        df,
        x="cost_usd",
        y="accuracy",
        color="method",
        size="gpu_memory_gb",
        hover_data=["runtime_min"],
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Saved cost vs accuracy plot to %s", output_path)
    return PlotArtifact(path=output_path, description=title)


def plot_memory_tradeoff(
    methods: List[str], memory_gb: List[float], accuracy: List[float], output_path: Path
) -> PlotArtifact:
    """
    Generate a dual-axis bar/line chart comparing memory footprint and accuracy.
    """

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(methods, memory_gb, color="#1f77b4", alpha=0.7, label="GPU Memory (GB)")
    ax1.set_xlabel("Method")
    ax1.set_ylabel("GPU Memory (GB)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(methods, accuracy, color="#ff7f0e", marker="o", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    plt.title("Accuracy vs GPU Memory")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved memory trade-off plot to %s", output_path)
    return PlotArtifact(path=output_path, description="Accuracy vs GPU Memory")
