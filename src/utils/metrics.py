from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass
class MetricTracker:
    """
    Tracks scalar metrics across steps and provides aggregated statistics.
    """

    name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self.values.append(value)
        self.timestamps.append(time.time())

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {f"{self.name}_count": 0}
        arr = np.asarray(self.values, dtype=np.float32)
        return {
            f"{self.name}_count": float(arr.size),
            f"{self.name}_mean": float(arr.mean()),
            f"{self.name}_std": float(arr.std(ddof=0)),
            f"{self.name}_min": float(arr.min()),
            f"{self.name}_max": float(arr.max()),
        }


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """
    Compute classification accuracy.
    """

    if len(predictions) == 0:
        return 0.0
    if len(predictions) != len(labels):
        raise ValueError("Prediction and label sequences must have identical length.")
    correct = sum(int(p == y) for p, y in zip(predictions, labels))
    return correct / len(predictions)


def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    """

    return float(math.exp(loss))


def throughput(num_tokens: int, duration_seconds: float) -> float:
    """
    Compute token throughput (tokens/sec).
    """

    if duration_seconds <= 0:
        return 0.0
    return num_tokens / duration_seconds


def latency_quantiles(latencies: Iterable[float], quantiles: Iterable[float]) -> Dict[str, float]:
    """
    Compute latency quantiles for a sequence of latencies.
    """

    latency_values = np.array(list(latencies), dtype=np.float32)
    if latency_values.size == 0:
        return {}
    return {
        f"latency_p{int(q * 100)}": float(np.quantile(latency_values, q)) for q in quantiles
    }


def summarize_metrics(tracker_map: Mapping[str, MetricTracker]) -> Dict[str, float]:
    """
    Compact helper to aggregate multiple MetricTracker summaries.
    """

    summary: Dict[str, float] = {}
    for tracker in tracker_map.values():
        summary.update(tracker.summary())
    return summary
