from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Tuple

import torch

from ..utils.logger import get_logger


logger = get_logger("analysis.memory_tracker")


def memory_tracker(label: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Tuple[Any, Dict[str, float]]]]:
    """
    Decorator for measuring GPU memory usage and runtime of a function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Tuple[Any, Dict[str, float]]]:
        tracker_label = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()
            else:
                start_mem = 0.0
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            metrics = {"runtime_s": end_time - start_time}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
                metrics["gpu_memory_gb"] = (peak_mem - start_mem) / 1024**3
            logger.info("%s metrics: %s", tracker_label, metrics)
            return result, metrics

        return wrapper

    return decorator
