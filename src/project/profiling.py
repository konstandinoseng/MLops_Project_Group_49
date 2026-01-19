from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


@dataclass(frozen=True)
class ProfilerConfig:
    enabled: bool = False
    # Output folder root
    log_root: str = "log"
    # Subfolder name (e.g., "train", "inference", "data")
    run_name: str = "run"
    # CPU only by default; add CUDA if available and desired
    use_cuda: bool = False

    # What to record
    record_shapes: bool = True
    profile_memory: bool = True  # <- this is the argument you asked about
    with_stack: bool = False  # optional: adds call stacks (more overhead)

    # Iterations: if you profile a loop, call prof.step() each iteration
    steps: int = 1

    # Console table config
    row_limit: int = 20
    sort_by_time: str = "cpu_time_total"
    sort_by_mem: str = "self_cpu_memory_usage"


def _activities(use_cuda: bool) -> list[ProfilerActivity]:
    acts = [ProfilerActivity.CPU]
    if use_cuda and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _print_tables(prof, cfg: ProfilerConfig) -> None:
    print("\n=== PyTorch profiler: CPU time (key_averages) ===")
    print(prof.key_averages().table(sort_by=cfg.sort_by_time, row_limit=cfg.row_limit))

    print("\n=== PyTorch profiler: grouped by input shape (CPU time) ===")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=cfg.sort_by_time, row_limit=cfg.row_limit))

    if cfg.profile_memory:
        print("\n=== PyTorch profiler: memory (key_averages) ===")
        print(prof.key_averages().table(sort_by=cfg.sort_by_mem, row_limit=cfg.row_limit))


@contextmanager
def torch_profile(cfg: ProfilerConfig):
    """
    Context manager that:
    - runs torch.profiler.profile with memory + shapes
    - writes TensorBoard trace files into ./log/<run_name>/
    - exports a chrome trace trace.json
    - prints tables like in the DTU exercise

    Usage:
        with torch_profile(cfg) as prof:
            for i in range(cfg.steps):
                ... workload ...
                prof.step()
    """
    if not cfg.enabled:
        yield None
        return

    out_dir = Path(cfg.log_root) / cfg.run_name
    _ensure_dir(out_dir)

    # TensorBoard trace handler writes *.pt.trace.json
    on_trace = tensorboard_trace_handler(str(out_dir))

    with profile(
        activities=_activities(cfg.use_cuda),
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        on_trace_ready=on_trace,
    ) as prof:
        yield prof

    # Export chrome trace (single file) for chrome://tracing
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    # Print tables to console
    _print_tables(prof, cfg)


def config_from_env(default_run_name: str, steps: int = 1) -> ProfilerConfig:
    """
    Enable profiler via env var:
      TORCH_PROFILER=1
      TORCH_PROFILER_RUN=inference
      TORCH_PROFILER_CUDA=1
    """
    enabled = os.getenv("TORCH_PROFILER", "0") == "1"
    run_name = os.getenv("TORCH_PROFILER_RUN", default_run_name)
    use_cuda = os.getenv("TORCH_PROFILER_CUDA", "0") == "1"

    return ProfilerConfig(
        enabled=enabled,
        run_name=run_name,
        use_cuda=use_cuda,
        steps=steps,
    )
