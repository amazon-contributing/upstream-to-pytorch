# Owner(s): ["oncall: distributed"]

"""Shared collection-time GPU requirements for distributed tests."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pytest
    from _pytest.config import Config


def _safe_obj(item: Any) -> object | None:
    # item.obj imports the test module, so any exception is possible here.
    try:
        return item.obj
    except Exception:
        return None


def _decorator_gpu_requirement(func: object | None) -> int:
    """Highest accelerator requirement stamped on ``func`` (or anything in its
    ``__wrapped__`` chain) by ``skip_if_lt_x_gpu`` (``_min_gpus_required``) or
    ``requires_world_size`` (``_required_world_size``). 0 when unstamped."""
    best = 0
    while func is not None:
        for attr in ("_min_gpus_required", "_required_world_size"):
            try:
                best = max(best, int(getattr(func, attr)))  # type: ignore[arg-type]
            except (AttributeError, TypeError, ValueError):
                # Missing or invalid stamps do not contribute a GPU requirement.
                pass
        func = getattr(func, "__wrapped__", None)
    return best


def _resolve_gpu_requirement(item: Any) -> int:
    """Accelerators a distributed test declares it needs, read at collection from
    its decorator stamps (see _decorator_gpu_requirement). 0 when undeclared.

    Class world_size is not consulted. On a 4-GPU runner a literal 4 is
    indistinguishable from capacity-scaled NUM_DEVICES/DEVICE_COUNT values that
    still run on 2 GPUs.
    """
    return _decorator_gpu_requirement(_safe_obj(item))


class MultiGpuMinFilterPlugin:
    """Deselect auto-`multigpu`-marked tests whose resolved accelerator
    requirement is below ``--multigpu-min-gpus`` (see _resolve_gpu_requirement),
    so a larger-runner config can run just the >N-GPU distributed tests on top of
    the existing `-m multigpu` selection. Only `multigpu`-marked items are
    considered, and the plugin is registered only when the option is set, so runs
    without it (e.g. every CUDA distributed job) are untouched."""

    def __init__(self, min_gpus: int) -> None:
        self.min_gpus = min_gpus

    def pytest_collection_modifyitems(self, config: Config, items: list[Any]) -> None:
        selected, deselected = [], []
        for item in items:
            is_multigpu = any(m.name == "multigpu" for m in item.iter_markers())
            if is_multigpu and _resolve_gpu_requirement(item) < self.min_gpus:
                deselected.append(item)
            else:
                selected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        selected = len(session.items)
        if session.config.getoption("verbose") >= 0:
            print(
                f"multigpu-min-gpus={self.min_gpus}: final selected {selected}",
                flush=True,
            )
        count_file = os.getenv("PYTORCH_MULTIGPU_SELECTION_COUNT_FILE")
        if count_file:
            with open(count_file, "a") as fp:
                fp.write(f"{selected}\n")
