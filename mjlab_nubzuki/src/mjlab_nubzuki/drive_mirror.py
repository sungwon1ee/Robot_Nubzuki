"""Mirror a training run directory to a second location (e.g. Google Drive).

Colab sessions die without warning, so the run directory has to exist
somewhere else before that happens. This copies the run's checkpoints, event
files and configs to `NUBZUKI_MIRROR_DIR` every time the runner saves, on a
background thread so a slow Drive write never stalls the training loop.

Enable it by setting the environment variable before launching training:

    NUBZUKI_MIRROR_DIR=/content/drive/MyDrive/nubzuki/runs/walking_v1

The run directory's *contents* are copied into that folder (not nested under
another dated directory), so resuming a run keeps writing into the same place
and TensorBoard can be pointed straight at it.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

ENV_DIR = "NUBZUKI_MIRROR_DIR"
ENV_KEEP = "NUBZUKI_MIRROR_KEEP"  # optional: keep only the newest N model_*.pt


def mirror_dir() -> Path | None:
    value = os.environ.get(ENV_DIR, "").strip()
    return Path(value) if value else None


def _copy(source: Path, destination: Path) -> None:
    """Copy through a temp file so a reader never sees a half-written file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, staging)
    os.replace(staging, destination)


def _is_stale(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return True
    source_stat, destination_stat = source.stat(), destination.stat()
    return (
        source_stat.st_size != destination_stat.st_size
        or source_stat.st_mtime > destination_stat.st_mtime + 1.0
    )


def _prune(destination_root: Path) -> None:
    keep = os.environ.get(ENV_KEEP, "").strip()
    if not keep.isdigit() or int(keep) <= 0:
        return
    checkpoints = sorted(
        destination_root.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for path in checkpoints[: max(0, len(checkpoints) - int(keep))]:
        path.unlink(missing_ok=True)


def _sync(log_dir: Path, destination_root: Path, checkpoint: Path | None) -> None:
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
        # The checkpoint just written comes first: it is the thing worth saving.
        if checkpoint is not None and checkpoint.exists():
            _copy(checkpoint, destination_root / checkpoint.name)
        for source in log_dir.rglob("*"):
            if not source.is_file() or source.name.endswith(".partial"):
                continue
            if checkpoint is not None and source == checkpoint:
                continue
            # Videos are large and reproducible; leave them out of the mirror.
            if "videos" in source.relative_to(log_dir).parts:
                continue
            destination = destination_root / source.relative_to(log_dir)
            if _is_stale(source, destination):
                _copy(source, destination)
        _prune(destination_root)
    except OSError as error:
        # A backup failure must never take the training run down with it.
        print(f"[WARN] Drive mirror failed: {error}")


class RunMirror:
    """One background copier per run; overlapping saves are serialized."""

    def __init__(self, log_dir: str | os.PathLike[str]):
        self.log_dir = Path(log_dir)
        self.destination = mirror_dir()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if self.destination is not None:
            print(f"[INFO] Mirroring run to: {self.destination}")

    @property
    def enabled(self) -> bool:
        return self.destination is not None

    def sync(self, checkpoint: str | os.PathLike[str] | None = None) -> None:
        if self.destination is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return  # A copy is still running; the next save covers this one.
        path = Path(checkpoint) if checkpoint is not None else None

        def work() -> None:
            with self._lock:
                _sync(self.log_dir, self.destination, path)  # type: ignore[arg-type]

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def flush(self, timeout: float = 300.0) -> None:
        """Block until the last copy finishes (call at the end of training)."""
        if self._thread is not None:
            self._thread.join(timeout)
        if self.destination is not None:
            _sync(self.log_dir, self.destination, None)
