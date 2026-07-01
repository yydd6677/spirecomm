from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import pathlib
from typing import Any, Iterator

from spirecomm.ai.torch_compat import torch


@contextmanager
def _portable_path_unpickle() -> Iterator[None]:
    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath
    try:
        if type(Path()) is pathlib.WindowsPath:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
        elif type(Path()) is pathlib.PosixPath:
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]
        yield
    finally:
        pathlib.PosixPath = original_posix_path  # type: ignore[misc,assignment]
        pathlib.WindowsPath = original_windows_path  # type: ignore[misc,assignment]


def torch_load_portable_path(path: str | Path, *, map_location: str, weights_only: bool = False) -> Any:
    with _portable_path_unpickle():
        return torch.load(Path(path), map_location=map_location, weights_only=weights_only)
