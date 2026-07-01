from __future__ import annotations


def _missing_torch(*_args, **_kwargs):
    raise ModuleNotFoundError("torch is required for model/training operations in spirecomm.ai")


try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch import nn  # type: ignore
except ModuleNotFoundError:
    torch = None
    F = None

    class _NNProxy:
        class Module:
            pass

        def __getattr__(self, _name):
            return _missing_torch

    nn = _NNProxy()


def require_torch():
    if torch is None:
        raise ModuleNotFoundError("torch is required for model/training operations in spirecomm.ai")
    return torch


__all__ = ["F", "nn", "require_torch", "torch"]
