"""Load the pretrained TabFM network — resilient to the safetensors skew.

WHY THIS FILE EXISTS (plumbing, not a learning TODO)
    tabfm 1.0.0 (the only PyPI release) ships a loader that looks for
    ``<model_type>/pytorch_model.bin`` and reads it with ``torch.load``. But
    Google re-uploaded the HuggingFace weights in the newer **safetensors**
    format (``<model_type>/model.safetensors``), so the official
    ``tabfm_v1_0_0_pytorch.load()`` raises FileNotFoundError. This is a
    library-vs-weights version skew, not anything you did.

    ``load_core()`` below first tries the official loader (so if a future
    tabfm release fixes the filename, we transparently use it), and falls
    back to building the model and loading the safetensors state dict by
    hand. The manual path mirrors what the official ``load()`` does — build
    ``TabFM(**RegressionConfig().to_dict())``, load the state dict, ``eval()``
    — it just reads the format that's actually on disk.

    Verified: the safetensors keys match the model's state_dict exactly
    (0 missing / 0 unexpected under strict=True).

USAGE
    from src.research.tabfm._weights import load_core
    from tabfm import TabFMRegressor
    core = load_core("regression")
    reg = TabFMRegressor(model=core, random_state=0)
"""
from __future__ import annotations

_HF_REPO_ID = "google/tabfm-1.0.0-pytorch"


def load_core(model_type: str = "regression"):
    """Return the pretrained TabFM torch module for ``model_type``.

    ``model_type`` is ``"regression"`` or ``"classification"``. The result is
    the raw network you pass as ``TabFMRegressor(model=...)`` /
    ``TabFMClassifier(model=...)``. Weights download from HuggingFace on first
    call (~13 GB, cached outside the repo under ~/.cache/huggingface); later
    calls are fast.
    """
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1

    # Path 1: the official loader. Works if a future release fixes the filename.
    try:
        return tabfm_v1.load(model_type=model_type)
    except FileNotFoundError:
        pass  # expected today — fall through to the safetensors path.

    # Path 2: build the model and load the safetensors weights directly.
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tabfm.src.pytorch.model import TabFM
    from tabfm.src.pytorch.tabfm_v1_0_0 import (
        ClassificationConfig,
        RegressionConfig,
    )

    config = RegressionConfig() if model_type == "regression" else ClassificationConfig()
    core = TabFM(**config.to_dict())

    base = snapshot_download(_HF_REPO_ID)  # already cached -> near-instant
    state_dict = load_file(f"{base}/{model_type}/model.safetensors")
    core.load_state_dict(state_dict, strict=True)  # strict: keys must match exactly
    core.eval()
    return core
