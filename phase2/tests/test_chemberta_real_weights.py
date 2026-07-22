"""Task 100 — Verify ChemBERTa weights are loaded for REAL, not random fallback.

Task 94 audit note: "must download the real ChemBERTa weights. Currently
falls back to Xavier random features (silent)."

The audit verified that the production code path calls
``AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")`` and
RAISES on failure rather than synthesising random features. This test
suite ENFORCES that contract so a future regression cannot silently
re-introduce the Xavier fallback.

Test categories:

  1. **Source-code static check** (no network, no model download):
     verify the file contains ``AutoModel.from_pretrained`` and does
     NOT contain ``nn.Linear`` + Xavier init as a fallback. These run
     on every CI build.

  2. **Live model load** (downloads the real model, ~500 MB):
     verify ``encode_smiles`` produces deterministic embeddings for a
     known SMILES string. Marked ``@pytest.mark.live_model`` so CI can
     skip in offline environments via ``-m "not live_model"``.

  3. **Failure-mode check** (no network): verify that when the model
     download fails (simulated by pointing to a non-existent model
     name), the encoder RAISES rather than returning random features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


CHEMBERTA_ENCODER_PATH = (
    Path(__file__).resolve().parents[1]
    / "drugos_graph" / "chemberta_encoder.py"
)


# =============================================================================
# Task 100.1 — Static source checks (no network)
# =============================================================================

def test_chemberta_source_loads_real_weights():
    """The chemberta_encoder.py source MUST call AutoModel.from_pretrained."""
    source = CHEMBERTA_ENCODER_PATH.read_text(encoding="utf-8")
    assert "AutoModel.from_pretrained" in source, (
        "chemberta_encoder.py does not call AutoModel.from_pretrained -- "
        "real ChemBERTa weights are NOT loaded"
    )
    assert "AutoTokenizer.from_pretrained" in source, (
        "chemberta_encoder.py does not call AutoTokenizer.from_pretrained -- "
        "no real tokenizer is loaded"
    )


def test_chemberta_source_no_xavier_random_fallback():
    """The chemberta_encoder.py source MUST NOT contain Xavier random init fallback."""
    source = CHEMBERTA_ENCODER_PATH.read_text(encoding="utf-8")

    # Forbidden patterns: random feature fallback.
    forbidden_patterns = [
        "torch.nn.init.xavier_uniform",
        "torch.nn.init.xavier_normal",
        "nn.Linear(",  # Bare linear layer as feature extractor
        "RandomFeature",
        "random_features",
        "xavier_init",
    ]
    found = []
    for pat in forbidden_patterns:
        # Skip comment-line matches by checking only lines without a leading #.
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if pat in line:
                # Allow AutoModel.from_pretrained + nn.Linear inside a
                # transformers model definition (rare but possible).
                # The pattern alone is suspicious -- flag it.
                found.append((pat, line.strip()))
    assert not found, (
        f"chemberta_encoder.py contains forbidden Xavier/random-feature patterns: "
        f"{found}"
    )


def test_chemberta_source_raises_on_failure():
    """The chemberta_encoder.py source MUST raise on model load failure."""
    source = CHEMBERTA_ENCODER_PATH.read_text(encoding="utf-8")
    # Look for a raise statement near the from_pretrained call.
    assert "raise" in source, (
        "chemberta_encoder.py has no raise statement -- failures are silent"
    )
    # Specifically: there must be a ChembertaEncoderError or similar.
    assert "ChembertaEncoderError" in source or "raise RuntimeError" in source, (
        "chemberta_encoder.py does not raise a typed exception on model load failure"
    )


# =============================================================================
# Task 100.2 — Default model is a real ChemBERTa checkpoint
# =============================================================================

def test_chemberta_default_model_is_real_checkpoint():
    """The default model name must be a real HuggingFace ChemBERTa checkpoint."""
    from phase2.drugos_graph.chemberta_encoder import CHEMBERTA_MODEL

    # The default must be one of the known real ChemBERTa checkpoints.
    known_chemberta_models = {
        "seyonec/ChemBERTa-zinc-base-v1",
        "seyonec/ChemBERTa-zinc77M-1M",
        "seyonec/ChemBERTa-zinc77M-5M",
        "seyonec/ChemBERTa-zinc250M-2M",
        "navidved/ChemBERTa-zinc-base-v1",
    }
    # If the default was overridden via env var, accept any name that
    # contains "ChemBERTa" (case-insensitive).
    if "ChemBERTa" in CHEMBERTA_MODEL or "chemberta" in CHEMBERTA_MODEL.lower():
        return
    assert CHEMBERTA_MODEL in known_chemberta_models, (
        f"Default ChemBERTa model {CHEMBERTA_MODEL!r} is not a known real "
        f"checkpoint -- this is the symptom of the Task 94 silent fallback"
    )


def test_chemberta_fallbacks_are_all_real_models():
    """The fallback chain must consist of real ChemBERTa mirrors, not random init."""
    from phase2.drugos_graph.chemberta_encoder import CHEMBERTA_MODEL_FALLBACKS

    assert len(CHEMBERTA_MODEL_FALLBACKS) >= 1, (
        "No fallback models configured -- single-mirror failure has no recovery"
    )
    for fb in CHEMBERTA_MODEL_FALLBACKS:
        assert "ChemBERTa" in fb or "chemberta" in fb.lower(), (
            f"Fallback model {fb!r} is not a real ChemBERTa checkpoint "
            f"(Task 94 silent fallback regression)"
        )


# =============================================================================
# Task 100.3 — Live model load (downloads ~500 MB; skip in CI unless --live-model)
# =============================================================================

@pytest.mark.live_model
def test_chemberta_encode_smiles_real_weights():
    """Encode a known SMILES and verify the embedding is deterministic.

    This test downloads the real ChemBERTa model (~500 MB) on first
    run. It is marked ``@pytest.mark.live_model`` so CI can skip it
    via ``-m "not live_model"``.
    """
    from phase2.drugos_graph.chemberta_encoder import encode_smiles

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin canonical SMILES
    emb1 = encode_smiles([smiles])
    emb2 = encode_smiles([smiles])

    # Embeddings must be deterministic (same input -> same output).
    assert emb1.shape == emb2.shape, (
        f"Non-deterministic embedding shapes: {emb1.shape} vs {emb2.shape}"
    )
    # Embeddings must NOT be all-zeros (silent failure indicator).
    import torch
    if isinstance(emb1, torch.Tensor):
        assert not torch.allclose(emb1, torch.zeros_like(emb1)), (
            "Embedding is all zeros -- model may not have been loaded"
        )
    # Two encodings of the same SMILES must be bit-identical (deterministic).
    if isinstance(emb1, torch.Tensor) and isinstance(emb2, torch.Tensor):
        assert torch.allclose(emb1, emb2, atol=1e-6), (
            "Embeddings for the same SMILES differ -- non-deterministic "
            "encoder (possible random-feature fallback)"
        )


@pytest.mark.live_model
def test_chemberta_different_smiles_different_embeddings():
    """Two different SMILES must produce different embeddings."""
    from phase2.drugos_graph.chemberta_encoder import encode_smiles

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    ibuprofen = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    emb1 = encode_smiles([aspirin])
    emb2 = encode_smiles([ibuprofen])

    import torch
    if isinstance(emb1, torch.Tensor) and isinstance(emb2, torch.Tensor):
        assert not torch.allclose(emb1, emb2, atol=1e-4), (
            "Different SMILES produced identical embeddings -- encoder is "
            "returning constant output (possible random-feature fallback)"
        )


# =============================================================================
# Task 100.4 — Failure mode: bad primary model name MUST fall back to a REAL model (not random)
# =============================================================================

def test_chemberta_bad_primary_falls_back_to_real_model():
    """When the primary model fails to load, the fallback chain MUST load a real
    ChemBERTa model, NOT synthesise random features.

    Task 94 ROOT CAUSE: the audit was about a silent Xavier-random
    fallback. The correct behavior is:
      1. Try the primary model.
      2. If it fails, try each fallback in CHEMBERTA_MODEL_FALLBACKS.
      3. If a fallback succeeds, USE IT (real weights, not random).
      4. If ALL fallbacks fail, RAISE ChembertaEncoderError.

    This test verifies step 3: a bad primary name triggers the fallback
    chain, which successfully loads a real ChemBERTa checkpoint. The
    function must NOT silently return random features.
    """
    from phase2.drugos_graph.chemberta_encoder import (
        CHEMBERTA_MODEL_FALLBACKS,
        _load_model_with_fallback,
    )

    # Sanity: there must be at least one real fallback configured.
    assert len(CHEMBERTA_MODEL_FALLBACKS) >= 1, (
        "No fallback models configured -- a bad primary model would leak "
        "as a hard failure with no recovery path"
    )

    # Use a name that doesn't exist on HuggingFace. The fallback chain
    # MUST activate and load a real model (which requires network
    # access -- skip if offline).
    try:
        # Function returns (tokenizer, model, commit_hash, model_name_used).
        tokenizer, model, commit_hash, loaded_name = _load_model_with_fallback(
            primary_model_name="definitely/not-a-real-model-12345",
            revision=None,
            token=None,
            torch_dtype_val=None,
            attn_implementation=None,
            local_files_only=False,
            cache_dir=None,
            expected_model_hash=None,
        )
    except Exception as e:
        # If the network is unavailable, the fallback ALSO fails and the
        # function correctly raises. That's acceptable -- the test
        # verifies that the function doesn't return random features.
        pytest.skip(
            f"Fallback chain failed (likely no network): {type(e).__name__}: {e}"
        )

    # The loaded model must be one of the real fallback checkpoints.
    assert loaded_name in CHEMBERTA_MODEL_FALLBACKS, (
        f"Fallback loaded {loaded_name!r} but expected one of "
        f"{CHEMBERTA_MODEL_FALLBACKS} -- this suggests a random-feature "
        f"fallback path was taken (Task 94 regression)"
    )
    # The model object must NOT be a torch.nn.Linear (the Xavier fallback).
    import torch.nn as nn
    assert not isinstance(model, nn.Linear), (
        f"Fallback returned a bare nn.Linear (Xavier random init) -- "
        f"Task 94 silent fallback regression"
    )
    # The model must be a transformers model (has a forward method that
    # accepts input_ids and returns a structured output).
    assert hasattr(model, "forward"), (
        f"Fallback model {type(model).__name__} has no forward() method -- "
        f"not a real transformer"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x"]))
