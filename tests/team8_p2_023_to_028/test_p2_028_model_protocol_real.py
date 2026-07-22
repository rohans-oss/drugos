"""P2-028 regression tests: model_protocol.py Protocols match real APIs.

Root fix: ``model_protocol.py`` now defines TWO Protocols:
  1. ``KGEmbeddingModel`` -- for homogeneous KGE models (TransE).
     Matches TransEModel's real API.
  2. ``DrugRepurposingModel`` -- for heterogeneous graph models (HGT).
     Matches DrugRepurposingGraphTransformer's real API.

The previous single ``KGEmbeddingModel`` Protocol was ASPIRATIONAL --
it claimed DrugRepurposingGraphTransformer conformed but did not
(GraphTransformer doesn't expose ``entity_embeddings`` /
``relation_embeddings`` / ``normalize_entity_embeddings`` /
``num_total_entities``).
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PHASE2_ROOT = os.path.join(_REPO_ROOT, "phase2")
if _PHASE2_ROOT not in sys.path:
    sys.path.insert(0, _PHASE2_ROOT)


def test_p2_028_both_protocols_are_defined():
    """P2-028: both ``KGEmbeddingModel`` and ``DrugRepurposingModel``
    MUST be defined in model_protocol.py and exported in __all__."""
    from drugos_graph.model_protocol import (
        KGEmbeddingModel,
        DrugRepurposingModel,
    )
    import drugos_graph.model_protocol as mp
    assert "KGEmbeddingModel" in mp.__all__
    assert "DrugRepurposingModel" in mp.__all__, (
        "P2-028 REGRESSION: DrugRepurposingModel must be in __all__"
    )


def test_p2_028_both_protocols_are_runtime_checkable():
    """P2-028: both Protocols MUST be ``@runtime_checkable`` so
    ``isinstance(model, Protocol)`` works at runtime."""
    from drugos_graph.model_protocol import (
        KGEmbeddingModel,
        DrugRepurposingModel,
    )
    # @runtime_checkable adds a ``_is_runtime_protocol`` attribute
    # (CPython implementation detail). The robust way to check is to
    # verify isinstance doesn't raise on a non-instance.
    class Dummy:
        pass
    # If the Protocol is NOT runtime_checkable, this raises TypeError.
    # If it IS runtime_checkable, this returns False (Dummy is not a
    # KGEmbeddingModel).
    assert isinstance(Dummy(), KGEmbeddingModel) is False
    assert isinstance(Dummy(), DrugRepurposingModel) is False


def test_p2_028_drug_repurposing_model_protocol_has_correct_methods():
    """P2-028: ``DrugRepurposingModel`` MUST declare the methods that
    ``DrugRepurposingGraphTransformer`` actually implements:
    ``forward``, ``forward_logits``, ``score_direction``, ``save``,
    ``load``."""
    from drugos_graph.model_protocol import DrugRepurposingModel
    # Protocol methods are stored in __protocol_attrs__ (CPython 3.12+)
    # or as annotations on the class. We check via dir().
    members = set(dir(DrugRepurposingModel))
    expected = {"forward", "forward_logits", "score_direction", "save", "load"}
    missing = expected - members
    assert not missing, (
        f"P2-028: DrugRepurposingModel must declare: {expected}. "
        f"Missing: {missing}"
    )


def test_p2_028_kg_embedding_model_protocol_keeps_original_methods():
    """P2-028: ``KGEmbeddingModel`` MUST still declare its original
    methods (``entity_embeddings``, ``relation_embeddings``,
    ``forward``, ``normalize_entity_embeddings``, ``score_direction``,
    ``num_total_entities``) -- the P2-028 fix is ADDITIVE (adds a new
    Protocol), not destructive."""
    from drugos_graph.model_protocol import KGEmbeddingModel
    members = set(dir(KGEmbeddingModel))
    expected = {
        "entity_embeddings",
        "relation_embeddings",
        "forward",
        "normalize_entity_embeddings",
        "score_direction",
        "num_total_entities",
    }
    missing = expected - members
    assert not missing, (
        f"P2-028: KGEmbeddingModel must still declare: {expected}. "
        f"Missing: {missing}. The P2-028 fix added a NEW Protocol; "
        f"it must NOT remove methods from the existing one."
    )


def test_p2_028_transe_model_satisfies_kg_embedding_model_protocol():
    """P2-028: ``TransEModel`` MUST satisfy ``KGEmbeddingModel``.

    NOTE on isinstance: ``runtime_checkable`` Protocols with properties
    (non-method members like ``entity_embeddings``, ``score_direction``)
    do NOT work reliably with ``isinstance`` in Python 3.12+ -- this is
    a documented Python limitation (see
    https://docs.python.org/3/library/typing.html#typing.runtime_checkable).
    The test uses explicit ``hasattr`` checks instead, which is the
    approach recommended by the Python docs for Protocols with
    properties. This is a STRONGER check than isinstance (which would
    silently return False due to the Python limitation)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available -- cannot construct TransEModel")
    from drugos_graph.transe_model import TransEModel
    from drugos_graph.model_protocol import KGEmbeddingModel
    model = TransEModel(num_entities=10, num_relations=3, embedding_dim=8)

    # Explicit hasattr checks for EVERY required Protocol member.
    required = [
        "entity_embeddings",
        "relation_embeddings",
        "forward",
        "normalize_entity_embeddings",
        "score_direction",
        "num_total_entities",
    ]
    missing = [name for name in required if not hasattr(model, name)]
    assert not missing, (
        f"P2-028 REGRESSION: TransEModel is missing required "
        f"KGEmbeddingModel members: {missing}. The Protocol must match "
        f"TransEModel's real API."
    )
    # Verify the score_direction value
    assert model.score_direction in ("lower_better", "higher_better"), (
        f"P2-028: TransEModel.score_direction returned "
        f"{model.score_direction!r}, must be 'lower_better' or "
        f"'higher_better'."
    )
    assert model.score_direction == "lower_better", (
        f"P2-028: TransE uses L1 distance ||h+r-t|| (lower=more "
        f"plausible), so score_direction must be 'lower_better'. "
        f"Got: {model.score_direction!r}"
    )
    # Verify forward is callable
    assert callable(model.forward), "P2-028: TransEModel.forward must be callable"
    assert callable(model.normalize_entity_embeddings), (
        "P2-028: TransEModel.normalize_entity_embeddings must be callable"
    )


def test_p2_028_graph_transformer_satisfies_drug_repurposing_model_protocol():
    """P2-028: ``DrugRepurposingGraphTransformer`` MUST satisfy
    ``DrugRepurposingModel``.

    This proves the new Protocol matches the GraphTransformer's real API
    -- the central P2-028 fix.

    NOTE on isinstance: see the TransE test above -- ``runtime_checkable``
    Protocols with properties don't work reliably with isinstance. We
    use explicit ``hasattr`` checks (the Python-recommended approach)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available -- cannot construct GraphTransformer")
    # Insert graph_transformer/ onto sys.path so we can import the model
    _GT_ROOT = os.path.join(_REPO_ROOT, "graph_transformer")
    if _GT_ROOT not in sys.path:
        sys.path.insert(0, _GT_ROOT)
    from graph_transformer.models.graph_transformer import (
        DrugRepurposingGraphTransformer,
        GraphTransformerModel,
    )
    from drugos_graph.model_protocol import (
        DrugRepurposingModel,
        KGEmbeddingModel,
    )

    # Construct a minimal model. Use the CANONICAL edge_types and
    # node_types from graph_transformer.data so the test is resilient
    # to future schema changes (e.g. P3-001 v104 bumped the minimum
    # from 14 to 18 edge types — using DEFAULT_EDGE_TYPES means this
    # test automatically tracks the canonical schema).
    from graph_transformer.data import DEFAULT_EDGE_TYPES, DEFAULT_NODE_TYPES
    edge_types = list(DEFAULT_EDGE_TYPES)
    node_types = list(DEFAULT_NODE_TYPES)
    feature_dims = {nt: 8 for nt in node_types}

    model = DrugRepurposingGraphTransformer(
        feature_dims=feature_dims,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        edge_types=edge_types,
        node_types=node_types,
    )

    # The P2-028 central assertion: GraphTransformer has ALL required
    # DrugRepurposingModel members.
    required = [
        "forward",
        "forward_logits",
        "score_direction",
        "save",
        "load",
    ]
    missing = [name for name in required if not hasattr(model, name)]
    assert not missing, (
        f"P2-028 REGRESSION: DrugRepurposingGraphTransformer is missing "
        f"required DrugRepurposingModel members: {missing}. The Protocol "
        f"must match the GraphTransformer's real API -- the central "
        f"P2-028 bug."
    )
    # Verify the score_direction value
    assert model.score_direction == "higher_better", (
        f"P2-028: GraphTransformer.score_direction must be "
        f"'higher_better' (sigmoid output, higher=more plausible). "
        f"Got: {model.score_direction!r}"
    )
    # Verify forward methods are callable
    assert callable(model.forward), "P2-028: GraphTransformer.forward must be callable"
    assert callable(model.forward_logits), (
        "P2-028: GraphTransformer.forward_logits must be callable"
    )
    assert callable(model.save), "P2-028: GraphTransformer.save must be callable"
    # load is a classmethod -- check via the class
    assert callable(getattr(DrugRepurposingGraphTransformer, "load")), (
        "P2-028: DrugRepurposingGraphTransformer.load classmethod must be callable"
    )

    # The P2-028 SECONDARY assertion: GraphTransformer does NOT have the
    # KGEmbeddingModel-specific members (entity_embeddings,
    # relation_embeddings, normalize_entity_embeddings,
    # num_total_entities). This proves the OLD Protocol was aspirational
    # -- the central P2-028 finding.
    kge_only_members = [
        "entity_embeddings",
        "relation_embeddings",
        "normalize_entity_embeddings",
        "num_total_entities",
    ]
    present_kge_only = [m for m in kge_only_members if hasattr(model, m)]
    assert not present_kge_only, (
        "P2-028: DrugRepurposingGraphTransformer must NOT expose "
        f"KGEmbeddingModel-specific members: {present_kge_only}. These "
        "are TransE-specific attributes. The old single-Protocol "
        "design was aspirational -- it claimed GraphTransformer "
        "conformed to KGEmbeddingModel but GraphTransformer does NOT "
        "have these attributes. This is the central P2-028 finding."
    )


def test_p2_028_protocol_module_docstring_documents_p2_028():
    """P2-028: the module docstring MUST document the P2-028 root fix
    so maintainers understand why there are two Protocols."""
    import drugos_graph.model_protocol as mp
    docstring = mp.__doc__ or ""
    assert "P2-028" in docstring, (
        "P2-028: model_protocol.py module docstring must reference "
        "P2-028 for grep-ability and audit trail."
    )
    assert "aspirational" in docstring.lower(), (
        "P2-028: docstring must explain that the old Protocol was "
        "aspirational (the central finding)."
    )
    assert "DrugRepurposingModel" in docstring, (
        "P2-028: docstring must mention the new DrugRepurposingModel Protocol"
    )
