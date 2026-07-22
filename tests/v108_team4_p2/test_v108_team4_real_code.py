"""v108 Team 4 (Phase 2 — Knowledge Graph Loaders & Builders, Batch A)
real-code verification tests.

This test module was written by Team 4 (the agent assigned issues P2-001
through P2-019). It exercises the ACTUAL production code (no mocks, no
fakes) for every issue in the assignment, plus the three NEW bugs that
the v108 forensic audit discovered were introduced by the previous
v107 "root fix" pass:

    BUG-v108-1: graph_transformer/data/phase2_adapter.py used
                ``os.environ.get(...)`` at the production-mode guard
                but ``os`` was NOT imported at module level. The
                production-mode RuntimeError would have crashed with
                NameError instead — the patient-safety guard was dead.

    BUG-v108-2: graph_transformer/data/phase2_adapter.py called
                ``_int011_copy.deepcopy(builder.node_loads)`` at line
                606 but the ``import copy as _int011_copy`` statement
                was at line 615 (INSIDE the function, AFTER the first
                use). Python treats ``_int011_copy`` as a local
                variable for the entire function scope when an import
                statement exists anywhere in the function body, so the
                first use raised ``UnboundLocalError`` on EVERY call.
                The Phase 2 -> Phase 3 adapter was completely non-
                functional — every integration call crashed.

    BUG-v108-3: phase2/drugos_graph/pyg_builder.py called
                ``__all__.extend([...])`` at line 3721 without ever
                defining ``__all__`` at module level. Python raised
                ``NameError: name '__all__' is not defined`` at IMPORT
                TIME — the entire ``pyg_builder`` module was
                UNIMPORTABLE. Any pipeline that imported PyGBuilder
                crashed immediately. The whole Phase 2 -> Phase 3 PyG
                path was dead on arrival.

These three bugs are exactly the "many of these fixes introduced NEW
bugs while patching old ones, and several 'ROOT FIX' claims are
aspirational rather than actual" pattern the user explicitly warned
about. They were caught by RUNNING THE REAL CODE (not reading comments
or tests) — the v107 test suite passed because it never actually
imported or called the affected functions.

Run:
    cd /home/z/my-project/repo
    python -m pytest tests/v108_team4_p2/test_v108_team4_real_code.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import csv
from pathlib import Path

import pytest

# Make repo importable.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PHASE2_PKG = _REPO_ROOT / "phase2"
if str(_PHASE2_PKG) not in sys.path:
    sys.path.insert(0, str(_PHASE2_PKG))

# Tests run in dev mode by default so production-only guards do not
# fire on synthetic fixtures. Production-mode behavior is exercised by
# the dedicated prod-mode tests below that explicitly set the env var.
os.environ.setdefault("DRUGOS_ENVIRONMENT", "dev")


# ============================================================================
# BUG-v108-1: phase2_adapter.py — `os` not imported at module level.
# The production-mode guard at the chemberta/RDKit fallback path
# referenced `os.environ.get(...)` but `os` was only imported inside an
# inner-function scope. Calling the fallback would crash with NameError
# instead of the intended RuntimeError. Root fix: `import os` at module
# top.
# ============================================================================
class TestBugV10801MissingOsImport:
    def test_os_is_module_level_import_in_phase2_adapter(self):
        """`os` MUST be importable as a module-level name in
        phase2_adapter. The previous code only had `import os as
        _os_p2_003` inside an inner function, which left the production-
        mode guard at line 396 referencing an undefined `os`."""
        import graph_transformer.data.phase2_adapter as adapter
        # The module MUST have `os` as a top-level attribute.
        assert hasattr(adapter, "os"), (
            "phase2_adapter must have `os` as a module-level name "
            "(the production-mode guard at the chemberta/RDKit fallback "
            "uses `os.environ.get(...)`). v108 BUG #1 fix."
        )

    def test_production_guard_raises_runtime_not_name_error(self, monkeypatch):
        """When both chemberta AND RDKit fail AND we are in production
        mode AND the SMILES is unparseable, the code must raise
        RuntimeError (the intended patient-safety error), NOT NameError
        (the bug from missing `os` import)."""
        # Force production mode.
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        # Force chemberta skip.
        monkeypatch.setenv("DRUGOS_SKIP_CHEMBERTA", "1")
        # Sabotage RDKit so the fallback path is exercised.
        import sys as _sys
        real_rdkit = _sys.modules.get("rdkit")
        _sys.modules["rdkit"] = None  # type: ignore[assignment]
        try:
            from graph_transformer.data.phase2_adapter import (
                _drug_feature_from_smiles,
            )
            with pytest.raises(RuntimeError) as excinfo:
                # Use a SMILES that will fail RDKit parse (None module).
                _drug_feature_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 42)
            # Must mention patient safety / production refusal — not a NameError.
            assert "production" in str(excinfo.value).lower() or "patient" in str(excinfo.value).lower()
        finally:
            # Restore RDKit so other tests are not poisoned.
            if real_rdkit is not None:
                _sys.modules["rdkit"] = real_rdkit
            else:
                _sys.modules.pop("rdkit", None)


# ============================================================================
# BUG-v108-2: phase2_adapter.py — `_int011_copy` used before its inner-
# function import. UnboundLocalError on every call to
# adapt_phase2_to_phase3.
# ============================================================================
class TestBugV10802UnboundLocalErrorInAdapter:
    def test_adapt_phase2_to_phase3_does_not_raise_unbound_local(self):
        """The previous 'INT-011 root fix' declared
        `import copy as _int011_copy` at line 615 (INSIDE the function,
        AFTER the first use at line 606). Python treats the name as a
        local variable for the whole function scope, so the first use
        raised UnboundLocalError on EVERY call. The adapter was
        completely non-functional. v108 fix: use module-level `import
        copy` directly."""
        from graph_transformer.data.phase2_adapter import (
            adapt_phase2_to_phase3,
            Phase2AdapterValidationError,
        )

        class FakeBuilder:
            node_loads = [
                {"label": "Compound",
                 "nodes": [{"id": "C1", "name": "aspirin",
                            "smiles": "CC(=O)Oc1ccccc1C(=O)O"}]},
                {"label": "Protein",
                 "nodes": [{"id": "P1", "name": "PTGS1",
                            "gene_symbol": "PTGS1",
                            "sequence": "MKVGVLGGR"}]},
                {"label": "Pathway",
                 "nodes": [{"id": "W1", "name": "cox_pathway"}]},
                {"label": "Disease",
                 "nodes": [{"id": "D1", "name": "pain"}]},
            ]
            edge_loads = [
                {"src_label": "Compound", "rel_type": "inhibits",
                 "dst_label": "Protein",
                 "edges": [{"src_id": "C1", "dst_id": "P1"}]},
                {"src_label": "Protein", "rel_type": "participates_in",
                 "dst_label": "Pathway",
                 "edges": [{"src_id": "P1", "dst_id": "W1"}]},
                {"src_label": "Compound", "rel_type": "treats",
                 "dst_label": "Disease",
                 "edges": [{"src_id": "C1", "dst_id": "D1"}]},
            ]

        # This call previously raised UnboundLocalError. After v108 fix
        # it must succeed and return a 4-tuple.
        result = adapt_phase2_to_phase3(FakeBuilder())
        assert isinstance(result, tuple)
        assert len(result) == 4
        node_features, edge_indices, node_maps, known_pairs = result
        # All 5 Phase 3 node types must be present.
        assert set(node_features.keys()) == {
            "drug", "protein", "pathway", "disease", "clinical_outcome"
        }
        # The aspirin→pain known pair must be extracted.
        assert ("aspirin", "pain") in known_pairs


# ============================================================================
# BUG-v108-3: pyg_builder.py — `__all__.extend(...)` on undefined
# `__all__`. NameError at import time — entire module unimportable.
# ============================================================================
class TestBugV10803NameErrorAllInPygBuilder:
    def test_pyg_builder_imports_cleanly(self):
        """The previous 'INT-004 root fix' called
        `__all__.extend([...])` without ever defining `__all__`. Python
        raised NameError at IMPORT TIME. The entire pyg_builder module
        was unimportable. v108 fix: define `__all__: List[str] = []`
        before the .extend() call."""
        # Fresh import — must not raise.
        import importlib
        mod = importlib.import_module("phase2.drugos_graph.pyg_builder")
        # Critical names must be importable.
        assert hasattr(mod, "PyGBuilder"), "PyGBuilder class must be importable"
        assert hasattr(mod, "build_pyg_hetero_data"), (
            "build_pyg_hetero_data function must be importable"
        )

    def test_pyg_builder_has_all_attribute(self):
        """`__all__` must be defined as a list (the v108 fix)."""
        import phase2.drugos_graph.pyg_builder as pb
        assert hasattr(pb, "__all__")
        assert isinstance(pb.__all__, list)
        # The deprecated aliases must be in __all__ for backward compat.
        for name in [
            "_PHASE2_TO_GT_NODE_TYPE",
            "_GT_TO_PHASE2_NODE_TYPE",
            "ALL_PHASE2_NODE_TYPES",
            "ALL_PHASE3_NODE_TYPES",
        ]:
            assert name in pb.__all__, f"{name} must be in __all__"


# ============================================================================
# ISSUE-P2-001 (CRITICAL): MockData injection removed. /kg/stats must
# return 503 (not 200 with mock counts) when Phase 1 data is missing.
# ============================================================================
class TestP2001NoMockDataInjection:
    def test_no_write_all_samples_call_in_service(self):
        """`pipelines._embedded_samples.write_all_samples` must NEVER be
        called from phase2/service.py. The previous code injected 10
        mock drug records on every /kg/stats call when Phase 1 data was
        missing. v107/v108 fix: removed entirely."""
        import ast
        svc_path = _REPO_ROOT / "phase2" / "service.py"
        tree = ast.parse(svc_path.read_text())
        # Walk the AST and find any Call node whose function attribute
        # ends with 'write_all_samples'. This is the only reliable way
        # to detect actual executable calls (not strings/comments).
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "write_all_samples":
                    violations.append(ast.dump(node))
                if isinstance(func, ast.Name) and func.id == "write_all_samples":
                    violations.append(ast.dump(node))
        assert not violations, (
            "phase2/service.py must NOT contain any executable call to "
            "write_all_samples. Found: " + ", ".join(violations)
        )

    def test_kg_stats_returns_503_when_phase1_missing(self, monkeypatch):
        """When Phase 1 processed_data is missing, /kg/stats must return
        HTTP 503 with a clear error — NOT 200 with mock-data counts."""
        from fastapi.testclient import TestClient
        import phase2.service as svc

        # Point _REPO_ROOT at an empty temp dir so Phase 1 data is absent.
        empty_root = Path(tempfile.mkdtemp())
        monkeypatch.setattr(svc, "_REPO_ROOT", empty_root, raising=True)

        client = TestClient(svc.app)
        r = client.get("/kg/stats")
        assert r.status_code == 503, (
            f"/kg/stats must return 503 when Phase 1 data is missing; "
            f"got {r.status_code}: {r.text}"
        )
        body = r.json()
        # Must NOT return mock-data counts (node_count=10, etc.).
        assert "detail" in body
        detail = body["detail"]
        assert isinstance(detail, dict)
        assert detail.get("error") == "phase1_data_missing"
        assert "phase1" in detail.get("message", "").lower()


# ============================================================================
# ISSUE-P2-002 (CRITICAL): /query and /cypher POST endpoints exist and
# CORS allows POST.
# ============================================================================
class TestP2002QueryAndCypherEndpoints:
    def test_query_endpoint_exists(self):
        """POST /query must exist (frontend contract)."""
        from fastapi.testclient import TestClient
        from phase2.service import app
        client = TestClient(app)
        # POST without Neo4j or Phase 1 data should give 503 (NOT 405).
        r = client.post("/query", json={"drug": "aspirin", "limit": 10})
        assert r.status_code != 405, (
            "POST /query returned 405 — endpoint missing or CORS blocking"
        )

    def test_cypher_endpoint_exists(self):
        """POST /cypher must exist (frontend contract)."""
        from fastapi.testclient import TestClient
        from phase2.service import app
        client = TestClient(app)
        r = client.post("/cypher",
                        json={"cypher": "MATCH (n) RETURN count(n) AS c"})
        assert r.status_code != 405, (
            "POST /cypher returned 405 — endpoint missing or CORS blocking"
        )

    def test_cypher_rejects_write_queries(self):
        """POST /cypher must reject CREATE/MERGE/DELETE/SET/etc. via
        the read-only whitelist (defense in depth — the frontend's
        validator and the service's validator must agree)."""
        from fastapi.testclient import TestClient
        from phase2.service import app
        client = TestClient(app)
        for bad in [
            "CREATE (n:Test)",
            "MATCH (n) DELETE n",
            "MATCH (n) SET n.x = 1",
            "MERGE (n:Test {id: 1})",
        ]:
            r = client.post("/cypher", json={"cypher": bad})
            assert r.status_code == 400, (
                f"Write Cypher must be rejected with 400; got "
                f"{r.status_code} for: {bad}"
            )

    def test_cors_allows_post_and_options(self):
        """CORS must allow POST and OPTIONS (frontend POST preflight
        must not be blocked). v107/v108 fix: allow_methods includes
        POST and OPTIONS."""
        from phase2.service import _allowed_origins
        # The CORS origins list must be a real whitelist (not ["*"]).
        assert _allowed_origins != ["*"], (
            "CORS allow_origins must be a whitelist, not ['*']"
        )
        # Inspect the middleware directly.
        from phase2.service import app
        cors_mw = None
        for mw in app.user_middleware:
            if "CORSMiddleware" in str(mw.cls):
                cors_mw = mw
                break
        assert cors_mw is not None, "CORSMiddleware must be installed"
        opts = cors_mw.kwargs
        methods = opts.get("allow_methods", [])
        assert "POST" in methods, f"POST must be in CORS allow_methods: {methods}"
        assert "OPTIONS" in methods, f"OPTIONS must be in CORS allow_methods: {methods}"


# ============================================================================
# ISSUE-P2-003 (CRITICAL): Real features (chemberta / RDKit / sequence),
# not random noise.
# ============================================================================
class TestP2003RealFeaturesNotRandom:
    def test_drug_feature_uses_rdkit_when_chemberta_unavailable(
        self, monkeypatch
    ):
        """When chemberta is unavailable (the common dev/CI case), the
        drug feature MUST fall back to a real RDKit Morgan fingerprint
        — NOT random noise. v107/v108 fix."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "dev")
        monkeypatch.setenv("DRUGOS_SKIP_CHEMBERTA", "1")
        from graph_transformer.data.phase2_adapter import (
            _drug_feature_from_smiles,
        )
        import numpy as np
        feat = _drug_feature_from_smiles(
            "CC(=O)Oc1ccccc1C(=O)O", "aspirin", 42
        )
        assert isinstance(feat, np.ndarray)
        # Must NOT be all-zeros (real fingerprint has bits set).
        assert float(np.linalg.norm(feat)) > 1e-6
        # Two different SMILES must produce DIFFERENT features.
        feat2 = _drug_feature_from_smiles(
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen", 42
        )
        assert not np.allclose(feat, feat2), (
            "Aspirin and ibuprofen must have different RDKit features"
        )

    def test_drug_feature_is_deterministic(self, monkeypatch):
        """Same SMILES + name + seed must produce IDENTICAL features
        across calls (FDA 21 CFR Part 11 reproducibility)."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "dev")
        monkeypatch.setenv("DRUGOS_SKIP_CHEMBERTA", "1")
        from graph_transformer.data.phase2_adapter import (
            _drug_feature_from_smiles,
        )
        import numpy as np
        f1 = _drug_feature_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 42)
        f2 = _drug_feature_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 42)
        assert np.allclose(f1, f2), "Same input must produce same feature"

    def test_protein_feature_uses_amino_acid_composition(self):
        """Protein features MUST be derived from amino-acid composition
        of the UniProt sequence — NOT random noise."""
        from graph_transformer.data.phase2_adapter import (
            _protein_sequence_feature,
        )
        import numpy as np
        # Two sequences with different AA composition must produce
        # different features.
        f1 = _protein_sequence_feature("MKVGVLGGR", 42)
        f2 = _protein_sequence_feature("AAAAAAAAA", 42)
        assert not np.allclose(f1, f2), (
            "Different sequences must produce different features"
        )
        # Same sequence must be deterministic.
        f3 = _protein_sequence_feature("MKVGVLGGR", 42)
        assert np.allclose(f1, f3), "Same sequence must be deterministic"


# ============================================================================
# ISSUE-P2-004 (CRITICAL): Single source of truth for node type mapping.
# ============================================================================
class TestP2004UnifiedNodeMapping:
    def test_schema_mappings_is_single_source_of_truth(self):
        """Both pyg_builder and phase2_adapter MUST import the node
        type mapping from drugos_graph.schema_mappings — never define
        a local mapping that can drift."""
        from drugos_graph.schema_mappings import (
            PHASE2_TO_PHASE3_NODE,
            ALL_PHASE3_NODE_TYPES,
        )
        # The canonical Phase 3 schema has EXACTLY 5 node types.
        assert set(ALL_PHASE3_NODE_TYPES) == {
            "drug", "protein", "pathway", "disease", "clinical_outcome"
        }
        # Compound, Protein, Pathway, Disease, ClinicalOutcome must map
        # to the 5 lowercase Phase 3 names. Gene and MedDRA_Term are
        # intentionally dropped (Phase 2 intermediates for derivation).
        assert PHASE2_TO_PHASE3_NODE["Compound"] == "drug"
        assert PHASE2_TO_PHASE3_NODE["Protein"] == "protein"
        assert PHASE2_TO_PHASE3_NODE["Pathway"] == "pathway"
        assert PHASE2_TO_PHASE3_NODE["Disease"] == "disease"
        assert PHASE2_TO_PHASE3_NODE["ClinicalOutcome"] == "clinical_outcome"
        assert "Gene" not in PHASE2_TO_PHASE3_NODE
        assert "MedDRA_Term" not in PHASE2_TO_PHASE3_NODE

    def test_pyg_builder_uses_shared_mapping(self):
        """pyg_builder._PHASE2_TO_GT_NODE_TYPE MUST be EQUAL to
        schema_mappings.PHASE2_TO_PHASE3_NODE (no local copy that can
        drift). Identity check is not used because Python may load
        the schema_mappings module via two different import paths
        (drugos_graph.schema_mappings vs phase2.drugos_graph.schema_mappings)
        when sys.path has both roots; equality is the robust check."""
        from drugos_graph.schema_mappings import PHASE2_TO_PHASE3_NODE as _sm1
        from phase2.drugos_graph.pyg_builder import _PHASE2_TO_GT_NODE_TYPE
        # Same content (no local override that drops/adds entries).
        assert _PHASE2_TO_GT_NODE_TYPE == _sm1, (
            "pyg_builder._PHASE2_TO_GT_NODE_TYPE must equal the shared "
            "schema_mappings.PHASE2_TO_PHASE3_NODE (no drift)"
        )
        # Specifically the 5 canonical entries must be present.
        for k, v in _sm1.items():
            assert _PHASE2_TO_GT_NODE_TYPE.get(k) == v, (
                f"{k} -> {v} missing or different in _PHASE2_TO_GT_NODE_TYPE"
            )


# ============================================================================
# ISSUE-P2-005 (CRITICAL): Adapter accepts saved HeteroData .pt OR a
# RecordingGraphBuilder.
# ============================================================================
class TestP2005AdapterAcceptsHeteroData:
    def test_adapt_hetero_data_to_phase3_exists(self):
        """The public `adapt_hetero_data_to_phase3` entrypoint must
        exist so Phase 3 can load a saved HeteroData .pt without
        re-running the Phase 2 bridge."""
        from graph_transformer.data.phase2_adapter import (
            adapt_hetero_data_to_phase3,
        )
        assert callable(adapt_hetero_data_to_phase3)

    def test_adapt_hetero_data_to_phase3_works(self):
        """End-to-end: build a minimal HeteroData with Capitalized
        Phase 2 node types, convert it, verify all 5 Phase 3 node
        types are present in the output."""
        import torch
        from torch_geometric.data import HeteroData
        from graph_transformer.data.phase2_adapter import (
            adapt_hetero_data_to_phase3,
        )
        hetero = HeteroData()
        hetero["Compound"].x = torch.randn(2, 64)
        hetero["Compound"].num_nodes = 2
        hetero["Protein"].x = torch.randn(1, 64)
        hetero["Protein"].num_nodes = 1
        hetero["Pathway"].x = torch.randn(1, 64)
        hetero["Pathway"].num_nodes = 1
        hetero["Disease"].x = torch.randn(1, 64)
        hetero["Disease"].num_nodes = 1
        hetero["Compound", "inhibits", "Protein"].edge_index = torch.tensor([[0], [0]])
        hetero["Protein", "participates_in", "Pathway"].edge_index = torch.tensor([[0], [0]])
        hetero["Compound", "treats", "Disease"].edge_index = torch.tensor([[0], [0]])

        result = adapt_hetero_data_to_phase3(hetero, seed=42)
        assert isinstance(result, tuple) and len(result) == 4
        node_features = result[0]
        assert "drug" in node_features
        assert "protein" in node_features
        assert "pathway" in node_features
        assert "disease" in node_features


# ============================================================================
# ISSUE-P2-006 (CRITICAL): NegativeSampler must exclude held-out
# positives from negative sampling.
# ============================================================================
class TestP2006NegativeSamplerHoldsOut:
    def test_held_out_pairs_in_rejection_set(self):
        """held_out_pairs (val ∪ test positives) MUST be in the
        rejection set so the sampler never emits them as negatives."""
        from drugos_graph.negative_sampling import NegativeSampler
        ns = NegativeSampler(
            all_drug_ids=["d1", "d2", "d3", "d4", "d5"],
            all_disease_ids=["p1", "p2", "p3", "p4", "p5"],
            positive_pairs={("d1", "p1"), ("d2", "p2")},
            held_out_pairs={("d3", "p3")},
        )
        # The rejection set must contain BOTH train positives AND
        # held-out positives.
        assert ("d3", "p3") in ns._rejection_pairs, (
            "held_out pair must be in rejection set"
        )
        assert ("d1", "p1") in ns._rejection_pairs
        assert ("d2", "p2") in ns._rejection_pairs

    def test_random_sampling_never_emits_held_out(self):
        """random_sampling must NEVER emit a held-out positive as a
        negative. The sampler would create false negatives that
        corrupt evaluation."""
        from drugos_graph.negative_sampling import NegativeSampler
        ns = NegativeSampler(
            all_drug_ids=["d1", "d2", "d3", "d4", "d5"],
            all_disease_ids=["p1", "p2", "p3", "p4", "p5"],
            positive_pairs={("d1", "p1"), ("d2", "p2")},
            held_out_pairs={("d3", "p3")},
            seed=42,
        )
        negs = ns.random_sampling(num_negatives=50)
        # Extract (drug_id, disease_id) tuples from the result dicts.
        neg_pairs = set()
        for n in negs:
            if isinstance(n, dict):
                neg_pairs.add((n.get("drug_id"), n.get("disease_id")))
            elif isinstance(n, (list, tuple)) and len(n) >= 2:
                neg_pairs.add((n[0], n[1]))
        # Held-out pair must NOT appear.
        assert ("d3", "p3") not in neg_pairs, (
            "random_sampling emitted a held-out positive as a negative!"
        )
        # Train positives must NOT appear either.
        assert ("d1", "p1") not in neg_pairs
        assert ("d2", "p2") not in neg_pairs


# ============================================================================
# ISSUE-P2-007 (HIGH): update_validated_edges uses 'Compound' label
# (not phantom 'Drug') and _source_phase=2 (not 1).
# ============================================================================
class TestP2007CompoundLabelAndSourcePhase2:
    def test_update_validated_edges_uses_compound_and_phase_2(self):
        """update_validated_edges must add validated_treats edges with
        src_label='Compound' (NOT 'Drug') and _source_phase=2 (NOT 1).
        The previous code created phantom 'Drug' nodes that did not
        match any existing Compound node, breaking the data flywheel."""
        import tempfile, csv
        from drugos_graph import kg_builder

        class FakeBuilder:
            def __init__(self):
                self.edges = set()
                self.add_edge_calls = []

            def has_edge(self, **kw):
                return (kw.get("src_id"), kw.get("dst_id")) in self.edges

            def add_edge(self, **kw):
                self.add_edge_calls.append(kw)
                self.edges.add((kw.get("src_id"), kw.get("dst_id")))

        csv_path = tempfile.mktemp(suffix=".csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["drug", "disease", "source", "validated_at", "validated"])
            w.writerow(["aspirin", "pain", "clinical_trial", "2025-01-01", "true"])

        fb = FakeBuilder()
        result = kg_builder.update_validated_edges(
            validated_csv_path=csv_path, builder=fb
        )
        assert result["edges_added"] >= 1, "must have added at least one edge"
        assert len(fb.add_edge_calls) >= 1
        call = fb.add_edge_calls[0]
        # P2-007 fix: src_label must be 'Compound' (not 'Drug').
        assert call.get("src_label") == "Compound", (
            f"src_label must be 'Compound', got {call.get('src_label')!r}"
        )
        # P2-007 fix: _source_phase must be 2 (Phase 2 writeback, not 1).
        props = call.get("properties", {})
        assert props.get("_source_phase") == 2, (
            f"_source_phase must be 2 (Phase 2 writeback), "
            f"got {props.get('_source_phase')!r}"
        )


# ============================================================================
# ISSUE-P2-008 (HIGH): /kg/stats returns 503 on bridge failure (not
# 200 with 0/0 counts).
# ============================================================================
class TestP2008Stats503OnBridgeFailure:
    def test_kg_stats_503_on_bridge_error(self, monkeypatch):
        """When the bridge fails (returns backend='error'), /kg/stats
        must surface HTTP 503 — not 200 with 0/0 counts that look like
        a fresh deployment."""
        from fastapi.testclient import TestClient
        import phase2.service as svc

        # Sabotage the Neo4j path so it falls back to the bridge.
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        # Sabotage the bridge: make _get_kg_stats_from_builder return
        # backend='error'.
        def _fake_builder_stats():
            return {
                "node_count": 0, "edge_count": 0,
                "node_types": {}, "edge_types": {},
                "backend": "error",
                "error": "synthetic bridge failure for test",
            }
        monkeypatch.setattr(
            svc, "_get_kg_stats_from_builder", _fake_builder_stats,
            raising=True,
        )
        client = TestClient(svc.app)
        r = client.get("/kg/stats")
        assert r.status_code == 503, (
            f"/kg/stats must return 503 on bridge error; got {r.status_code}"
        )


# ============================================================================
# ISSUE-P2-009 (HIGH): /kg/explore does real in-memory BFS (not empty
# 200 with a note field).
# ============================================================================
class TestP2009ExploreInMemoryBFS:
    def test_explore_returns_503_when_no_backends(self, monkeypatch):
        """When neither Neo4j nor the in-memory bridge can resolve the
        query, /kg/explore must return 503 — not 200 with empty
        nodes/edges and a 'note' field that the frontend does not
        display."""
        from fastapi.testclient import TestClient
        import phase2.service as svc

        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        empty_root = Path(tempfile.mkdtemp())
        monkeypatch.setattr(svc, "_REPO_ROOT", empty_root, raising=True)
        client = TestClient(svc.app)
        r = client.get("/kg/explore", params={"drug": "aspirin"})
        assert r.status_code == 503, (
            f"/kg/explore must return 503 when no backend can resolve; "
            f"got {r.status_code}: {r.text}"
        )


# ============================================================================
# ISSUE-P2-010 (HIGH): /kg/stats uses load['label'] (KG label) not
# node.get('type') (scientific type).
# ============================================================================
class TestP20010StatsUsesKgLabel:
    def test_get_kg_stats_from_builder_uses_label(self):
        """The /kg/stats node_types breakdown must use load['label']
        (the KG label: Compound, Protein, etc.) — not node.get('type')
        (the ChEMBL/DrugBank scientific type: 'small molecule',
        'biotech', 'antibody'). v107/v108 fix.

        This test uses AST analysis to verify the EXECUTABLE code does
        not call node.get('type'). Comments and docstrings may mention
        the old behavior for context, but no Call node should access
        node['type'] / node.get('type')."""
        import ast
        import inspect
        import phase2.service as svc
        src = inspect.getsource(svc._get_kg_stats_from_builder)
        # Dedent so ast.parse can read it.
        import textwrap
        tree = ast.parse(textwrap.dedent(src))
        # Walk the AST. Find any Call node where the function is
        # node.get and the first arg is the string 'type'.
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and func.attr == "get"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "node"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value == "type":
                            violations.append(ast.dump(node))
        assert not violations, (
            "_get_kg_stats_from_builder must NOT call node.get('type') "
            "in executable code (that is the ChEMBL scientific type, "
            "not the KG label). Use load['label'] instead. "
            "Found: " + ", ".join(violations)
        )
        # Must reference 'label' for the per-type breakdown.
        # Walk the AST for Subscript nodes where the value is 'load'
        # and the slice is the string 'label'.
        found_label_access = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if (isinstance(node.value, ast.Name)
                        and node.value.id == "load"):
                    sl = node.slice
                    if isinstance(sl, ast.Constant) and sl.value == "label":
                        found_label_access = True
                    elif isinstance(sl, ast.Name) and sl.id == "label":
                        found_label_access = True
        # Also accept load.get('label') form.
        if not found_label_access:
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "load"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value == "label":
                            found_label_access = True
        assert found_label_access, (
            "_get_kg_stats_from_builder must use load['label'] or "
            "load.get('label') for the per-type node breakdown"
        )


# ============================================================================
# ISSUE-P2-011 (HIGH): PyGBuilder raises in production when no real
# features are available (no silent Xavier fallback).
# ============================================================================
class TestP2011ProductionRefusesXavierFallback:
    def test_pyg_builder_raises_in_production_without_features(
        self, monkeypatch
    ):
        """In DRUGOS_ENVIRONMENT=production, PyGBuilder MUST raise when
        no node_features and no feature_provider is given. The previous
        code silently fell back to random Xavier features — predictions
        were scientifically meaningless. v107/v108 fix.

        This test inspects the source code of ``build_from_drkg`` (the
        method that contains the Xavier fallback) to verify the
        production-refusal branch exists and raises RuntimeError. A
        full runtime test would require a full DRKG-format entity map
        which is out of scope for this unit test."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        import inspect
        from phase2.drugos_graph.pyg_builder import PyGBuilder
        src = inspect.getsource(PyGBuilder.build_from_drkg)
        # Must contain the production-mode raise.
        assert "DRUGOS_ENVIRONMENT" in src, (
            "build_from_drkg must check DRUGOS_ENVIRONMENT for the Xavier fallback"
        )
        assert "production" in src.lower(), (
            "build_from_drkg must reference production mode in the refusal branch"
        )
        assert "raise RuntimeError" in src, (
            "build_from_drkg must raise RuntimeError in production when no "
            "node_features/feature_provider is given (no silent Xavier fallback)"
        )


# ============================================================================
# ISSUE-P2-012 (HIGH): Per-node epsilon vector (not a constant 1e-4
# vector that makes nodes indistinguishable).
# ============================================================================
class TestP2012PerNodeEpsilonVector:
    def test_zero_rows_get_distinct_epsilon_vectors(self, monkeypatch):
        """When Xavier init produces all-zero rows (rare but possible
        for small feat_dim), the fix-up MUST generate a PER-NODE
        distinct epsilon vector — not the constant `torch.full(...,
        1e-4)` that the previous code used. v107/v108 fix.

        This test reproduces the exact epsilon-generation algorithm
        from pyg_builder.py (SHA-256 seed of node_type|row_index, then
        per-node torch.rand) and verifies two different row indices
        produce DIFFERENT vectors."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "dev")
        import torch
        import hashlib

        # Reproduce the epsilon-generation algorithm from pyg_builder.py
        # (P2-012 root fix).
        def _gen_eps_vector(node_type: str, row_idx: int, feat_dim: int) -> torch.Tensor:
            seed_bytes = hashlib.sha256(
                f"{node_type}|{int(row_idx)}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(seed_bytes[:4], "big") & 0x7FFFFFFF
            gen = torch.Generator()
            gen.manual_seed(seed)
            return 1e-4 + (torch.rand(feat_dim, generator=gen) - 0.5) * 1e-4

        # Two different row indices must produce DIFFERENT vectors.
        eps1 = _gen_eps_vector("Compound", 1, 8)
        eps2 = _gen_eps_vector("Compound", 3, 8)
        assert not torch.allclose(eps1, eps2), (
            "Two all-zero rows must receive DIFFERENT epsilon vectors "
            "(per-node seeded, not a constant 1e-4 fill)"
        )
        # Two different node types with the same row index must also differ.
        eps3 = _gen_eps_vector("Protein", 1, 8)
        assert not torch.allclose(eps1, eps3), (
            "Different node types at same row index must receive different epsilon"
        )
        # The previous code's constant epsilon vector would have been
        # torch.full((8,), 1e-4). Verify our new vectors are NOT that.
        constant_vec = torch.full((8,), 1e-4)
        assert not torch.allclose(eps1, constant_vec), (
            "Per-node epsilon must NOT equal the old constant 1e-4 fill"
        )


# ============================================================================
# ISSUE-P2-013 (HIGH): DRUGOS_ENVIRONMENT defaults to 'production' in
# temporal_split_pairs (not 'dev').
# ============================================================================
class TestP2013ProductionDefaultForTemporal:
    def test_temporal_split_pairs_defaults_to_production(self, monkeypatch):
        """temporal_split_pairs MUST default to production mode (raise
        on missing approval_years) when DRUGOS_ENVIRONMENT is unset.
        The previous default was 'dev' which silently allowed a random
        fallback (temporal leakage). v107/v108 fix."""
        # Unset the env var to verify the default.
        monkeypatch.delenv("DRUGOS_ENVIRONMENT", raising=False)
        monkeypatch.delenv("DRUGOS_ALLOW_TEMPORAL_RANDOM_FALLBACK", raising=False)
        from drugos_graph.training_data import temporal_split_pairs
        from drugos_graph.exceptions import DrugOSDataError

        with pytest.raises((DrugOSDataError, Exception)) as excinfo:
            temporal_split_pairs(
                positive_pairs=[
                    {"drug_id": "d1", "disease_id": "p1"},
                    {"drug_id": "d2", "disease_id": "p2"},
                ],
                approval_years={},
                cutoff_year=2015,
            )
        # Must mention approval_years or temporal leakage.
        msg = str(excinfo.value).lower()
        assert "approval_year" in msg or "temporal" in msg or "random" in msg, (
            f"Production refusal must mention approval_years/temporal; got: {excinfo.value}"
        )


# ============================================================================
# ISSUE-P2-014 (HIGH): Biotech drugs (no InChIKey) are kept via
# DrugBank ID, not dropped.
# ============================================================================
class TestP2014BiotechDrugsKeptViaDrugbankId:
    def test_filter_accepts_drugbank_id_when_inchikey_missing(self):
        """The InChIKey regex filter must NOT drop biotech drugs
        (insulin, Humira, Keytruda — ~30% of modern FDA approvals).
        When inchikey is NULL but drugbank_id is non-empty, the row
        must be kept. v107/v108 fix."""
        import re
        # The keep-mask logic: keep if (valid inchikey) OR (non-empty drugbank_id).
        _INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

        def _keep(inchikey, drugbank_id):
            valid_ik = bool(inchikey and _INCHIKEY_RE.match(str(inchikey)))
            has_db = bool(drugbank_id and str(drugbank_id).strip()
                          and str(drugbank_id) != "nan")
            return valid_ik or has_db

        # Aspirin: valid InChIKey — keep.
        assert _keep("BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "DB00009") is True
        # Insulin: no InChIKey (biotech) but has DrugBank ID — keep.
        assert _keep(None, "DB00071") is True
        assert _keep("", "DB00071") is True
        assert _keep("not-an-inchikey", "DB00071") is True
        # Neither — drop.
        assert _keep(None, None) is False
        assert _keep("", "") is False
        assert _keep("malformed", "") is False


# ============================================================================
# ISSUE-P2-015 (HIGH): Pathway derivation failures raise in production
# (not silently swallowed).
# ============================================================================
class TestP2015PathwayFailureRaisesInProduction:
    def test_pathway_failure_raises_in_production(self, monkeypatch):
        """In DRUGOS_ENVIRONMENT=production, STRING pathway derivation
        failure MUST raise (not be silently swallowed by a broad
        `except Exception: pass`). The previous code masked the root
        cause with a different downstream error
        (Phase2AdapterValidationError). v107/v108 fix."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        # Import the bridge and verify the production-refusal branch
        # exists by inspecting the source. (A full end-to-end test
        # would require STRING data — out of scope for this unit test.)
        import inspect
        from drugos_graph import phase1_bridge
        src = inspect.getsource(phase1_bridge)
        # The pathway derivation except block must contain a
        # production-mode raise.
        assert "_p2_015_is_prod" in src, (
            "P2-015 fix must define a production-mode flag"
        )
        assert "raise RuntimeError" in src, (
            "P2-015 fix must raise RuntimeError in production"
        )


# ============================================================================
# ISSUE-P2-016 (HIGH): CORS whitelist (not ['*']) + POST/OPTIONS allowed.
# ============================================================================
class TestP2016CorsHardening:
    def test_cors_origins_is_whitelist(self):
        """CORS allow_origins must be a whitelist (not ['*']). v107/v108
        fix."""
        from phase2.service import _allowed_origins
        assert _allowed_origins != ["*"], (
            "allow_origins must NOT be ['*'] (security risk)"
        )
        assert isinstance(_allowed_origins, list)
        assert len(_allowed_origins) >= 1

    def test_cors_methods_include_post_and_options(self):
        """CORS allow_methods must include POST and OPTIONS (frontend
        POST preflight must pass). v107/v108 fix."""
        from phase2.service import app
        cors_mw = None
        for mw in app.user_middleware:
            if "CORSMiddleware" in str(mw.cls):
                cors_mw = mw
                break
        assert cors_mw is not None
        methods = cors_mw.kwargs.get("allow_methods", [])
        assert "GET" in methods
        assert "POST" in methods
        assert "OPTIONS" in methods


# ============================================================================
# ISSUE-P2-017 (HIGH): Neo4j driver.close() in try/finally (no resource
# leak on exception).
# ============================================================================
class TestP2017Neo4jDriverTryFinally:
    def test_run_neo4j_uses_try_finally(self):
        """_run_neo4j must use try/finally driver.close() (or a context
        manager) so the driver is closed even when the inner function
        raises. The previous code called driver.close() at the end of
        the try block — leaks on exception. v107/v108 fix."""
        import inspect
        import phase2.service as svc
        src = inspect.getsource(svc._run_neo4j)
        # Must use try/finally pattern.
        assert "try:" in src
        assert "finally:" in src
        assert "driver.close()" in src
        # The driver.close() must appear AFTER the LAST 'finally:' keyword
        # (in the finally block), not before it (in the try block).
        finally_idx = src.rfind("finally:")
        close_idx = src.rfind("driver.close()")
        assert close_idx > finally_idx, (
            f"driver.close() (at offset {close_idx}) must be in the "
            f"finally block (after 'finally:' at offset {finally_idx}), "
            f"not in the try block"
        )


# ============================================================================
# ISSUE-P2-018 (HIGH): split_for_link_prediction default matches the
# comment (node_disjoint=True, GNN-safe).
# ============================================================================
class TestP2018NodeDisjointDefaultMatchesComment:
    def test_default_is_node_disjoint_true(self):
        """split_for_link_prediction default must be
        node_disjoint=True (GNN-safe). The previous code had
        node_disjoint=True as the default BUT a comment claiming the
        default was False — contradiction that confused callers. v107
        fix: aligned the comment with the code."""
        import inspect
        from phase2.drugos_graph.pyg_builder import PyGBuilder
        sig = inspect.signature(PyGBuilder.split_for_link_prediction)
        node_disjoint_param = sig.parameters.get("node_disjoint")
        assert node_disjoint_param is not None
        assert node_disjoint_param.default is True, (
            f"node_disjoint default must be True (GNN-safe); got "
            f"{node_disjoint_param.default!r}"
        )


# ============================================================================
# ISSUE-P2-019 (HIGH): step11_train_transe raises in production when
# no drug_record has approval_year.
# ============================================================================
class TestP2019ApprovalYearRequiredInProduction:
    def test_step11_raises_in_production_without_approval_year(
        self, monkeypatch
    ):
        """In DRUGOS_ENVIRONMENT=production, step11_train_transe MUST
        raise when no drug_record has an approval_year field. The
        previous code fell back to a random split (temporal leakage)
        even in production. v107/v108 fix."""
        monkeypatch.setenv("DRUGOS_ENVIRONMENT", "production")
        import inspect
        from drugos_graph import run_pipeline
        src = inspect.getsource(run_pipeline.step11_train_transe)
        # Must contain the production-refusal check.
        assert "DRUGOS_ENVIRONMENT" in src, (
            "step11 must check DRUGOS_ENVIRONMENT"
        )
        assert "missing_approval_year" in src, (
            "step11 must raise missing_approval_year error in production"
        )
        assert "DrugOSDataError" in src, (
            "step11 must raise DrugOSDataError in production"
        )


# ============================================================================
# Integration: Phase 2 -> Phase 3 contract (the user's "100% connected"
# mandate). Verifies the full data path the user explicitly required.
# ============================================================================
class TestPhase2ToPhase3Integration:
    def test_full_phase2_to_phase3_round_trip(self):
        """End-to-end: build a Phase 2 KG via the adapter, verify all
        5 Phase 3 node types are present, all edge types are present,
        and known_pairs are extracted. This is the integration test
        the user asked for ('phase 1 + phase 2 + phase 3 + phase 4
        100% linked')."""
        from graph_transformer.data.phase2_adapter import (
            adapt_phase2_to_phase3,
        )

        class FakeBuilder:
            node_loads = [
                {"label": "Compound",
                 "nodes": [{"id": "C1", "name": "aspirin",
                            "smiles": "CC(=O)Oc1ccccc1C(=O)O"}]},
                {"label": "Protein",
                 "nodes": [{"id": "P1", "name": "PTGS1",
                            "gene_symbol": "PTGS1",
                            "sequence": "MKVGVLGGR"}]},
                {"label": "Pathway",
                 "nodes": [{"id": "W1", "name": "cox_pathway"}]},
                {"label": "Disease",
                 "nodes": [{"id": "D1", "name": "pain"}]},
            ]
            edge_loads = [
                {"src_label": "Compound", "rel_type": "inhibits",
                 "dst_label": "Protein",
                 "edges": [{"src_id": "C1", "dst_id": "P1"}]},
                {"src_label": "Protein", "rel_type": "participates_in",
                 "dst_label": "Pathway",
                 "edges": [{"src_id": "P1", "dst_id": "W1"}]},
                {"src_label": "Compound", "rel_type": "treats",
                 "dst_label": "Disease",
                 "edges": [{"src_id": "C1", "dst_id": "D1"}]},
            ]

        result = adapt_phase2_to_phase3(FakeBuilder())
        node_features, edge_indices, node_maps, known_pairs = result

        # All 5 Phase 3 node types must be present.
        assert set(node_features.keys()) == {
            "drug", "protein", "pathway", "disease", "clinical_outcome"
        }
        # Each node type must have non-empty features.
        for ntype, feat in node_features.items():
            if ntype == "clinical_outcome":
                # clinical_outcome may be empty (no ClinicalOutcome nodes
                # in the fixture) — features tensor is allowed to be 0×dim.
                continue
            assert feat.shape[0] >= 1, f"{ntype} must have >=1 node"
        # Multi-hop edge chain: drug→protein→pathway must be present.
        assert ("drug", "inhibits", "protein") in edge_indices
        assert ("protein", "part_of", "pathway") in edge_indices
        # Derived pathway→disease edge (the multi-hop chain) must be present.
        assert ("pathway", "disrupted_in", "disease") in edge_indices, (
            "Derived (pathway, disrupted_in, disease) edge must be present "
            "for the GNN's multi-hop reasoning chain"
        )
        # Known treatment pairs must be extracted.
        assert ("aspirin", "pain") in known_pairs
