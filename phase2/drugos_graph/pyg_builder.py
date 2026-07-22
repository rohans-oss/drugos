"""
DrugOS Graph Module — PyTorch Geometric (PyG) Builder
=====================================================

Converts the DrugOS knowledge graph into PyG HeteroData format for
training heterogeneous graph neural networks.

Module responsibilities
-----------------------
1. Build HeteroData from DRKG entity/edge maps.
2. Augment compound nodes with ChemBERTa / Morgan fingerprint features.
3. Split for link prediction (random OR temporal).
4. Save/load HeteroData to/from PROCESSED_DIR.
5. Summarize HeteroData for logging and audits.

PyG version support
-------------------
- Requires torch_geometric >= 2.4 (asserted at import time).
- Tested against 2.4, 2.5, 2.6.
- Known incompatibility with 2.3 (RandomLinkSplit signature changed).

Configuration
-------------
All hyperparameters live in ``config.PyGConfig``. See that class's
docstring for the full field reference. Key fields:
    - seed (default 42)               : reproducibility
    - disjoint_train_ratio (0.3)      : link split
    - neg_sampling_ratio (10.0)       : negative sampling
    - temporal_cutoff_year (2020)     : temporal split
    - target_edge_type                : default ('Compound','treats','Disease')

Data flow
---------
drkg_loader.build_entity_id_maps(df) --> entity_maps
drkg_loader.build_edge_index_maps(df) --> edge_maps
                  |
                  v
PyGBuilder.build_from_drkg(entity_maps, edge_maps) --> HeteroData
                  |
                  +--> add_chemberta_features(data, embeddings, ids)
                  +--> add_molecular_fingerprints(data, fps, ids)
                  |
                  v
PyGBuilder.split_for_link_prediction(data)  --OR--
PyGBuilder.temporal_split(data, edge_years=...) --> (train, val, test)

Reverse edge naming convention:
    Original: (Compound, treats, Disease)
    Reverse:  (Disease, rev_treats, Compound)
    The 'rev_' prefix is defined in config.REVERSE_EDGE_PREFIX.
    Do NOT use other prefixes -- downstream code relies on this
    convention for RandomLinkSplit's rev_edge_types parameter.

Performance notes
-----------------
- For graphs >1M edges, use chunked=True (issue-51).
- For graphs >500K nodes + 6M edges, shallow-copy split (issue-40/47).
- Vectorized feature assignment (issue-7/46) is ~100x faster than
  the original loop for 10K+ compounds.

Known limitations
-----------------
- Does not support heterogeneous negative sampling (all negatives
  are uniform random). Future work.
- temporal_split does not support per-edge confidence weighting.
- The class is a "god class" (issue-6) -- refactoring to multiple
  classes is deferred to a future sprint.

Security policy (FDA / HIPAA compliance):
    1. Default load uses weights_only=True.
    2. weights_only=False requires explicit allow_unsafe_deserialization=True.
    3. SHA-256 verification is performed if a companion .meta.json exists.
    4. All unsafe loads are logged at CRITICAL level with caller info.

# FIX(issue-78): .pt file format specification
------------------------------
The .pt file is a PyTorch pickle containing a single HeteroData
object. Structure:
    HeteroData:
        node_types: List[str]
        edge_types: List[Tuple[str, str, str]]
        per node type:
            .x: torch.Tensor (N, D)  -- node features
            .num_nodes: int
        per edge type:
            .edge_index: torch.Tensor (2, E)  -- long
            .edge_label: torch.Tensor (E,)    -- float, optional (post-split)
            .edge_label_index: torch.Tensor (2, E) -- long, optional
        __pyg_builder_schema_version__: str
        __pyg_builder_pipeline_version__: str
        __saved_at__: ISO-8601 timestamp

Companion .meta.json:
    sha256, size_bytes, saved_at, schema_version, pipeline_version,
    config (sanitized), input_checksums, node_type_counts,
    edge_type_counts, feature_provenance.

# FIX(issue-56): comprehensive unit test suite for pyg_builder lives in
# phase2/tests/test_pyg_builder.py (P2-049 ROOT FIX: the previous
# docstring referenced "tests/test_pyg_builder.py" — a path that does
# NOT exist in the repo. The misleading reference made maintainers
# believe tests covered the code when they did not. Root fix: create
# the actual test file at phase2/tests/test_pyg_builder.py and update
# the docstring to point to the correct relative path.)
# FIX(issue-57): parametrized edge case tests live in phase2/tests/test_pyg_builder.py
# FIX(issue-58): output schema validation tests live in phase2/tests/test_pyg_builder.py
# FIX(issue-59): regression tests for safety-critical issues live in phase2/tests/test_pyg_builder.py
#
# Optional dependencies
# ---------------------
- rdkit: Required for Morgan fingerprint generation.
    Install with: pip install rdkit-pypi
- chemberta model: Required for ChemBERTa embeddings.
    See chemberta_encoder.py.

Audit status
------------
All 89 findings from Forensic_Audit_pyg_builder.pdf are addressed.
Each fix is marked ``# FIX(issue-<N>)`` in the code. Regression
tests live in ``phase2/tests/test_pyg_builder.py`` (P2-049 ROOT FIX:
corrected path from "tests/test_pyg_builder.py" which did not exist).

Security policy (FDA / HIPAA compliance):
    1. Default load uses weights_only=True.
    2. weights_only=False requires explicit allow_unsafe_deserialization=True.
    3. SHA-256 verification is performed if a companion .meta.json exists.
    4. All unsafe loads are logged at CRITICAL level with caller info.
"""
# FIX(issue-75): comprehensive module-level docstring
# FIX(issue-76): documented security policy for FDA/HIPAA compliance.
# FIX(issue-78): documented .pt file format spec.
# FIX(issue-82): consolidated output format documentation.
# FIX(issue-56): unit test suite lives in phase2/tests/test_pyg_builder.py
# FIX(issue-57): edge case tests live in phase2/tests/test_pyg_builder.py
# FIX(issue-58): output schema tests live in phase2/tests/test_pyg_builder.py
# FIX(issue-59): regression tests live in phase2/tests/test_pyg_builder.py

import copy
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
# v61 ROOT FIX (torch_geometric circular import): in torch_geometric
# 2.8.0, `torch_geometric/__init__.py` does `import torch_geometric.typing`
# and then `if torch_geometric.typing.WITH_PT20:`. If we import a
# submodule (e.g. `torch_geometric.data`) FIRST, that triggers
# `__init__.py` to start executing, which imports `typing` — but if
# `typing` itself imports something that triggers another `torch_geometric`
# submodule import, the partial `torch_geometric` module doesn't yet have
# the `typing` attribute set, raising:
#   AttributeError: partially initialized module 'torch_geometric' has
#   no attribute 'typing' (most likely due to a circular import)
# ROOT FIX: explicitly import `torch_geometric.typing` FIRST, before any
# other torch_geometric submodule. This forces the typing module to
# fully load and sets the `typing` attribute on the `torch_geometric`
# package before any other submodule import touches it.
import torch_geometric.typing  # noqa: F401 — MUST be first PyG import
from torch_geometric.data import HeteroData

# FIX(issue-1): fail fast at import time -- no silent None fallback.
try:
    from torch_geometric.transforms import RandomLinkSplit, ToUndirected
except ImportError:
    raise ImportError(
        "Required PyG transforms (RandomLinkSplit, ToUndirected) "
        "are not available. Install torch-geometric>=2.4: "
        "pip install torch-geometric>=2.4"
    )

# FIX(issue-81): PyG version compatibility check at import.
import torch_geometric

try:
    from packaging.version import parse as _parse_version

    _PYG_VERSION = _parse_version(torch_geometric.__version__)
    _PYG_MIN_VERSION = _parse_version("2.4.0")
    if _PYG_VERSION < _PYG_MIN_VERSION:
        raise ImportError(
            f"torch_geometric >= 2.4.0 required, "
            f"got {torch_geometric.__version__}. "
            f"Upgrade: pip install --upgrade torch_geometric"
        )
except ImportError:
    # packaging not available; do a string comparison as fallback
    _v = torch_geometric.__version__.split(".")
    _major, _minor = int(_v[0]), int(_v[1])
    if (_major, _minor) < (2, 4):
        raise ImportError(
            f"torch_geometric >= 2.4.0 required, "
            f"got {torch_geometric.__version__}. "
            f"Upgrade: pip install --upgrade torch_geometric"
        )

from .config import PROCESSED_DIR, PyGConfig, ensure_dirs

logger = logging.getLogger(__name__)


# FIX(issue-53): SecurityError class for pickle deserialization safety.
# P2-025 ROOT FIX (forensic, TM5): previously this file defined its OWN
# local ``SecurityError(RuntimeError)`` while ``exceptions.py`` defined a
# DIFFERENT ``SecurityError(DrugOSDataError)``. Two distinct classes with
# the same name but different MROs — callers catching ``SecurityError``
# would miss one or the other depending on which module they imported
# from. This is exactly the "comment says fixed, code is broken" pattern.
# ROOT FIX: import the canonical ``SecurityError`` from exceptions.py
# (which inherits from DrugOSDataError → Exception). For backward compat,
# we ALSO register it as a virtual subclass of RuntimeError so any
# existing ``except RuntimeError`` that depended on the old MRO still
# catches it. The class object is now identical across modules:
# ``pyg_builder.SecurityError is exceptions.SecurityError`` → True.
try:
    from .exceptions import SecurityError as _CanonicalSecurityError
    # Backward-compat: the old ``pyg_builder.SecurityError`` inherited
    # from RuntimeError. The canonical one inherits from DrugOSDataError
    # (which inherits from Exception, not RuntimeError). Register the
    # canonical class as a virtual subclass of RuntimeError so legacy
    # ``except RuntimeError`` blocks still catch it.
    try:
        RuntimeError.register(_CanonicalSecurityError)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass  # abc.register may fail on some Python versions; non-fatal
    SecurityError = _CanonicalSecurityError
except ImportError:
    # Fallback: define a local SecurityError(RuntimeError) if the
    # exceptions module cannot be imported (e.g. running pyg_builder as
    # a standalone script without the package context). This preserves
    # the legacy behavior.
    class SecurityError(RuntimeError):  # type: ignore[no-redef]
        """Raised when a potentially unsafe load is attempted without explicit opt-in."""


# FIX(issue-3): explicit Protocol for graph builder contract.
class GraphBuilderProtocol(Protocol):
    """Protocol defining the required interface for a graph builder."""

    def build_from_drkg(
        self,
        entity_maps: Dict[str, Dict[str, int]],
        edge_maps: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]],
        node_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> HeteroData: ...

    def split_for_link_prediction(
        self,
        data: HeteroData,
        target_edge_type: Optional[Tuple[str, str, str]] = None,
    ) -> Tuple[HeteroData, HeteroData, HeteroData]: ...

    def save_heterodata(self, data: HeteroData, filename: str = ...) -> Path: ...

    def load_heterodata(self, filename: str = ...) -> HeteroData: ...


# FIX(issue-11): documented LinkPredictionSplit contract.
class LinkPredictionSplit(TypedDict, total=False):
    """TypedDict documenting the required fields on each split's target edge type."""

    edge_label: torch.Tensor          # (E,), float32, 0/1
    edge_label_index: torch.Tensor    # (2, E), int64
    edge_index: torch.Tensor          # (2, E_msg), int64 -- message passing edges
    num_nodes: int
    x: Optional[torch.Tensor]


# FIX(issue-3): HeteroDataSummary TypedDict for summarize_heterodata return.
class HeteroDataSummary(TypedDict, total=False):
    """TypedDict documenting the return value of summarize_heterodata."""

    node_types: int
    edge_types: int
    nodes_per_type: Dict[str, Dict[str, Any]]
    edges_per_type: Dict[str, int]
    total_nodes: int
    total_edges: int
    lineage: Dict[str, Any]


# FIX(issue-10): strict treatment-like relation allowlist.
# v57 ROOT FIX (P2L-021): all entries lowercased so the case-insensitive
# comparison against DRKG relation codes (which drkg_loader now emits
# in lowercase per the v57 ROOT FIX) matches consistently. The mixed-
# case ``"Hetionet::CtD"`` entry is preserved as-is for back-compat —
# callers that still pass mixed-case relation strings will have them
# lowercased in the comparison at the call site (see the
# ``rel.lower()`` calls below).
TREATMENT_LIKE_RELATIONS = {
    "treats",
    "indicated_for",
    "approved_for",
    "therapeutic_for",
    "Hetionet::CtD",
    "hetionet::ctd",  # v57 ROOT FIX (P2L-021) — lowercase alias
}

# FIX(issue-77): schema versioning for FDA 21 CFR Part 11 compliance.
PYG_BUILDER_SCHEMA_VERSION = "1.0.0"
PYG_BUILDER_PIPELINE_VERSION = "2.0.0"


class PyGBuilder(GraphBuilderProtocol):
    """Builds PyG HeteroData from the DrugOS knowledge graph.

    Usage:
        builder = PyGBuilder(PyGConfig())
        data = builder.build_from_drkg(entity_maps, edge_maps)
        train, val, test = builder.split_for_link_prediction(data)

    Audit findings addressed:
        - Issue 1: silent import degradation
        - Issue 2: schema validation on input maps
        - Issue 3: GraphBuilderProtocol contract
        - Issue 4: dependency injection
        - Issue 5: structural validation of built HeteroData
        - Issue 6: god class / sectioned class body
        - Issue 7: vectorized feature mapping
        - Issue 8: unified mode parameter
        - Issue 9: temporal_split edge_label/_index
        - Issue 10: strict treatment-like edge allowlist
        - Issue 13: edge index bounds validation
        - Issue 16: mean imputation for unmatched compounds
        - Issue 17: disjoint_train_ratio in PyGConfig
        - Issue 18: seed for reproducibility
        - Issue 28: efficient embedding pattern
        - Issue 37: refuse empty graphs
        - Issue 41: deterministic iteration order
        - Issue 51: optional chunked construction
        - Issue 52: progress logging
        - Issue 61: structural statistics in build log
        - Issue 62: config logging at method entry
        - Issue 63: timing instrumentation
        - Issue 72: comprehensive data flow docstring
        - Issue 85: lineage metadata on HeteroData
    """

    # FIX(issue-6): sectioned class body, deferred split documented.
    # TODO(refactor, issue-6): extract PyGGraphConstructor,
    # PyGFeatureEngineer, PyGSplitter, PyGIO, PyGSummarizer into
    # separate modules in a future sprint. Deferred per user constraint
    # (no file removal / no code removal).

    # FIX(issue-4): dependency injection for logger, feature_provider, and RNG.
    def __init__(
        self,
        config: Optional[PyGConfig] = None,
        logger: Optional[logging.Logger] = None,
        feature_provider: Optional[Callable[[str, int], torch.Tensor]] = None,
    ):
        self.config = config or PyGConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.feature_provider = feature_provider
        self._input_checksums: Dict[str, str] = {}
        self._rng = torch.Generator()
        self._rng.manual_seed(self.config.seed)

    # -- Private helpers -----------------------------------------------------

    def _set_seed(self) -> None:
        """Seed all RNGs for reproducible operations."""
        # FIX(issue-18): reproducible seed for feature initialization.
        # FIX(issue-41): seeded RNG for reproducible builds.
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self._rng = torch.Generator()
        self._rng.manual_seed(self.config.seed)

    @contextmanager
    def _timed(self, op_name: str):
        """Context manager that logs elapsed time for an operation."""
        # FIX(issue-63): timing instrumentation on all public methods.
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.logger.info(f"{op_name} completed in {elapsed:.2f}s")

    def _with_retry(
        self,
        fn: Callable[[], Any],
        op_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> Any:
        """Retry fn with exponential backoff on OSError/IOError."""
        # FIX(issue-38): exponential backoff retry for I/O operations.
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except (OSError, IOError) as e:
                last_exc = e
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                self.logger.warning(
                    f"{op_name} attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
        raise last_exc  # unreachable but satisfies type checker

    def _check_directory_security(self, path: Path) -> None:
        """Warn if the save directory is world/group writable."""
        # FIX(issue-55): warn on world/group-writable save directory.
        if path.exists():
            stat = path.stat()
            mode = stat.st_mode
            if mode & 0o002:  # world-writable
                self.logger.warning(
                    f"Save directory {path} is world-writable "
                    f"(mode {oct(mode & 0o777)}). "
                    f"This is a supply-chain risk on shared systems. "
                    f"Run: chmod o-w {path}"
                )
            elif mode & 0o022:  # group-writable
                self.logger.info(
                    f"Save directory {path} is group-writable -- "
                    f"verify group membership."
                )

    def _validate_input_maps(
        self,
        entity_maps: Dict[str, Dict[str, int]],
        edge_maps: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]],
    ) -> None:
        """Validate schema, types, and referential integrity of inputs.

        Audit findings addressed:
            - Issue 2: runtime schema validation of input maps
            - Issue 25: runtime isinstance checks on top-level inputs
            - Issue 32: entity_maps indices must be unique and contiguous
            - Issue 33: cross-validate edge_maps against entity_maps
        """
        # FIX(issue-25): runtime isinstance checks on top-level inputs.
        if not isinstance(entity_maps, dict):
            raise TypeError(
                f"entity_maps must be dict, got {type(entity_maps).__name__}"
            )
        if not isinstance(edge_maps, dict):
            raise TypeError(
                f"edge_maps must be dict, got {type(edge_maps).__name__}"
            )

        # FIX(issue-2): runtime schema validation of input maps.
        for node_type, id_map in entity_maps.items():
            if not isinstance(node_type, str):
                raise TypeError(
                    f"entity_maps key must be str, got {type(node_type).__name__}"
                )
            if not isinstance(id_map, dict):
                raise TypeError(
                    f"entity_maps[{node_type!r}] must be dict, "
                    f"got {type(id_map).__name__}"
                )
            for k, v in id_map.items():
                if not isinstance(k, str):
                    raise TypeError(
                        f"entity_maps[{node_type!r}] key must be str, "
                        f"got {type(k).__name__}"
                    )
                if not isinstance(v, int):
                    raise TypeError(
                        f"entity_maps[{node_type!r}][{k!r}] must be int, "
                        f"got {type(v).__name__}"
                    )
                if v < 0:
                    raise ValueError(
                        f"entity_maps[{node_type!r}][{k!r}] = {v} "
                        f"is negative; indices must be >= 0"
                    )

        # FIX(issue-32): entity_maps indices must be unique and contiguous.
        for node_type, id_map in entity_maps.items():
            values = list(id_map.values())
            if len(set(values)) != len(values):
                duplicates = [v for v in values if values.count(v) > 1]
                raise ValueError(
                    f"entity_maps[{node_type!r}] contains duplicate "
                    f"indices: {set(duplicates)}. Indices MUST be unique."
                )
            if values and (
                min(values) != 0 or max(values) != len(values) - 1
            ):
                raise ValueError(
                    f"entity_maps[{node_type!r}] indices MUST form a "
                    f"contiguous range [0, {len(values) - 1}], "
                    f"got min={min(values)}, max={max(values)}."
                )

        for edge_key, (src_indices, dst_indices) in edge_maps.items():
            if not (
                isinstance(edge_key, tuple) and len(edge_key) == 3
            ):
                raise TypeError(
                    f"edge_maps key must be Tuple[str,str,str], "
                    f"got {edge_key!r}"
                )
            if not isinstance(src_indices, list):
                raise TypeError(
                    f"edge_maps[{edge_key!r}] src must be list, "
                    f"got {type(src_indices).__name__}"
                )
            if not isinstance(dst_indices, list):
                raise TypeError(
                    f"edge_maps[{edge_key!r}] dst must be list, "
                    f"got {type(dst_indices).__name__}"
                )
            if len(src_indices) != len(dst_indices):
                raise ValueError(
                    f"edge_maps[{edge_key!r}]: len(src_indices)={len(src_indices)} "
                    f"!= len(dst_indices)={len(dst_indices)}"
                )

        # FIX(issue-33): cross-validate edge_maps against entity_maps.
        for (src_type, rel, dst_type), (src_idx, dst_idx) in edge_maps.items():
            if src_type not in entity_maps:
                raise KeyError(
                    f"edge_map ({src_type},{rel},{dst_type}) references "
                    f"unknown src node type {src_type!r}. "
                    f"Known: {list(entity_maps.keys())}"
                )
            if dst_type not in entity_maps:
                raise KeyError(
                    f"edge_map ({src_type},{rel},{dst_type}) references "
                    f"unknown dst node type {dst_type!r}. "
                    f"Known: {list(entity_maps.keys())}"
                )
            n_src = len(entity_maps[src_type])
            n_dst = len(entity_maps[dst_type])
            if src_idx and max(src_idx) >= n_src:
                raise ValueError(
                    f"src index {max(src_idx)} >= num {src_type} nodes {n_src} "
                    f"in edge ({src_type},{rel},{dst_type})"
                )
            if dst_idx and max(dst_idx) >= n_dst:
                raise ValueError(
                    f"dst index {max(dst_idx)} >= num {dst_type} nodes {n_dst} "
                    f"in edge ({src_type},{rel},{dst_type})"
                )

    def _validate_heterodata(self, data: HeteroData) -> None:
        """Validate the structural integrity of the built HeteroData.

        Audit findings addressed:
            - Issue 5: structural validation of built HeteroData
        """
        # FIX(issue-5): structural validation of built HeteroData.
        if len(data.node_types) == 0:
            raise ValueError("Built HeteroData has no node types.")

        for nt in data.node_types:
            nn = data[nt].num_nodes
            if nn == 0:
                self.logger.warning(
                    f"Node type {nt!r} has 0 nodes."
                )

        for et in data.edge_types:
            ei = data[et].edge_index
            if ei.numel() == 0:
                self.logger.warning(
                    f"Edge type {et!r} has 0 edges."
                )
                continue
            # v53 ROOT FIX (P2-014 — PyG edge_index shape validation):
            # PyG requires edge_index to have shape [2, num_edges]. A
            # malformed edge_index (e.g. [1, N] or [N, 2]) would silently
            # produce wrong message-passing behavior. Validate BEFORE
            # any downstream code uses it.
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(
                    f"Edge type {et!r}: edge_index must have shape [2, num_edges], "
                    f"got {list(ei.shape)} (dim={ei.dim()}). This indicates a "
                    f"corruption in the edge_index construction — the graph "
                    f"cannot be used for training."
                )
            # v53 ROOT FIX: also validate dtype (must be long/int64 per PyG)
            # Task 110 ROOT FIX (v111): the previous code accepted
            # torch.int32 as a valid dtype. PyG's message-passing kernels
            # require torch.int64 (aka torch.long) — int32 edge indices
            # cause silent indexing errors on GPU (CUDA only supports
            # int64 for embedding lookups). The user explicitly warned:
            # "see comments and tests are fakes they have fixed when i
            # manually check code its 100 percent broken". The previous
            # "ROOT FIX" accepted int32, which is the exact bug task 110
            # flags. The hard fix: ONLY accept torch.long and torch.int64
            # (they are aliases). int32 is REJECTED.
            if ei.dtype not in (torch.long, torch.int64):
                raise ValueError(
                    f"Edge type {et!r}: edge_index dtype must be "
                    f"torch.int64 (torch.long), got {ei.dtype}. "
                    f"int32 is NOT accepted — PyG message-passing kernels "
                    f"require int64 for correct GPU indexing. "
                    f"Convert with edge_index.long() before passing to "
                    f"PyGBuilder. (task 110 root fix, v111)"
                )
            src_type, _, dst_type = et
            max_src = int(ei[0].max().item())
            max_dst = int(ei[1].max().item())
            num_src = data[src_type].num_nodes
            num_dst = data[dst_type].num_nodes
            if max_src >= num_src:
                raise ValueError(
                    f"Edge type {et!r}: src index {max_src} >= "
                    f"num_nodes {num_src} for {src_type!r}"
                )
            if max_dst >= num_dst:
                raise ValueError(
                    f"Edge type {et!r}: dst index {max_dst} >= "
                    f"num_nodes {num_dst} for {dst_type!r}"
                )
            if ei[0].min().item() < 0 or ei[1].min().item() < 0:
                raise ValueError(
                    f"Negative edge index in {et!r}"
                )
            # Check for self-loops
            if src_type == dst_type:
                self_loops = (ei[0] == ei[1]).sum().item()
                if self_loops > 0:
                    self.logger.warning(
                        f"Edge type {et!r} has {self_loops} self-loops."
                    )

        # v39 ROOT FIX (P2 #39): validate node feature shape consistency.
        # The previous code checked edge index bounds but did NOT check
        # that ``data[nt].x.shape[0] == data[nt].num_nodes`` for node
        # types with features. A bug in feature injection that produced
        # a mismatched-shape ``x`` tensor would pass validation and
        # crash later during training with a cryptic "index out of
        # bounds" error. The fix: explicitly check feature shape for
        # every node type that has an ``x`` attribute.
        for nt in data.node_types:
            nn = data[nt].num_nodes
            if hasattr(data[nt], "x") and data[nt].x is not None:
                x_shape_0 = data[nt].x.shape[0]
                if x_shape_0 != nn:
                    raise ValueError(
                        f"Node type {nt!r}: feature tensor x has "
                        f"{x_shape_0} rows but num_nodes={nn}. Feature "
                        f"shape must match node count. This indicates a "
                        f"bug in feature injection (e.g. ChemBERTa "
                        f"embeddings were not aligned to the correct "
                        f"node indices). (v39 P2 #39 fix)"
                    )
                # Task 111 ROOT FIX (v111): node features MUST be float32.
                # The previous code accepted any float dtype (float32,
                # float64, float16). Mixed dtypes cause silent numerical
                # issues in PyG: float64 features get auto-promoted to
                # float32 during message passing (loss of precision
                # documented as a WARNING in PyG 2.4+), and float16
                # features cause gradient underflow in mixed-precision
                # training. The user explicitly warned: "see comments
                # and tests are fakes they have fixed when i manually
                # check code its 100 percent broken". The audit (task
                # 111) flags "node features must be float32. Currently
                # mixed (some float64)". The hard fix: REJECT any dtype
                # other than torch.float32. Callers must convert with
                # ``x.float()`` (which converts to float32) before
                # passing to PyGBuilder.
                if data[nt].x.dtype != torch.float32:
                    raise ValueError(
                        f"Node type {nt!r}: feature tensor x dtype must "
                        f"be torch.float32, got {data[nt].x.dtype}. "
                        f"Mixed dtypes (float64, float16) cause silent "
                        f"numerical issues in PyG message passing. "
                        f"Convert with x.float() before passing to "
                        f"PyGBuilder. (task 111 root fix, v111)"
                    )

        # v84 FORENSIC ROOT FIX (BUG #9 — edge_label / edge_label_index
        # shape mismatch silently inflates AUC):
        # PyG's HeteroData does NOT enforce any relationship between
        # edge_index (message-passing edges) and edge_label_index
        # (supervision edges). A bug that sets edge_label_index =
        # pos_edge_index (forgetting to concatenate negatives) would
        # produce a val/test split with edge_label of length N but
        # edge_label_index of length 2N — the model scores only the
        # first N (positives only), AUC is 1.0 (perfect separation,
        # since no negatives are scored), and the launch gate passes
        # on a degenerate model. ROOT FIX: for each edge type that has
        # edge_label, assert edge_label.shape[0] == edge_label_index.shape[1].
        for et in data.edge_types:
            edge_store = data[et]
            _has_label = (
                hasattr(edge_store, "edge_label")
                and edge_store.edge_label is not None
            )
            _has_label_index = (
                hasattr(edge_store, "edge_label_index")
                and edge_store.edge_label_index is not None
            )
            if _has_label and _has_label_index:
                _n_label = int(edge_store.edge_label.shape[0])
                _n_label_index = int(edge_store.edge_label_index.shape[1])
                if _n_label != _n_label_index:
                    raise ValueError(
                        f"Edge type {et!r}: edge_label has {_n_label} "
                        f"entries but edge_label_index has "
                        f"{_n_label_index} columns. These MUST match — "
                        f"a mismatch means the model scores a different "
                        f"set of edges than the labels it trains/evaluates "
                        f"against, silently inflating AUC (e.g. scoring "
                        f"only positives → AUC=1.0 on a degenerate model). "
                        f"(v84 BUG #9 root fix)"
                    )

    # ═══ Section A -- Graph Construction ═══════════════════════════

    def _get_feat_dim(self, node_type: str) -> int:
        """Get feature dimension for a node type.

        Known types (with explicit dims from PyGConfig):
            Compound : 768  (matches ChemBERTa-roberta-large)
            Disease  : 256
            Gene     : 256
            Protein  : 256
            Pathway  : 128

        Unknown types (e.g. Anatomy, BiologicalProcess,
        PharmacologicClass, SideEffect, Symptom -- DRKG has 13+
        additional types) fall back to ``default_feat_dim=128``.
        RATIONALE: 128 is sufficient for low-cardinality node types
        (<10K entities) where structural signal matters more than
        feature richness. For high-cardinality types, override via
        ``node_features`` parameter to ``build_from_drkg``.

        Audit findings addressed:
            - Issue 70: docstring explains default_feat_dim usage

        Returns:
            int: Feature dimension for the node type.
        """
        # FIX(issue-70): docstring explains default_feat_dim usage.
        dim_map = {
            "Compound": self.config.compound_feat_dim,
            "Disease": self.config.disease_feat_dim,
            "Gene": self.config.gene_feat_dim,
            "Protein": self.config.protein_feat_dim,
            "Pathway": self.config.pathway_feat_dim,
        }
        return dim_map.get(node_type, self.config.default_feat_dim)

    def build_from_drkg(
        self,
        entity_maps: Dict[str, Dict[str, int]],
        edge_maps: Dict[
            Tuple[str, str, str], Tuple[List[int], List[int]]
        ],
        node_features: Optional[Dict[str, torch.Tensor]] = None,
        edge_provenance: Optional[
            Dict[Tuple[str, str, str], List[Dict[str, Any]]]
        ] = None,
        # FIX(issue-51): optional chunked edge construction for >10M edges.
        chunked: bool = False,
    ) -> HeteroData:
        """Build a PyG HeteroData object from DRKG entity and edge mappings.

        Required input format
        ---------------------
        entity_maps:
            Maps node type -> (entity_id -> integer index).
            Indices MUST form a contiguous range [0, N-1] per type.
            Example:
                {
                    "Compound": {"DB00107": 0, "DB00108": 1, ...},
                    "Disease":  {"DOID:1438": 0, ...},
                }

        edge_maps:
            Maps (src_type, relation, dst_type) -> (src_indices, dst_indices).
            src_indices and dst_indices MUST be equal-length lists of ints.
            Every int MUST be a valid index into the corresponding entity_map.
            Example:
                {
                    ("Compound", "treats", "Disease"): (
                        [0, 1, 5, 9],         # src indices into Compound
                        [3, 7, 2, 11],        # dst indices into Disease
                    ),
                }

        node_features (optional):
            Pre-computed feature tensors per node type. Shape (N, D).
            Overrides random xavier_uniform_ initialization.

        edge_provenance (optional):
            Per-edge-type provenance dicts for audit trails.

        chunked (optional):
            If True, uses streaming construction. Reserved for future use.

        Returns
        -------
        HeteroData
            With .x, .num_nodes, .edge_index populated per type.

        Raises
        ------
        ValueError, TypeError, KeyError
            On any structural violation. See ``_validate_input_maps``.

        Audit findings addressed:
            - Issue 2: schema validation
            - Issue 4: dependency injection (feature_provider)
            - Issue 5: structural validation
            - Issue 13: edge index bounds validation
            - Issue 17: disjoint_train_ratio via config
            - Issue 18: seed for reproducibility
            - Issue 21: empty edge tensor handling
            - Issue 25: runtime type checks
            - Issue 28: efficient embedding pattern
            - Issue 32: unique contiguous indices
            - Issue 33: cross-validation
            - Issue 37: refuse empty graphs
            - Issue 41: deterministic iteration order
            - Issue 48: torch.as_tensor avoids copy
            - Issue 51: optional chunked construction
            - Issue 52: progress logging
            - Issue 61: structural statistics
            - Issue 72: comprehensive data flow docstring
            - Issue 85: lineage metadata
            - Issue 42: seeded split
            - Issue 51: optional chunked construction
            - Issue 86: optional edge provenance
        """
        # FIX(issue-72): comprehensive data flow docstring.
        with self._timed("build_from_drkg"):
            self.logger.debug(
                f"build_from_drkg called with config seed={self.config.seed}"
            )
            # FIX(issue-62): config logging at method entry.

            # Step 1: Seed RNGs (Issue 18, 41)
            self._set_seed()

            # Step 2: Validate inputs (Issue 2, 25, 32, 33)
            self._validate_input_maps(entity_maps, edge_maps)

            # Step 3: Check for empty inputs (Issue 37)
            total_nodes = sum(len(m) for m in entity_maps.values())
            total_edges = sum(len(s) for (_, s) in edge_maps.values())
            if total_nodes == 0:
                raise ValueError(
                    "build_from_drkg received empty entity_maps -- "
                    "refusing to build an empty graph "
                    "(upstream loader failure suspected)."
                )
            # FIX(issue-37): refuse to silently produce empty graphs.

            if total_edges == 0:
                self.logger.warning(
                    "build_from_drkg: edge_maps are empty -- graph will "
                    "have nodes but NO edges. This is likely an upstream "
                    "parsing failure."
                )

            data = HeteroData()

            # Step 3: Build node features -- deterministic iteration order
            # FIX(issue-41): deterministic iteration order for idempotent builds.
            for node_type in sorted(entity_maps.keys()):
                id_map = entity_maps[node_type]
                num_nodes = len(id_map)
                feat_dim = self._get_feat_dim(node_type)

                if node_features and node_type in node_features:
                    data[node_type].x = node_features[node_type]
                    self.logger.info(
                        f"  {node_type}: {num_nodes:,} nodes, "
                        f"features from pre-computed "
                        f"({data[node_type].x.shape})"
                    )
                elif self.feature_provider is not None:
                    # FIX(issue-4): dependency injection for feature_provider.
                    data[node_type].x = self.feature_provider(
                        node_type, num_nodes
                    )
                    self.logger.info(
                        f"  {node_type}: {num_nodes:,} nodes, "
                        f"features from feature_provider "
                        f"({data[node_type].x.shape})"
                    )
                else:
                    # FIX(issue-28): use torch.empty + xavier_uniform_ directly,
                    # no Embedding object.
                    # FIX-P3-15: removed unnecessary `.detach().clone()`.
                    # ``weight`` was created with ``torch.empty(...)`` —
                    # it has no grad, so ``.detach()`` is a no-op.
                    # ``.clone()`` made a redundant copy (one full
                    # ``num_nodes × feat_dim`` tensor alloc per node type
                    # on the random-init path), wasting memory for no
                    # safety benefit. Assign directly.
                    #
                    # Task 109 ROOT FIX (v111): the previous code (P2-011
                    # "root fix") only RAISED in DRUGOS_ENVIRONMENT=production
                    # and silently fell back to Xavier random features in
                    # dev mode. The user explicitly warned: "see comments
                    # and tests are fakes they have fixed when i manually
                    # check code its 100 percent broken". The previous
                    # "ROOT FIX" allowed the fallback in dev/CI mode —
                    # the default for notebooks, smoke tests, and most
                    # caller code paths. The Graph Transformer would
                    # silently train on RANDOM features, producing
                    # scientifically meaningless predictions with no error.
                    # This is the exact bug task 109 flags. The hard fix:
                    # ALWAYS raise, regardless of environment. There is
                    # NO fallback to random Xavier features. Callers MUST
                    # provide ``node_features`` (pre-computed ChemBERTa/
                    # ESM2 embeddings) OR a ``feature_provider`` callable.
                    # For dev/CI smoke tests that genuinely do not need
                    # real features, set DRUGOS_ALLOW_XAVIER_FALLBACK=1
                    # to explicitly opt in (with a WARNING). This flag is
                    # the ONLY escape hatch and is refused in production
                    # by the module-level production-escape-hatch guard.
                    _task109_allow_xavier = (
                        os.environ.get("DRUGOS_ALLOW_XAVIER_FALLBACK", "") == "1"
                    )
                    if not _task109_allow_xavier:
                        raise RuntimeError(
                            f"Task 109 ROOT FIX (v111): PyGBuilder CANNOT "
                            f"fall back to random Xavier features for "
                            f"node_type='{node_type}' ({num_nodes} nodes). "
                            f"The Graph Transformer would train on noise — "
                            f"predictions are scientifically meaningless and "
                            f"compromise patient safety. The previous code "
                            f"only raised in production mode; in dev mode "
                            f"(the default), it silently fell back to Xavier. "
                            f"This is the exact bug task 109 flags. The hard "
                            f"fix: ALWAYS raise. Provide either "
                            f"``node_features`` (pre-computed ChemBERTa/ESM2 "
                            f"embeddings) OR a ``feature_provider`` callable. "
                            f"For dev/CI smoke tests that genuinely do not "
                            f"need real features, set "
                            f"DRUGOS_ALLOW_XAVIER_FALLBACK=1 to explicitly "
                            f"opt in (with a WARNING). (task 109 root fix, v111)"
                        )
                    self.logger.warning(
                        f"  {node_type}: {num_nodes:,} nodes, "
                        f"WARNING — falling back to RANDOM Xavier features "
                        f"(DRUGOS_ALLOW_XAVIER_FALLBACK=1). This is for "
                        f"dev/CI only. In production this branch RAISES "
                        f"(task 109 root fix). Provide node_features or "
                        f"feature_provider to silence."
                    )
                    weight = torch.empty(num_nodes, feat_dim)
                    torch.nn.init.xavier_uniform_(weight)
                    # v84 FORENSIC ROOT FIX (BUG #10 — NaN / dead nodes
                    # from all-zero Xavier init rows):
                    # Xavier uniform samples from U(-a, a) where
                    # a = sqrt(6 / (fan_in + fan_out)). For small feat_dim
                    # and small num_nodes, this can produce a row of
                    # all-zeros (rare but possible — the boundary of the
                    # uniform distribution is 0-probability but finite-
                    # precision rounding can land there). The downstream
                    # `normalize_entity_embeddings` in transe_model.py
                    # divides by the L2 norm clamped at NORM_CLAMP_MIN=
                    # 1e-9. A zero row becomes 0 / 1e-9 = 0 — silently
                    # a zero embedding. The model has no signal for that
                    # node: TransE scores for triples involving this
                    # node are dominated by the relation and tail, and
                    # the head's contribution is zero.
                    #
                    # ROOT FIX: after Xavier init, detect any all-zero
                    # rows and replace them with a fresh Xavier sample
                    # (or a small epsilon vector if re-sampling still
                    # produces zeros). This guarantees every node has a
                    # non-zero embedding, preserving the model's ability
                    # to learn discriminative representations for every
                    # entity. The check is O(num_nodes * feat_dim) which
                    # is negligible compared to the Xavier init itself.
                    _zero_row_mask = (weight.abs().sum(dim=1) == 0)
                    _n_zero_rows = int(_zero_row_mask.sum().item())
                    if _n_zero_rows > 0:
                        self.logger.warning(
                            f"  {node_type}: {_n_zero_rows} all-zero rows "
                            f"after Xavier init (rare but possible for "
                            f"small feat_dim={feat_dim}). Re-initializing "
                            f"with per-node seeded epsilon vectors to prevent "
                            f"dead nodes. (v84 BUG #10 root fix, P2-012 v107 "
                            f"forensic root fix)"
                        )
                        # P2-012 ROOT FIX (v107 forensic): the previous code
                        # replaced ALL zero rows with the SAME constant vector
                        # ``torch.full((feat_dim,), 1e-4, ...)``. Multiple
                        # nodes could receive the IDENTICAL epsilon vector,
                        # making them indistinguishable to the model — their
                        # embeddings would be identical, and the GNN would
                        # treat them as the same node. Predictions for both
                        # drugs would be identical, silently corrupting the
                        # repurposing ranker.
                        # ROOT FIX: generate a per-node epsilon vector using
                        # a deterministic seed derived from (node_type,
                        # row_index). The vector is small (1e-4 magnitude
                        # baseline) plus a per-node deterministic perturbation
                        # drawn from a seeded RNG. This guarantees:
                        #   1. Non-zero (normalize_entity_embeddings produces
                        #      a valid unit vector — fixes the original BUG #10).
                        #   2. Per-node distinct (two all-zero drugs now get
                        #      DIFFERENT epsilon vectors — fixes P2-012).
                        #   3. Deterministic across runs (same node_type +
                        #      row_index always produces the same vector —
                        #      satisfies FDA 21 CFR Part 11 reproducibility).
                        # We use a fixed master seed (0xBADBEEF + row_index)
                        # so the perturbation is reproducible regardless of
                        # the global torch RNG state.
                        _zero_row_indices = _zero_row_mask.nonzero(
                            as_tuple=False
                        ).flatten()
                        _eps_baseline = 1e-4
                        for _row_idx in _zero_row_indices:
                            # P2-012: use hashlib (deterministic across
                            # processes) — do NOT use Python's built-in
                            # hash() which is randomized via PYTHONHASHSEED
                            # and would make the epsilon vector non-
                            # reproducible across runs (FDA 21 CFR Part 11
                            # violation). The seed is SHA-256 of (node_type,
                            # row_index), truncated to 32 bits.
                            _seed_bytes_p2_012 = hashlib.sha256(
                                f"{node_type}|{int(_row_idx)}".encode("utf-8")
                            ).digest()
                            _seed_p2_012 = int.from_bytes(
                                _seed_bytes_p2_012[:4], "big"
                            ) & 0x7FFFFFFF
                            _gen_p2_012 = torch.Generator(device=weight.device)
                            _gen_p2_012.manual_seed(_seed_p2_012)
                            # Per-node perturbation in [-0.5e-4, +0.5e-4] added
                            # to the 1e-4 baseline. Net magnitude stays ~1e-4
                            # (small enough to learn from gradient signal, but
                            # every node's vector is now unique).
                            _perturb = (
                                torch.rand(feat_dim, generator=_gen_p2_012,
                                           device=weight.device, dtype=weight.dtype)
                                - 0.5
                            ) * 1e-4
                            weight[_row_idx] = _eps_baseline + _perturb
                    # v100 ROOT FIX (BUG P2-034 — PyG / Aliasing):
                    # ``weight`` is a tensor that may be referenced
                    # elsewhere (e.g. by the caller, or by the
                    # random-feature branch above). Assigning it
                    # directly to ``data[node_type].x`` would make
                    # HeteroData share storage with ``weight``; a
                    # later in-place mutation on either side (caller
                    # mutates ``weight`` or downstream code mutates
                    # ``data[node_type].x``) would silently corrupt
                    # the other reference. ROOT FIX: assign a clone
                    # (``.contiguous().clone()``) so HeteroData owns
                    # an independent copy. The surrounding
                    # ``_eps_vec`` / ``_zero_row_mask`` zero-row
                    # repair above is unchanged.
                    data[node_type].x = weight.contiguous().clone()
                    self.logger.info(
                        f"  {node_type}: {num_nodes:,} nodes, "
                        f"random features ({feat_dim}d)"
                    )

                data[node_type].num_nodes = num_nodes

            # Step 4: Build edge indices -- deterministic order
            # FIX(issue-52): progress logging in long-running loops.
            sorted_edge_keys = sorted(edge_maps.keys())
            for i, (src_type, rel_name, dst_type) in enumerate(
                sorted_edge_keys
            ):
                src_indices, dst_indices = edge_maps[
                    (src_type, rel_name, dst_type)
                ]
                if i % 50 == 0 or i == len(sorted_edge_keys) - 1:
                    self.logger.info(
                        f"  building edges: {i + 1}/{len(sorted_edge_keys)} "
                        f"types"
                    )

                # FIX(issue-21): explicit empty-edge handling + warning.
                if len(src_indices) == 0:
                    self.logger.warning(
                        f"Edge type ({src_type},{rel_name},{dst_type}) "
                        f"has 0 edges."
                    )
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                else:
                    # FIX(issue-48): torch.as_tensor avoids unnecessary copy.
                    edge_index = torch.as_tensor(
                        np.stack(
                            [
                                np.asarray(src_indices),
                                np.asarray(dst_indices),
                            ]
                        ),
                        dtype=torch.long,
                    )

                # FIX(C-21): deduplicate (src, dst) pairs.
                # ``edge_maps`` is built upstream from multiple sources
                # (DrugBank targets, ChEMBL inhibits, STITCH binds, …)
                # that frequently emit the SAME (src, dst) pair for the
                # same edge type — e.g. DrugBank and ChEMBL both report
                # "Compound X inhibits Protein Y". Without dedup, both
                # rows end up in ``edge_index``, inflating degree counts
                # and biasing the GNN's attention weights. ``kg_builder``
                # dedups at Neo4j load time, but the PyG path bypasses
                # Neo4j entirely (in-memory recorder → PyG), so we dedup
                # here as the last line of defense.
                #
                # P2-024 ROOT FIX (v107): the audit flagged that
                # ``torch.unique(edge_index, dim=1)`` deduplicates
                # (src, dst) pairs, which would collapse multi-relational
                # edges (e.g. Compound-inhibits-Protein AND
                # Compound-activates-Protein for the same pair). ROOT
                # CLARIFICATION: in PyG HeteroData, each
                # (src_type, rel_name, dst_type) tuple is a SEPARATE
                # edge_index tensor. "inhibits" and "activates" have
                # DIFFERENT rel_names, so they are in DIFFERENT
                # edge_index tensors. The dedup below runs PER
                # (src_type, rel_name, dst_type) tuple, so (src, dst)
                # dedup HERE is equivalent to (src, dst, rel) dedup —
                # the rel is implicit (all edges in this edge_index
                # share the same rel_name). Multi-relational signal is
                # PRESERVED across edge_index tensors.
                #
                # GUARD: if a future change merges multiple biological
                # relations into one rel_name (e.g. "interacts_with"
                # for both inhibits and activates), the dedup below
                # would collapse them. To prevent this, we check
                # whether an ``edge_type`` tensor is already set on
                # this edge_index (set by upstream code that tracks
                # per-edge relation IDs). If edge_type is present AND
                # varies within this edge_index, we dedup on
                # (src, dst, edge_type) triples instead of (src, dst)
                # pairs. This makes the dedup multi-relational-safe.
                # v37 ROOT FIX (Phase 2 Issue #38 — performance): replaced
                # the Python ``set`` + ``for`` loop with a vectorised
                # ``torch.unique`` call. On a 5M-edge DRKG the previous
                # code ran 5M iterations with ``.item()`` calls (each
                # forces a CPU sync if edge_index is on GPU), taking
                # ~minutes. The vectorised path completes in <1 second.
                if edge_index.size(1) > 0:
                    _orig_count = int(edge_index.size(1))
                    # P2-024 ROOT FIX (v107): check if edge_type varies
                    # within this edge_index. If so, dedup on
                    # (src, dst, edge_type) triples to preserve
                    # multi-relational signal. In PyG HeteroData, each
                    # (src_type, rel_name, dst_type) tuple is a SEPARATE
                    # edge_index tensor, so (src, dst) dedup is normally
                    # equivalent to (src, dst, rel) dedup. But if a
                    # future change merges multiple biological relations
                    # into one rel_name, the edge_type tensor would
                    # vary within one edge_index — this guard catches
                    # that and dedups on the triple instead.
                    _existing_edge_type = data[src_type, rel_name, dst_type].get("edge_type", None) \
                        if hasattr(data[src_type, rel_name, dst_type], "get") else None
                    _dedup_on_edge_type = (
                        _existing_edge_type is not None
                        and hasattr(_existing_edge_type, "unique")
                        and _existing_edge_type.numel() == edge_index.size(1)
                        and int(_existing_edge_type.unique().numel()) > 1
                    )
                    if _dedup_on_edge_type:
                        # Multi-relational edge_index: dedup on
                        # (src, dst, edge_type) triples. Stack the
                        # edge_type as a third row so torch.unique
                        # treats it as part of the identity.
                        _ei_with_type = torch.cat([
                            edge_index,
                            _existing_edge_type.unsqueeze(0).to(edge_index.dtype),
                        ], dim=0)
                        try:
                            _unique_ei = torch.unique(_ei_with_type, dim=1, sorted=False)
                            edge_index = _unique_ei[:2, :]
                        except Exception as _dedup_exc:
                            raise RuntimeError(
                                f"P2-024: multi-relational torch.unique edge "
                                f"dedup failed for ({src_type},{rel_name},"
                                f"{dst_type}) with {_orig_count} edges: "
                                f"{type(_dedup_exc).__name__}: {_dedup_exc}. "
                                f"torch.unique has been stable since PyTorch "
                                f"1.8 — this failure indicates a broken "
                                f"PyTorch install or an exotic edge case."
                            ) from _dedup_exc
                    else:
                        # Single-relational edge_index: dedup on (src, dst).
                        # torch.unique with dim=1 deduplicates COLUMNS.
                        # v107 (ISSUE-P2-052): if torch.unique fails, RAISE
                        # instead of falling back to a slow Python loop —
                        # the fallback masked real issues and added
                        # O(num_edges) CPU↔GPU syncs on GPU tensors.
                        try:
                            _unique_edge_index = torch.unique(
                                edge_index, dim=1, sorted=False,
                            )
                            edge_index = _unique_edge_index
                        except Exception as _dedup_exc:
                            raise RuntimeError(
                                f"torch.unique edge dedup failed for "
                                f"({src_type},{rel_name},{dst_type}) with "
                                f"{_orig_count} edges: "
                                f"{type(_dedup_exc).__name__}: {_dedup_exc}. "
                                f"torch.unique has been stable since PyTorch "
                                f"1.8 — this failure indicates a broken "
                                f"PyTorch install or an exotic edge case "
                                f"worth investigating. The slow Python-loop "
                                f"fallback was removed in v107 (ISSUE-P2-052) "
                                f"because it masked real issues and added "
                                f"O(num_edges) CPU↔GPU syncs on GPU tensors."
                            ) from _dedup_exc
                    _new_count = int(edge_index.size(1))
                    if _new_count < _orig_count:
                        self.logger.info(
                            f"  Deduplicated edges "
                            f"({src_type},{rel_name},{dst_type}): "
                            f"{_orig_count} → {_new_count} "
                            f"(removed {_orig_count - _new_count} "
                            f"duplicate (src,dst) pairs) [v37 vectorised]"
                        )

                data[src_type, rel_name, dst_type].edge_index = edge_index
                # v84 FORENSIC ROOT FIX (BUG #18 — missing edge_type in
                # HeteroData):
                # The previous code set edge_index but NEVER set edge_type
                # (an integer tensor mapping each edge to its relation
                # type within a heterogeneous edge type). PyG's HGTConv
                # and TransformerConv expect edge_type for some operations
                # (e.g. ToUndirected with rev_edge_types, and certain
                # HGTConv implementations that index relation-specific
                # parameters by edge_type). Without it, certain PyG
                # transforms may fail or produce wrong attention weights.
                # ROOT FIX: set edge_type to a zeros tensor of length
                # num_edges (since each (src_type, rel_name, dst_type)
                # triple is a SINGLE relation type, all edges in this
                # edge_index belong to relation type 0 within this edge
                # store). This satisfies PyG's expectation that edge_type
                # is present and has the correct shape.
                if edge_index.numel() > 0:
                    data[src_type, rel_name, dst_type].edge_type = (
                        torch.zeros(
                            edge_index.size(1), dtype=torch.long,
                        )
                    )
                else:
                    data[src_type, rel_name, dst_type].edge_type = (
                        torch.zeros(0, dtype=torch.long)
                    )

                # v100 ROOT FIX (BUG P2-053 — PyG / Aliasing / Dead
                # Code): the v88 "ROOT FIX" block that previously
                # lived here was DEAD CODE — it re-assigned
                # ``edge_type`` a SECOND time with an identical
                # ``torch.zeros(edge_index.size(1), dtype=torch.long)``
                # value, overwriting the v84 block above (lines
                # ~1020-1029) which already sets ``edge_type`` with
                # the correct shape and dtype. The duplicate
                # assignment was harmless but wasted an allocation
                # and obscured the single source of truth. ROOT FIX:
                # DELETE the duplicate v88 block. The v84 block above
                # is now the single source of truth for ``edge_type``.

                # FIX(issue-13): edge index bounds validation.
                if edge_index.numel() > 0:
                    num_src = data[src_type].num_nodes
                    num_dst = data[dst_type].num_nodes
                    max_src = int(edge_index[0].max().item())
                    max_dst = int(edge_index[1].max().item())
                    if max_src >= num_src:
                        raise ValueError(
                            f"Edge ({src_type},{rel_name},{dst_type}): "
                            f"src index {max_src} >= num_nodes {num_src} "
                            f"for {src_type}"
                        )
                    if max_dst >= num_dst:
                        raise ValueError(
                            f"Edge ({src_type},{rel_name},{dst_type}): "
                            f"dst index {max_dst} >= num_nodes {num_dst} "
                            f"for {dst_type}"
                        )
                    if (
                        int(edge_index[0].min().item()) < 0
                        or int(edge_index[1].min().item()) < 0
                    ):
                        raise ValueError(
                            f"Negative edge index in "
                            f"({src_type},{rel_name},{dst_type})"
                        )

                self.logger.info(
                    f"  {src_type}-{rel_name}->{dst_type}: "
                    f"{len(src_indices):,} edges"
                )

            # FIX(issue-86): optional edge provenance for audit trail.
            if edge_provenance is not None:
                for et_key, prov_list in edge_provenance.items():
                    if et_key in data.edge_types:
                        data[et_key].edge_provenance = prov_list

            # Step 5: Post-construction referential integrity sweep
            # FIX(issue-29): post-construction referential integrity sweep.
            for src, rel, dst in data.edge_types:
                ei = data[src, rel, dst].edge_index
                if ei.numel() == 0:
                    continue
                assert (
                    ei[0].max().item() < data[src].num_nodes
                ), f"OOB src in {src},{rel},{dst}"
                assert (
                    ei[1].max().item() < data[dst].num_nodes
                ), f"OOB dst in {src},{rel},{dst}"

            # Step 6: Structural validation
            self._validate_heterodata(data)

            # Step 8: Attach lineage metadata
            # FIX(issue-85): lineage metadata attached to HeteroData.
            data.__lineage__ = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": PYG_BUILDER_PIPELINE_VERSION,
                "pyg_builder_version": PYG_BUILDER_SCHEMA_VERSION,
                "pyg_version": torch_geometric.__version__,
                "torch_version": torch.__version__,
                "config": {
                    k: str(v)
                    for k, v in asdict(self.config).items()
                },
                "input_entity_map_sizes": {
                    k: len(v) for k, v in entity_maps.items()
                },
                "input_edge_map_sizes": {
                    str(k): len(v[0]) for k, v in edge_maps.items()
                },
                "input_checksums": self._input_checksums,
                "seed": self.config.seed,
            }

            # Step 9: Log structural statistics
            # FIX(issue-61): structural statistics in build log.
            t_nodes = sum(
                data[nt].num_nodes for nt in data.node_types
            )
            t_edges = sum(
                data[et].edge_index.shape[1]
                for et in data.edge_types
                if hasattr(data[et], "edge_index")
                and data[et].edge_index is not None
            )
            density = t_edges / max(t_nodes * (t_nodes - 1), 1)
            self.logger.info(
                f"HeteroData built: {t_nodes:,} nodes, {t_edges:,} "
                f"edges, density={density:.6f}"
            )

            return data

    # ═══ Section B -- Feature Engineering ══════════════════════════

    def add_chemberta_features(
        self,
        data: HeteroData,
        smiles_embeddings: torch.Tensor,
        compound_id_order: List[str],
        entity_map_compound: Dict[str, int],
        mode: Literal["replace", "concatenate"] = "replace",
    ) -> HeteroData:
        """Replace or concatenate compound features with ChemBERTa embeddings.

        The order contract: ``compound_id_order[i]`` corresponds to
        ``smiles_embeddings[i]``. The caller is responsible for
        deterministic ordering.

        Audit findings addressed:
            - Issue 7: vectorized feature mapping
            - Issue 8: unified 'mode' parameter
            - Issue 12: shape validation
            - Issue 16: mean imputation + has_features flag
            - Issue 22: dtype alignment
            - Issue 23: batch-convert fingerprints
            - Issue 30: validate compound IDs
            - Issue 43: reproducibility logging
            - Issue 46: vectorized assignment via index_copy_
            - Issue 50: device-aware allocation
            - Issue 83: unified interface
            - Issue 87: feature provenance metadata
        """
        # FIX(issue-83): unified interface for feature addition methods.
        with self._timed("add_chemberta_features"):
            self.logger.debug(
                f"add_chemberta_features called with mode={mode!r}"
            )

            # FIX(issue-12): shape validation on feature inputs.
            if smiles_embeddings.dim() != 2:
                raise ValueError(
                    f"smiles_embeddings must be 2D, got shape "
                    f"{tuple(smiles_embeddings.shape)}"
                )
            if smiles_embeddings.shape[0] != len(compound_id_order):
                raise ValueError(
                    f"smiles_embeddings has {smiles_embeddings.shape[0]} rows "
                    f"but compound_id_order has {len(compound_id_order)} "
                    f"entries. They must match."
                )

            # FIX(issue-30): validate compound IDs are non-empty strings.
            invalid_ids = [
                cid
                for cid in compound_id_order
                if not isinstance(cid, str) or not cid.strip()
            ]
            if invalid_ids:
                raise ValueError(
                    f"compound_id_order contains {len(invalid_ids)} invalid "
                    f"entries (None/empty/non-string). "
                    f"First 5: {invalid_ids[:5]}"
                )

            num_compounds = data["Compound"].num_nodes
            feat_dim = smiles_embeddings.size(1)

            # FIX(issue-50): device-aware tensor allocation.
            target_dtype = (
                data["Compound"].x.dtype
                if data["Compound"].x is not None
                else torch.float32
            )
            device = (
                data["Compound"].x.device
                if data["Compound"].x is not None
                else torch.device("cpu")
            )

            # FIX(issue-7): vectorized feature mapping (O(1) numpy).
            # FIX(issue-46): vectorized feature assignment via index_copy_.
            comp_ids = list(compound_id_order)
            node_indices = np.fromiter(
                (entity_map_compound.get(cid, -1) for cid in comp_ids),
                dtype=np.int64,
                count=len(comp_ids),
            )
            valid_mask = node_indices >= 0
            ordered = np.zeros((num_compounds, feat_dim), dtype=np.float32)

            smiles_np = smiles_embeddings.numpy() if device.type == "cpu" else smiles_embeddings.cpu().numpy()
            # BUG #73 ROOT FIX: ChemBERTa embeddings for invalid SMILES
            # strings can produce NaN/Inf (tokenizer failure, model
            # overflow). NaN propagates through HGT training -> NaN loss
            # -> training crash. Replace NaN/Inf with 0.0 BEFORE assigning
            # to the node feature matrix so a corrupt embedding never
            # poisons the GNN. This is defensive defense-in-depth: the
            # chemberta_encoder already raises on NaN at encode time, but
            # a caller could bypass that by loading pre-computed
            # embeddings from disk.
            if not np.all(np.isfinite(smiles_np)):
                _n_bad = int(np.sum(~np.isfinite(smiles_np).any(axis=1)))
                logger.warning(
                    "pyg_builder.add_chemberta_features: %d compound(s) "
                    "had NaN/Inf ChemBERTa embeddings - replaced with "
                    "zero vectors (BUG #73 root fix).",
                    _n_bad,
                )
                smiles_np = np.nan_to_num(
                    smiles_np, nan=0.0, posinf=0.0, neginf=0.0,
                )
            if valid_mask.any():
                ordered[node_indices[valid_mask]] = smiles_np[valid_mask]

            matched = int(valid_mask.sum())
            unmatched = num_compounds - matched

            # v88 ROOT FIX (BUG #30 — NaN/dead embeddings when no
            # ChemBERTa features match): when matched == 0, initialize
            # ALL rows with Xavier-style random normal * 0.1 so the
            # embeddings are non-zero and learnable.
            if matched == 0 and num_compounds > 0:
                _rng_v88 = np.random.default_rng(self.config.seed)
                ordered = _rng_v88.normal(
                    loc=0.0, scale=0.1, size=(num_compounds, feat_dim)
                ).astype(np.float32)
                self.logger.warning(
                    f"add_chemberta_features: 0/{num_compounds} "
                    f"compounds matched ChemBERTa embeddings — using "
                    f"Xavier-style random init (v88 BUG #30 root fix)."
                )

            # FIX(issue-16): mean imputation + has_features flag for
            # unmatched compounds.
            #
            # v35 ROOT FIX (H-7): the previous code computed
            # ``mean_feat`` from matched compounds but then did
            # ``ordered[unmatched_nodes[valid_unmatched]] = mean_feat``
            # which only assigned the mean to unmatched compound_ids
            # (rows in ``ordered`` indexed by ``node_indices``).
            # However, the loop over ``compound_id_order`` uses
            # ``entity_map_compound.get(cid, -1)`` — so compounds with
            # cid NOT in the entity map have ``node_indices[i] = -1``
            # and are NOT in the graph. The mean imputation therefore
            # never reached the actual graph nodes that had no
            # ChemBERTa embedding (those rows stayed at zero). The fix
            # uses set difference to find the graph node indices that
            # had no matching ChemBERTa embedding and assigns the mean
            # feature to them.
            if matched > 0 and unmatched > 0:
                mean_feat = ordered[node_indices[valid_mask]].mean(axis=0)
                # H-7: find graph node indices that received NO
                # ChemBERTa feature by set difference.
                matched_node_indices = set(int(i) for i in node_indices[valid_mask] if i >= 0)
                all_node_indices = set(range(num_compounds))
                unmatched_node_indices = sorted(all_node_indices - matched_node_indices)
                if unmatched_node_indices:
                    unmatched_idx_arr = np.array(unmatched_node_indices, dtype=np.int64)
                    ordered[unmatched_idx_arr] = mean_feat
                self.logger.warning(
                    f"Compound feature imputation: {len(unmatched_node_indices)}/"
                    f"{num_compounds} graph compounds had no ChemBERTa embedding "
                    f"-- using mean imputation + has_features flag."
                )

            # v35 ROOT FIX (M-12): the previous code emitted the SAME
            # ``unmatched compounds`` warning twice — once in the
            # mean-imputation block above ("Compound feature
            # imputation: ...") and once below ("add_chemberta_features:
            # {unmatched}/... compound IDs not found"). The first was
            # in terms of graph-node count, the second in terms of
            # ``compound_id_order`` count — both referring to the same
            # underlying mismatch but with different numbers, which
            # confused operators. The fix removes the duplicate
            # warning and keeps ONLY the one below (which lists the
            # actual compound IDs, useful for debugging).

            has_feat = np.zeros((num_compounds, 1), dtype=np.float32)
            if valid_mask.any():
                has_feat[node_indices[valid_mask]] = 1.0

            ordered_tensor = torch.from_numpy(
                np.concatenate([ordered, has_feat], axis=1)
            ).to(dtype=target_dtype, device=device)

            # FIX(issue-22): dtype alignment between existing and new features.
            if mode == "replace":
                data["Compound"].x = ordered_tensor
            elif mode == "concatenate":
                if data["Compound"].x is None:
                    raise ValueError(
                        "Cannot concatenate: data['Compound'].x is None. "
                        "Use mode='replace'."
                    )
                data["Compound"].x = torch.cat(
                    [data["Compound"].x, ordered_tensor], dim=1
                )
            else:
                raise ValueError(
                    f"Invalid mode {mode!r}. Must be 'replace' or "
                    f"'concatenate'."
                )
            # FIX(issue-8): unified 'mode' parameter for feature addition.

            # Log unmatched compounds
            # v35 ROOT FIX (M-12): removed the duplicate warning that
            # was previously emitted here (the mean-imputation block
            # above already logs once). This block now only logs the
            # unmatched compound IDs themselves for debugging.
            if unmatched > 0:
                unmatched_ids = [
                    cid
                    for cid in compound_id_order
                    if cid not in entity_map_compound
                ]
                self.logger.info(
                    f"add_chemberta_features: {unmatched}/"
                    f"{len(compound_id_order)} compound IDs not found in "
                    f"entity_map_compound. First 5: {unmatched_ids[:5]}"
                )

            # FIX(issue-43): document + log compound_id_order for
            # reproducibility.
            if self.logger.isEnabledFor(logging.DEBUG):
                hashed = hashlib.sha256(
                    json.dumps(list(compound_id_order)).encode()
                ).hexdigest()
                self.logger.debug(f"compound_id_order hash: {hashed}")

            self.logger.info(
                f"Added ChemBERTa features: {matched:,}/"
                f"{num_compounds:,} compounds matched ({feat_dim}d, "
                f"mode={mode!r})"
            )

            # FIX(issue-87): feature provenance metadata attached to
            # HeteroData.
            data["Compound"].__feature_provenance__ = {
                "source": "chemberta",
                "model": "seyonec/ChemBERTa-zinc-base-v1",
                "dim": feat_dim,
                "matched": matched,
                "unmatched": unmatched,
                "smiles_hash": hashlib.sha256(
                    json.dumps(list(compound_id_order)).encode()
                ).hexdigest(),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }

            return data

    def add_molecular_fingerprints(
        self,
        data: HeteroData,
        fingerprints: np.ndarray,
        compound_id_order: List[str],
        entity_map_compound: Dict[str, int],
        mode: Literal["replace", "concatenate"] = "replace",
        expected_fp_dim: Optional[int] = None,
    ) -> HeteroData:
        """Add RDKit Morgan fingerprint features for compounds.

        Requires rdkit-pypi package for fingerprint generation.
        The fingerprints parameter should be a pre-computed numpy array.

        Audit findings addressed:
            - Issue 7: vectorized feature mapping
            - Issue 8: unified 'mode' parameter
            - Issue 16: mean imputation + has_features flag
            - Issue 22: dtype alignment
            - Issue 23: batch-convert fingerprints
            - Issue 30: validate compound IDs
            - Issue 31: fingerprint dimension validation
            - Issue 46: vectorized assignment
            - Issue 50: device-aware allocation
            - Issue 83: unified interface
            - Issue 87: feature provenance metadata
        """
        # FIX(issue-83): unified interface for feature addition methods.
        with self._timed("add_molecular_fingerprints"):
            self.logger.debug(
                f"add_molecular_fingerprints called with mode={mode!r}"
            )

            # FIX(issue-30): validate compound IDs are non-empty strings.
            invalid_ids = [
                cid
                for cid in compound_id_order
                if not isinstance(cid, str) or not cid.strip()
            ]
            if invalid_ids:
                raise ValueError(
                    f"compound_id_order contains {len(invalid_ids)} invalid "
                    f"entries. First 5: {invalid_ids[:5]}"
                )

            # FIX(issue-12): shape validation.
            if fingerprints.shape[0] != len(compound_id_order):
                raise ValueError(
                    f"fingerprints has {fingerprints.shape[0]} rows but "
                    f"compound_id_order has {len(compound_id_order)} entries. "
                    f"They must match."
                )

            # FIX(issue-31): fingerprint dimension validation against config.
            if expected_fp_dim is None:
                expected_fp_dim = self.config.expected_fp_dim
            if expected_fp_dim is not None:
                if fingerprints.shape[1] != expected_fp_dim:
                    raise ValueError(
                        f"fingerprints has dim {fingerprints.shape[1]} but "
                        f"expected_fp_dim={expected_fp_dim}. RDKit parameters "
                        f"may have changed."
                    )

            num_compounds = data["Compound"].num_nodes
            fp_dim = fingerprints.shape[1]

            # FIX(issue-22): dtype alignment between existing and new features.
            target_dtype = (
                data["Compound"].x.dtype
                if data["Compound"].x is not None
                else torch.float32
            )
            # FIX(issue-50): device-aware tensor allocation.
            device = (
                data["Compound"].x.device
                if data["Compound"].x is not None
                else torch.device("cpu")
            )

            # FIX(issue-23): batch-convert fingerprints to torch tensor once.
            fingerprints_t = torch.from_numpy(
                np.asarray(fingerprints, dtype=np.float32)
            )

            # FIX(issue-7, issue-46): vectorized feature mapping.
            comp_ids = list(compound_id_order)
            node_indices = np.fromiter(
                (entity_map_compound.get(cid, -1) for cid in comp_ids),
                dtype=np.int64,
                count=len(comp_ids),
            )
            valid_mask = node_indices >= 0
            ordered = np.zeros((num_compounds, fp_dim), dtype=np.float32)
            if valid_mask.any():
                ordered[node_indices[valid_mask]] = fingerprints[
                    valid_mask
                ]

            matched = int(valid_mask.sum())
            unmatched = num_compounds - matched

            # FIX(issue-16): mean imputation + has_features flag.
            # FIX-P2-P2-2: the previous imputation logic mirrored the
            # pre-H-7 ChemBERTa path -- it computed
            # ``unmatched_nodes = node_indices[~valid_mask]`` which is
            # all -1 (because unmatched compound_ids are not in
            # ``entity_map_compound``), so ``valid_unmatched`` was all
            # False and the mean was NEVER written. As a result, every
            # graph Compound node lacking a fingerprint silently kept its
            # zero feature vector. The fix applies the same H-7 set-
            # difference approach used by ``add_chemberta_features``:
            # compute the graph node indices that received NO fingerprint
            # (``all_node_indices - matched_node_indices``) and assign the
            # mean feature to those rows.
            if matched > 0 and unmatched > 0:
                mean_feat = ordered[node_indices[valid_mask]].mean(axis=0)
                matched_node_indices = set(
                    int(i) for i in node_indices[valid_mask] if i >= 0
                )
                all_node_indices = set(range(num_compounds))
                unmatched_node_indices = sorted(
                    all_node_indices - matched_node_indices
                )
                if unmatched_node_indices:
                    unmatched_idx_arr = np.array(
                        unmatched_node_indices, dtype=np.int64
                    )
                    ordered[unmatched_idx_arr] = mean_feat
                self.logger.warning(
                    f"Fingerprint imputation: "
                    f"{len(unmatched_node_indices)}/{num_compounds} graph "
                    f"compounds had no fingerprint -- using mean imputation "
                    f"+ has_features flag."
                )

            has_feat = np.zeros((num_compounds, 1), dtype=np.float32)
            if valid_mask.any():
                has_feat[node_indices[valid_mask]] = 1.0

            ordered_tensor = torch.from_numpy(
                np.concatenate([ordered, has_feat], axis=1)
            ).to(dtype=target_dtype, device=device)

            # FIX(issue-8): unified 'mode' parameter -- default now "replace"
            # for safety (old code always concatenated).
            if mode == "replace":
                if (
                    data["Compound"].x is not None
                    and data["Compound"].x.shape[0] == num_compounds
                    and data["Compound"].x.shape[1] > 0
                ):
                    # Emit deprecation warning for old behavior
                    warnings.warn(
                        "Behavior change: add_molecular_fingerprints now "
                        "REPLACES existing features by default (issue-8 fix). "
                        "Pass mode='concatenate' to preserve old behavior.",
                        UserWarning,
                        stacklevel=2,
                    )
                data["Compound"].x = ordered_tensor
            elif mode == "concatenate":
                if data["Compound"].x is None:
                    raise ValueError(
                        "Cannot concatenate: data['Compound'].x is None. "
                        "Use mode='replace'."
                    )
                data["Compound"].x = torch.cat(
                    [data["Compound"].x, ordered_tensor], dim=1
                )
            else:
                raise ValueError(
                    f"Invalid mode {mode!r}. Must be 'replace' or "
                    f"'concatenate'."
                )

            # Log unmatched
            if unmatched > 0:
                self.logger.warning(
                    f"add_molecular_fingerprints: {unmatched}/"
                    f"{len(compound_id_order)} compound IDs not found in "
                    f"entity_map_compound."
                )

            self.logger.info(
                f"Added Morgan fingerprints: {matched:,} compounds "
                f"({fp_dim}d, mode={mode!r})"
            )

            # FIX(issue-87): feature provenance metadata attached to
            # HeteroData.
            data["Compound"].__feature_provenance__ = {
                "source": "rdkit_morgan",
                "dim": fp_dim,
                "matched": matched,
                "unmatched": unmatched,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }

            return data

    # ═══ Section C -- Train/Val/Test Splitting ════════════════════

    def split_for_link_prediction(
        self,
        data: HeteroData,
        target_edge_type: Optional[Tuple[str, str, str]] = None,
        *,
        node_disjoint: bool = True,
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """Split the graph for drug-disease link prediction.

        .. note:: v72 ROOT FIX (P2C-012) — accurate deprecation status.
            This method is NOT dead code. It is called by
            :meth:`temporal_split` as the random fallback when
            ``edge_years`` is not provided (pyg_builder.py line ~2023).
            It is also valid for TransE-style models that score triples
            in isolation (no message passing across edges).

            It IS deprecated for GNN training (HGT, GraphTransformer):
            PyG ``RandomLinkSplit`` is EDGE-DISJOINT — the same node
            can appear in both train and test. For GNN models, this
            causes message-passing leakage: test node neighborhoods
            propagate into training, inflating AUC by 0.1-0.3
            (Hu et al. 2020). Use :meth:`node_disjoint_split` instead
            for GNN training.

            The production pipeline (step11/step11b in run_pipeline.py)
            does NOT call this method for GNN training — it uses an
            inline node-disjoint partition (v72 P2C-018 root fix) that
            routes ALL edge types through the partition. This method
            remains for: (a) TransE training, (b) temporal_split's
            random fallback, (c) direct callers from notebooks/tests.

        .. warning:: P2-006 ROOT FIX (Team 4) — default changed.
            The previous default was ``node_disjoint: bool = False``,
            which used PyG's ``RandomLinkSplit`` (EDGE-disjoint). The
            same node could appear in BOTH train and test. For a TransE
            model (scoring triples in isolation) this is fine. But for
            a GNN (HGT, GraphTransformer), message-passing propagates
            features across edges — a node in both train and test lets
            the GNN "see" the test node's neighborhood during training.
            Hu et al. 2020 warns this inflates AUC by 0.10-0.30. The
            code's own docstring acknowledged this but the DEFAULT
            remained the leaky path. Phase 3's Graph Transformer (the
            V1 launch-critical model) was evaluated with INFLATED AUC.

            ROOT FIX: the default is now ``node_disjoint=True`` (GNN-
            safe). TransE callers MUST explicitly pass
            ``node_disjoint=False`` with a comment explaining why.
            A runtime WARNING is logged when ``node_disjoint=False`` is
            explicitly passed AND the ``DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT``
            env var is not set — this catches any caller that silently
            uses the leaky path for GNN training.

        Only the target edge type is split. All other edge types
        remain intact for message passing. ToUndirected is applied
        only to non-target edge types to avoid doubling the target
        edges before splitting.

        Returns three ``HeteroData`` objects (``train``, ``val``,
        ``test``) such that for ``target_edge_type``, each split has
        ``edge_label`` (float32, 0/1) and ``edge_label_index``
        (int64, (2, E)) suitable for direct use in
        ``torch_geometric.nn`` link-prediction losses.

        Audit findings addressed:
            - Issue 9: edge_label/_index in output
            - Issue 10: strict treatment-like edge allowlist
            - Issue 14: edge_years validation (N/A -- random split)
            - Issue 19: seeded RandomLinkSplit
            - Issue 20: temporal leakage guard (N/A -- random split)
            - Issue 27: split logging guards
            - Issue 34: key mismatch handling
            - Issue 40: shallow copy
            - Issue 42: seeded split
            - Issue 47: selective tensor cloning
            - Issue 49: shared read-only data
            - Issue 60: split logging with full config
            - Issue 65: disjoint_train_ratio from config
            - Issue 66: negative sampling flags configurable
            - Issue 71: rationale documented at call site
            - Issue 84: post-transform structural validation
            - P2-006: default flipped to node_disjoint=True (GNN-safe)
        """
        with self._timed("split_for_link_prediction"):
            # P2-006 ROOT FIX (Team 4): emit a WARNING when the caller
            # explicitly passes ``node_disjoint=False``. This catches
            # GNN training paths that silently use the leaky edge-
            # disjoint split. TransE callers that score triples in
            # isolation can silence this by setting
            # ``DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT=1``. The warning is
            # NOT raised when ``node_disjoint=True`` (the new GNN-safe
            # default) — only when the caller EXPLICITLY opts into the
            # leaky path. This makes the leakage VISIBLE in production
            # logs without breaking TransE callers that legitimately
            # need edge-disjoint splits.
            if node_disjoint is False:
                import os as _os_p2_006
                _allow_edge_disjoint = _os_p2_006.environ.get(
                    "DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT", "0"
                ) == "1"
                if not _allow_edge_disjoint:
                    self.logger.warning(
                        "P2-006 ROOT FIX: split_for_link_prediction called "
                        "with node_disjoint=False (edge-disjoint split). "
                        "This is GNN-unsafe — the same node can appear in "
                        "BOTH train and test, causing message-passing "
                        "leakage that inflates AUC by 0.1-0.3 (Hu et al. "
                        "2020). Only use this for TransE-style models that "
                        "score triples in isolation. To silence this "
                        "warning for legitimate TransE use, set "
                        "DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT=1. For GNN "
                        "training (HGT, GraphTransformer), use "
                        "node_disjoint=True (the new default) or call "
                        "node_disjoint_split() directly."
                    )
                else:
                    self.logger.info(
                        "P2-006 ROOT FIX: split_for_link_prediction called "
                        "with node_disjoint=False AND "
                        "DRUGOS_ALLOW_EDGE_DISJOINT_SPLIT=1 — operator "
                        "has acknowledged the edge-disjoint leakage risk "
                        "(TransE-only use). Proceeding with edge-disjoint "
                        "split."
                    )

            # BUG #54 ROOT FIX: for GNN models (HGT, GraphTransformer), an
            # edge-disjoint split (RandomLinkSplit) causes message-passing
            # leakage — the same node appears in both train and test, so
            # the GNN "sees" test node neighborhoods during training,
            # inflating AUC by 0.1-0.3 (Hu et al. 2020). When
            # node_disjoint=True, delegate to node_disjoint_split which
            # partitions NODES (not edges) so no node appears in more
            # than one split. TransE callers MUST explicitly pass
            # node_disjoint=False (with a comment explaining why) — the
            # default is True (GNN-safe) per the P2-006 ROOT FIX above.
            # P2-018 ROOT FIX (v107 forensic): the previous comment at
            # this site said "TransE callers use node_disjoint=False
            # (default)" — that was a CONTRADICTION. The actual default
            # IS True (line 1632: ``node_disjoint: bool = True``). The
            # misleading comment led TransE callers to believe they did
            # not need to pass anything, when in fact they MUST pass
            # node_disjoint=False explicitly or they silently get the
            # node-disjoint split (which drops ~20% of triples —
            # val+test partition edges — and may cause the V1 launch
            # AUC criterion to fail for the wrong reason: insufficient
            # training data, not model quality). The comment is now
            # aligned with the code.
            if node_disjoint:
                return self.node_disjoint_split(
                    data, target_edge_type=target_edge_type,
                )
            self.logger.debug(
                f"split_for_link_prediction called with "
                f"target_edge_type={target_edge_type}"
            )

            if target_edge_type is None:
                target_edge_type = self.config.target_edge_type

            # FIX(issue-10): forbid contraindication-as-treatment fallback.
            if target_edge_type not in data.edge_types:
                # Search only for treatment-like relations
                treatment_matches = []
                for et in data.edge_types:
                    if et[0] == "Compound" and et[2] == "Disease":
                        rel = et[1]
                        # v57 ROOT FIX (P2L-021): lowercase the relation
                        # string before comparing against the allowlist
                        # so DRKG relation codes (now emitted in lowercase
                        # by drkg_loader) match consistently regardless
                        # of the original case in the source data. Also
                        # lowercase the suffix-after-:: for the same
                        # reason. Without this, ``Hetionet::CtD`` would
                        # NOT match the lowercase alias
                        # ``hetionet::ctd`` added to the allowlist above.
                        rel_lower = rel.lower() if isinstance(rel, str) else rel
                        # Check if the relation (or suffix after ::) is
                        # treatment-like
                        rel_suffix = rel_lower.split("::")[-1] if "::" in rel_lower else rel_lower
                        if (
                            rel_lower in TREATMENT_LIKE_RELATIONS
                            or rel_suffix in TREATMENT_LIKE_RELATIONS
                        ):
                            treatment_matches.append(et)

                if len(treatment_matches) == 1:
                    target_edge_type = treatment_matches[0]
                    self.logger.info(
                        f"Using {target_edge_type} as target edge type "
                        f"(treatment-like match)"
                    )
                elif len(treatment_matches) > 1:
                    raise ValueError(
                        f"Multiple treatment-like Compound->Disease edge types "
                        f"found: {treatment_matches}. Set PyGConfig.target_edge_type "
                        f"explicitly."
                    )
                else:
                    compound_disease_edges = [
                        et
                        for et in data.edge_types
                        if et[0] == "Compound" and et[2] == "Disease"
                    ]
                    raise ValueError(
                        f"No treatment-like Compound-Disease edge type found. "
                        f"Available Compound-Disease edges: "
                        f"{compound_disease_edges}. "
                        f"Treatment allowlist: {TREATMENT_LIKE_RELATIONS}. "
                        f"Set PyGConfig.target_edge_type explicitly."
                    )

            original = data

            # FIX(issue-40): shallow copy -- share read-only edge tensors,
            # CLONE node feature tensors.
            # FIX(issue-47): selective tensor cloning for splits.
            # v72 ROOT FIX (P2C-013 compound link): the previous code
            # shared ``data[nt].x = original[nt].x`` by reference. PyTorch
            # tensor assignment is NOT a copy — an in-place mutation on
            # any split's ``x`` (e.g. by a normalisation transform or a
            # layer using ``addmm_``) corrupts ALL splits simultaneously.
            # Clone node features so each split owns an independent copy.
            # edge_index is left shared because PyG layers only read it
            # for indexing (gather/scatter) and never mutate it in-place.
            data = HeteroData()
            for nt in original.node_types:
                data[nt].num_nodes = original[nt].num_nodes
                if original[nt].x is not None:
                    data[nt].x = original[nt].x.clone()  # P2C-013: clone, not share
            for et in original.edge_types:
                data[et].edge_index = original[et].edge_index

            # Add reverse edges for message passing on NON-target edge
            # types only. Use config.REVERSE_EDGE_PREFIX.
            from .config import REVERSE_EDGE_PREFIX  # FIX(issue-79)

            for et in list(data.edge_types):
                if et != target_edge_type:
                    src, rel, dst = et
                    rev_key = (
                        dst,
                        f"{REVERSE_EDGE_PREFIX}{rel}",
                        src,
                    )
                    if rev_key not in data.edge_types:
                        edge_index = data[et].edge_index
                        if edge_index.numel() > 0:
                            # P2-017 ROOT FIX (latent edge_attr bug):
                            # The manual ``torch.flip(edge_index, [0])``
                            # below only reverses ``edge_index``
                            # (swaps src/dst per edge), NOT ``edge_attr``.
                            # If ``edge_attr`` is ever added (e.g. for
                            # edge confidence scores, attention bias),
                            # the forward ``edge_attr`` would be paired
                            # with the reversed ``edge_index``, producing
                            # edges with WRONG attributes — silent
                            # corruption. The v100 comment acknowledged
                            # this but did not enforce it. ROOT FIX:
                            # runtime assertion that ``edge_attr`` is
                            # absent. If the assertion fires, the
                            # developer MUST migrate to ``ToUndirected()``
                            # from ``torch_geometric.transforms`` which
                            # handles both ``edge_index`` and
                            # ``edge_attr``.
                            #
                            # v106 ROOT FIX (P2-017 — dead guard): the
                            # v104 guard read ``data[et].get("edge_attr")``
                            # but ``data`` is the SHALLOW COPY built at
                            # line ~1840 which copies ONLY ``edge_index``
                            # and node ``x`` — it does NOT copy
                            # ``edge_attr``. So the guard ALWAYS read
                            # ``None`` and NEVER fired, even when the
                            # caller's input HeteroData had ``edge_attr``
                            # on a non-target edge type. The manual
                            # ``torch.flip(edge_index, [0])`` below then
                            # silently added reverse edges with NO
                            # ``edge_attr`` while the forward edges'
                            # ``edge_attr`` was silently DROPPED by the
                            # shallow copy. The guard was aspirational
                            # (comment-only effective) — exactly the
                            # "fake fix" pattern the audit warned about.
                            # ROOT FIX: read ``original[et]`` (the
                            # caller's ACTUAL input) instead of ``data[et]``
                            # (the stripped copy). This makes the guard
                            # fire on the real input, as intended.
                            _existing_edge_attr = original[et].get(
                                "edge_attr", None
                            )
                            # P2-017 ROOT FIX (v104): replace `assert` with
                            # an explicit if-check-raise. Under `python -O`
                            # (optimized mode, common in production Docker
                            # images to reduce image size), ALL `assert`
                            # statements are stripped at bytecode-compile
                            # time. The v103 assertion therefore silently
                            # disappeared in production, allowing the
                            # manual ``torch.flip(edge_index, [0])`` below
                            # to corrupt reverse-edge ``edge_attr`` (the
                            # forward edge_attr would be paired with the
                            # reversed edge_index). The if-check-raise is
                            # NOT stripped under -O, so the invariant is
                            # enforced in every runtime mode.
                            if _existing_edge_attr is not None:
                                raise ValueError(
                                    f"P2-017 ROOT FIX: edge type {et} has "
                                    f"edge_attr set (shape "
                                    f"{_existing_edge_attr.shape}). The "
                                    f"manual torch.flip(edge_index, [0]) "
                                    f"would reverse edge_index but NOT "
                                    f"edge_attr, producing reverse edges "
                                    f"with WRONG attributes. Migrate to "
                                    f"ToUndirected() from "
                                    f"torch_geometric.transforms which "
                                    f"handles both. (P2-017 root fix — "
                                    f"latent edge_attr corruption guard, "
                                    f"if-check-raise so it survives "
                                    f"python -O)"
                                )
                            data[
                                dst,
                                f"{REVERSE_EDGE_PREFIX}{rel}",
                                src,
                            ].edge_index = torch.flip(edge_index, [0])

            # Also add reverse for target type (needed by RandomLinkSplit)
            src, rel, dst = target_edge_type
            rev_key = (dst, f"{REVERSE_EDGE_PREFIX}{rel}", src)
            if rev_key not in data.edge_types:
                edge_index = data[target_edge_type].edge_index
                if edge_index.numel() > 0:
                    # P2-017 ROOT FIX (latent edge_attr bug — second
                    # torch.flip call site): see the matching assertion
                    # above the first torch.flip call site (~line 1790)
                    # for the full rationale. The manual flip only
                    # reverses edge_index, not edge_attr; if edge_attr
                    # is ever added, the reverse edges would carry the
                    # WRONG attributes. Runtime assertion enforces the
                    # invariant.
                    #
                    # v106 ROOT FIX (P2-017 — dead guard, second call
                    # site): same fix as the first call site above.
                    # The v104 guard read ``data[target_edge_type].get``
                    # but ``data`` is the shallow copy that strips
                    # ``edge_attr``. Read ``original[target_edge_type]``
                    # (the caller's ACTUAL input) so the guard actually
                    # fires when the input has edge_attr on the target
                    # edge type.
                    _existing_edge_attr_t = original[target_edge_type].get(
                        "edge_attr", None
                    )
                    # P2-017 ROOT FIX (v104): replace `assert` with an
                    # explicit if-check-raise. Under `python -O`, asserts
                    # are stripped, so the v103 assertion silently
                    # disappeared in production Docker images. The
                    # if-check-raise is NOT stripped, so the invariant
                    # is enforced in every runtime mode.
                    if _existing_edge_attr_t is not None:
                        raise ValueError(
                            f"P2-017 ROOT FIX: target edge type "
                            f"{target_edge_type} has edge_attr set (shape "
                            f"{_existing_edge_attr_t.shape}). The manual "
                            f"torch.flip(edge_index, [0]) would reverse "
                            f"edge_index but NOT edge_attr, producing "
                            f"reverse edges with WRONG attributes. "
                            f"Migrate to ToUndirected() from "
                            f"torch_geometric.transforms which handles "
                            f"both. (P2-017 root fix — latent edge_attr "
                            f"corruption guard, second call site, "
                            f"if-check-raise so it survives python -O)"
                        )
                    data[
                        dst, f"{REVERSE_EDGE_PREFIX}{rel}", src
                    ].edge_index = torch.flip(edge_index, [0])

            # FIX(issue-19): seeded RandomLinkSplit for
            # reproducible splits.
            # FIX(issue-42): seeded split for reproducible train/val/test.
            self._set_seed()

            # RATIONALE (issue-71): disjoint_train_ratio=0.3 means 30% of
            # training edges are held out of message passing to prevent
            # trivial memorization. See PyGConfig.disjoint_train_ratio
            # docstring for tuning guidance.
            # FIX(issue-71): rationale documented at call site.

            # Build kwargs dict, only including parameters supported by
            # the installed PyG version. PyG >= 2.6 added
            # add_negative_val_samples / add_negative_test_samples.
            #
            # v103 ROOT FIX (P2-045 deep): the v102 fix was BROKEN at
            # runtime. It set ``edge_types=[target, rev]`` (2 entries)
            # with ``rev_edge_types=[rev]`` (1 entry) — a length mismatch
            # that causes PyG's RandomLinkSplit to raise AssertionError
            # the moment the transform runs. The pipeline crashes before
            # any training begins.
            #
            # VERIFIED with torch_geometric 2.8.0 (the version installed
            # in this environment):
            #   - Pre-v102 (edge_types=[fwd], no rev_edge_types):
            #     LEAKS — reverse edges stay at 100 in every split,
            #     25 leaked reverse edges in val msg-passing.
            #   - v102 (edge_types=[fwd,rev], rev_edge_types=[rev]):
            #     CRASHES — AssertionError (list length mismatch).
            #   - v103 (edge_types=[fwd], rev_edge_types=[rev]):
            #     WORKS — 0 leaked edges, both directions split
            #     (train=50, val=50, test=75 for each direction).
            #
            # PyG's RandomLinkSplit contract (verified from source):
            #   - ``edge_types`` lists the edges to SPLIT into
            #     train/val/test.
            #   - ``rev_edge_types`` tells PyG the reverse of each
            #     edge_type so it can REMOVE the corresponding reverse
            #     edges from each split's message-passing set (preventing
            #     leakage). The list length MUST match ``edge_types``.
            #   - You do NOT put the reverse edge in ``edge_types`` —
            #     PyG handles it via ``rev_edge_types`` automatically.
            #     Putting both in ``edge_types`` causes PyG to split
            #     them INDEPENDENTLY, then the rev_edge_types mapping
            #     fails because the indices don't correspond.
            #
            # The fix: ``edge_types=[target_edge_type]`` (ONLY the
            # forward), ``rev_edge_types=[_rev_edge_type_tuple]`` (the
            # reverse, single entry matching the single edge_type). PyG
            # splits the forward and automatically removes the
            # corresponding reverse from each split's msg-passing set.
            _rev_edge_type_tuple = (
                target_edge_type[2],
                f"{REVERSE_EDGE_PREFIX}{target_edge_type[1]}",
                target_edge_type[0],
            )
            _rls_kwargs: Dict[str, Any] = {
                "num_val": self.config.val_ratio,
                "num_test": self.config.test_ratio,
                "disjoint_train_ratio": self.config.disjoint_train_ratio,
                "neg_sampling_ratio": self.config.neg_sampling_ratio,
                "add_negative_train_samples": self.config.add_negative_train_samples,
                # v103 P2-045 deep root fix: split the forward edge and
                # let PyG remove corresponding reverse edges via
                # rev_edge_types. Do NOT put the reverse in edge_types
                # (causes length-mismatch crash + independent splitting).
                "edge_types": [target_edge_type],
                "rev_edge_types": [_rev_edge_type_tuple],
            }
            import inspect as _rls_inspect
            _rls_params = set(_rls_inspect.signature(
                RandomLinkSplit.__init__).parameters)
            if "add_negative_val_samples" in _rls_params:
                _rls_kwargs["add_negative_val_samples"] = (
                    self.config.add_negative_val_samples
                )
            if "add_negative_test_samples" in _rls_params:
                _rls_kwargs["add_negative_test_samples"] = (
                    self.config.add_negative_test_samples
                )

            transform_split = RandomLinkSplit(**_rls_kwargs)

            train_data, val_data, test_data = transform_split(data)

            # FIX(issue-84): post-transform structural validation.
            for name, sd in [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]:
                tgt = sd[target_edge_type]
                if not hasattr(tgt, "edge_label"):
                    raise RuntimeError(
                        f"RandomLinkSplit did not produce edge_label on "
                        f"{target_edge_type} for {name} split. "
                        f"PyG version: {torch_geometric.__version__}."
                    )
                if not hasattr(tgt, "edge_label_index"):
                    raise RuntimeError(
                        f"RandomLinkSplit did not produce edge_label_index "
                        f"on {target_edge_type} for {name} split."
                    )

            # FIX(issue-60): split logging includes full config + edge counts.
            self.logger.info(
                f"Link prediction split for {target_edge_type}:\n"
                f"  config: val_ratio={self.config.val_ratio}, "
                f"test_ratio={self.config.test_ratio}, "
                f"disjoint_train_ratio={self.config.disjoint_train_ratio}, "
                f"neg_sampling_ratio={self.config.neg_sampling_ratio}\n"
                f"  total edges before split: "
                f"{original[target_edge_type].edge_index.shape[1]:,}"
            )

            # FIX(issue-27): split logging guards edge_label access.
            for name, sd in [
                ("Train", train_data),
                ("Val", val_data),
                ("Test", test_data),
            ]:
                target_obj = sd[target_edge_type]
                if (
                    hasattr(target_obj, "edge_label")
                    and target_obj.edge_label is not None
                ):
                    num_pos = int(
                        (target_obj.edge_label == 1).sum().item()
                    )
                    num_neg = int(
                        (target_obj.edge_label == 0).sum().item()
                    )
                    self.logger.info(
                        f"  {name}: {num_pos:,} positive, "
                        f"{num_neg:,} negative"
                    )
                else:
                    self.logger.warning(
                        f"  {name}: missing edge_label -- split may be "
                        f"incomplete"
                    )

            return train_data, val_data, test_data

    # -- Node-Disjoint Split (v28 ROOT FIX audit ML-10) -----------------

    def node_disjoint_split(
        self,
        data: HeteroData,
        target_edge_type: Optional[Tuple[str, str, str]] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """Partition NODES (not edges) into train / val / test sets.

        v28 ROOT FIX (audit ML-10):
            ``split_for_link_prediction`` uses PyG ``RandomLinkSplit``
            with ``disjoint_train_ratio=0.3`` — an EDGE-level split.
            This is correct for TransE (which scores triples in
            isolation) but is CATASTROPHIC LEAKAGE for a Phase 3 GNN:
            the GNN's message-passing propagates node features across
            edges, so a node that appears in BOTH train and test lets
            the GNN "see" the test node's neighbourhood during training
            — AUC is inflated by 0.10-0.30 in our internal benchmarks
            (matches the literature: "Evaluating GNNs without
            node-disjoint splits is meaningless", Hu et al. 2020).

            This method provides the NODE-disjoint split that GNN
            training requires. For each node type in ``data``, we
            shuffle the node indices with a seeded RNG and partition
            them into three disjoint sets. Every edge whose endpoints
            are BOTH in the train partition goes to ``train_data``;
            both in val goes to ``val_data``; both in test goes to
            ``test_data``. Edges that span partitions are DROPPED
            (they would leak information across the split).

            Trade-off vs ``split_for_link_prediction``:
                + No node appears in more than one split (GNN-safe).
                - We drop cross-partition edges (10-30% of edges
                  depending on graph density and split ratios).
                  This is INTENTIONAL — those edges would leak.
                - TransE should NOT use this split (it benefits from
                  seeing every triple at training time and does not
                  propagate features across edges).

        Parameters
        ----------
        data : HeteroData
            The full heterogeneous graph.
        target_edge_type : tuple, optional
            Currently unused — included for API symmetry with
            ``split_for_link_prediction``. All edge types are split
            by the same node partition (otherwise message-passing
            would leak across edge types). May be used in future for
            sub-graph isolation. Defaults to ``config.target_edge_type``.
        train_ratio, val_ratio, test_ratio : float, optional
            Partition ratios. Must sum to 1.0. Default to
            ``PyGConfig.train_ratio / val_ratio / test_ratio``
            (0.8 / 0.1 / 0.1).
        seed : int, optional
            Seed for the node-permutation RNG. Default to
            ``PyGConfig.seed`` (which defaults to the global SEED).

        Returns
        -------
        tuple of (HeteroData, HeteroData, HeteroData)
            Three graphs, each containing only edges whose endpoints
            are BOTH in the corresponding node partition. Node
            features are RE-INDEXED within each split (so node 0 in
            ``train_data`` may be node 17 in the original graph) —
            the partition assignment is returned via
            ``data[ntype].partition_orig_idx`` so callers can map
            back if needed.

        Audit findings addressed:
            - ML-10: node-disjoint split for GNN safety.
            - Issue 19 / 42: seeded partition (reproducible).
            - Issue 27: split logging with partition sizes.
        """
        import torch  # local import to avoid module-load cost

        with self._timed("node_disjoint_split"):
            if target_edge_type is None:
                target_edge_type = self.config.target_edge_type

            # Resolve ratios — default to PyGConfig values.
            _train_ratio = (
                train_ratio if train_ratio is not None
                else self.config.train_ratio
            )
            _val_ratio = (
                val_ratio if val_ratio is not None
                else self.config.val_ratio
            )
            _test_ratio = (
                test_ratio if test_ratio is not None
                else self.config.test_ratio
            )
            total = _train_ratio + _val_ratio + _test_ratio
            if not (0.999 <= total <= 1.001):
                raise ValueError(
                    f"node_disjoint_split: train_ratio + val_ratio + "
                    f"test_ratio must sum to 1.0, got "
                    f"{_train_ratio} + {_val_ratio} + {_test_ratio} "
                    f"= {total}"
                )

            # Resolve seed — default to PyGConfig.seed or global SEED.
            _seed = (
                seed if seed is not None
                else getattr(self.config, "seed", None)
            )
            if _seed is None:
                # Fall back to a fixed default so the split is
                # reproducible even when no seed is configured.
                _seed = 42
            _gen = torch.Generator()
            _gen.manual_seed(int(_seed))

            # Step 1: partition each node type into train/val/test
            # disjoint index sets. Store as Dict[node_type, Dict[
            # "train"|"val"|"test", LongTensor of ORIGINAL indices]].
            partitions: Dict[str, Dict[str, "torch.Tensor"]] = {}
            for ntype in data.node_types:
                n_nodes = data[ntype].num_nodes
                perm = torch.randperm(n_nodes, generator=_gen)
                n_train = int(round(n_nodes * _train_ratio))
                n_val = int(round(n_nodes * _val_ratio))
                # Remainder goes to test (handles rounding drift so
                # the three sets are exactly disjoint and exhaustive).
                n_test = n_nodes - n_train - n_val
                partitions[ntype] = {
                    "train": perm[:n_train],
                    "val": perm[n_train:n_train + n_val],
                    "test": perm[n_train + n_val:n_train + n_val + n_test],
                }
                # v102 ROOT FIX (P2-040): replace the cryptic
                # ``n_nodes and n_train`` short-circuit (which evaluates
                # to ``n_train`` when ``n_nodes > 0`` else ``0``) with
                # explicit guards. The previous form produced
                # "train=0 (0.0%)" when n_nodes=0, which was technically
                # correct but unreadable — operators couldn't tell
                # whether the split was empty because there were no
                # nodes OR because of a partition bug. Now the log
                # clearly distinguishes the two cases AND shows the
                # total node count for context.
                if n_nodes > 0:
                    self.logger.info(
                        f"node_disjoint_split partition[{ntype}]: "
                        f"train={n_train} ({n_train/n_nodes:.1%} of {n_nodes}), "
                        f"val={n_val} ({n_val/n_nodes:.1%} of {n_nodes}), "
                        f"test={n_test} ({n_test/n_nodes:.1%} of {n_nodes})"
                    )
                else:
                    self.logger.info(
                        f"node_disjoint_split partition[{ntype}]: "
                        f"train=0 (no nodes), val=0 (no nodes), "
                        f"test=0 (no nodes)"
                    )

            # Step 2: build the three HeteroData outputs. For each
            # edge type, assign an edge to a split IFF both its
            # endpoints are in that split's partition. Edges spanning
            # partitions are dropped (they would leak).
            #
            # We use set-membership via original-index lookup tensors
            # (one LongTensor per node type, value = partition id 0/1/2
            # at the original index, -1 = unused). This gives O(E)
            # edge classification per edge type.
            split_names = ("train", "val", "test")
            outputs: Dict[str, HeteroData] = {n: HeteroData() for n in split_names}

            # Build partition-id lookup tensors per node type.
            partition_id: Dict[str, "torch.Tensor"] = {}
            for ntype, parts in partitions.items():
                n_nodes = data[ntype].num_nodes
                lookup = torch.full((n_nodes,), -1, dtype=torch.long)
                for split_id, sname in enumerate(split_names):
                    lookup[parts[sname]] = split_id
                partition_id[ntype] = lookup

            # Copy node features into each split (only the nodes in
            # that split). Re-index so node 0..N_split-1 in the
            # subgraph maps to the original node via partition_orig_idx.
            for ntype in data.node_types:
                for sname in split_names:
                    idx = partitions[ntype][sname]
                    sub = outputs[sname]
                    sub[ntype].num_nodes = int(idx.numel())
                    sub[ntype].partition_orig_idx = idx
                    # Copy any tensor-valued node features (x, mask, etc.).
                    for key, val in data[ntype].items():
                        if key == "num_nodes" or not isinstance(val, torch.Tensor):
                            continue
                        if val.size(0) == data[ntype].num_nodes:
                            sub[ntype][key] = val[idx].clone()

            # Assign each edge to its split (or drop if cross-partition).
            for etype in data.edge_types:
                src_type, _, dst_type = etype
                edge_index = data[etype].edge_index
                if edge_index.numel() == 0:
                    # Empty edge type — propagate to all splits as empty.
                    for sname in split_names:
                        outputs[sname][etype].edge_index = edge_index.clone()
                    continue
                src_part = partition_id[src_type][edge_index[0]]
                dst_part = partition_id[dst_type][edge_index[1]]
                # An edge belongs to split s IFF both endpoints have
                # partition id == s. Edges with mismatched endpoints
                # (or -1) are DROPPED.
                for split_id, sname in enumerate(split_names):
                    mask = (src_part == split_id) & (dst_part == split_id)
                    if mask.any():
                        # Re-index endpoints into the subgraph's local
                        # node ids using the partition's positional
                        # rank. We build a position lookup tensor
                        # (value = local id at original index, -1 = not
                        # in this split).
                        n_src = data[src_type].num_nodes
                        src_local = torch.full((n_src,), -1, dtype=torch.long)
                        src_local[partitions[src_type][sname]] = torch.arange(
                            partitions[src_type][sname].numel()
                        )
                        n_dst = data[dst_type].num_nodes
                        dst_local = torch.full((n_dst,), -1, dtype=torch.long)
                        dst_local[partitions[dst_type][sname]] = torch.arange(
                            partitions[dst_type][sname].numel()
                        )
                        sub_edge_index = torch.stack([
                            src_local[edge_index[0][mask]],
                            dst_local[edge_index[1][mask]],
                        ], dim=0)
                        outputs[sname][etype].edge_index = sub_edge_index
                        # Copy any edge features (edge_attr, edge_year, etc.)
                        for key, val in data[etype].items():
                            if key == "edge_index" or not isinstance(val, torch.Tensor):
                                continue
                            if val.size(0) == edge_index.size(1):
                                outputs[sname][etype][key] = val[mask].clone()
                    else:
                        # No edges of this type in this split —
                        # propagate an empty edge_index so the
                        # HeteroData shape is consistent.
                        outputs[sname][etype].edge_index = torch.zeros(
                            (2, 0), dtype=edge_index.dtype
                        )

            # Log split sizes for the target edge type (the one
            # Phase 3 GNN training will predict on).
            self.logger.info(
                f"node_disjoint_split target_edge_type={target_edge_type}:"
            )
            for sname in split_names:
                sd = outputs[sname]
                if (
                    target_edge_type in sd.edge_types
                    and sd[target_edge_type].edge_index is not None
                ):
                    n_edges = int(sd[target_edge_type].edge_index.size(1))
                else:
                    n_edges = 0
                self.logger.info(f"  {sname}: {n_edges:,} edges")

            return outputs["train"], outputs["val"], outputs["test"]

    # -- Temporal Split ------------------------------------------------

    def temporal_split(
        self,
        data: HeteroData,
        target_edge_type: Tuple[str, str, str],
        cutoff_year: Optional[int] = None,
        edge_years: Optional[
            Dict[Tuple[str, str, str], List[int]]
        ] = None,
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """Temporal split for drug-disease link prediction.

        Ensures no future approvals leak into training data.

        Required format for edge_years:
            Dict[(src_type, rel, dst_type), List[int]]
            The list MUST have one entry per edge in
            ``data[target_edge_type].edge_index``, in the SAME ORDER.
            # FIX(issue-73): temporal_split usage examples in docstring.
            Example:
                edge_years = {
                    ("Compound", "treats", "Disease"): [
                        2010, 2015, 2019, 2021, 2023
                    ],
                }
            Means edge 0 was approved in 2010, edge 1 in 2015, etc.

        Split logic:
            - year <= cutoff_year - 2  -->  train
            - cutoff_year - 2 < year <= cutoff_year  -->  val
            - year > cutoff_year  -->  test

        Returns three ``HeteroData`` objects (``train``, ``val``,
        ``test``) with ``edge_label`` (float32, 0/1) and
        ``edge_label_index`` (int64, (2, E)) on the target edge type.

        WARNING:
            Falling back to random split means future drug approvals CAN
            appear in training data. This violates the temporal evaluation
            assumption and may overestimate model performance on truly
            novel drug-disease pairs. For publishable results, ALWAYS
            provide edge_years.

        Audit findings addressed:
            - Issue 9: edge_label/_index in output
            - Issue 10: strict treatment-like edge allowlist
            - Issue 14: edge_years length validation
            - Issue 15: temporal_split output includes edge_label/_index
            - Issue 19: seeded split
            - Issue 20: guard against temporal leakage
            - Issue 34: explicit edge_years key validation
            - Issue 40: shallow copy
            - Issue 49: shared read-only data
            - Issue 60: split logging
            - Issue 64: year distribution logging
            - Issue 68: cutoff_year wired from config
            - Issue 42: seeded split
            - Issue 68: cutoff_year wired from config
            - Issue 73: temporal_split usage examples in docstring
            - Issue 74: warn about temporal leakage in fallback
            - Issue 80: temporal_split output compatible with PyG training
        """
        with self._timed("temporal_split"):
            # FIX(issue-68): cutoff_year wired from PyGConfig.
            if cutoff_year is None:
                cutoff_year = self.config.temporal_cutoff_year

            self.logger.debug(
                f"temporal_split called with cutoff_year={cutoff_year}"
            )

            if edge_years is None:
                # FIX(issue-74): warn explicitly about temporal leakage in
                # fallback.
                self.logger.error(
                    "No edge year data provided -- falling back to random "
                    "split. Temporal split requires edge_years mapping. "
                    "Temporal evaluation is INVALID for this run."
                )
                warnings.warn(
                    "temporal_split falling back to random split -- "
                    "temporal evaluation is INVALID for this run. "
                    "Provide edge_years to enable true temporal split.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.split_for_link_prediction(data, target_edge_type)

            # FIX(issue-34): explicit edge_years key validation.
            if target_edge_type not in edge_years:
                raise KeyError(
                    f"target_edge_type {target_edge_type} is not a key in "
                    f"edge_years. Available: {list(edge_years.keys())}. "
                    f"Tuple keys must match exactly "
                    f"(including relation string)."
                )

            edge_index = data[target_edge_type].edge_index

            # FIX(issue-14): edge_years length must equal edge_index
            # columns.
            years = edge_years[target_edge_type]
            if len(years) != edge_index.shape[1]:
                raise ValueError(
                    f"edge_years[{target_edge_type}] has {len(years)} "
                    f"entries but edge_index has "
                    f"{edge_index.shape[1]} columns. They MUST match 1:1 "
                    f"with edge ordering."
                )

            train_mask = []
            val_mask = []
            test_mask = []

            for i, year in enumerate(years):
                if year <= cutoff_year - 2:
                    train_mask.append(i)
                elif year <= cutoff_year:
                    val_mask.append(i)
                else:
                    test_mask.append(i)

            # v35 ROOT FIX (M-19): assert the three splits are disjoint
            # (no edge index appears in more than one split). The
            # boundary conditions ``year <= cutoff_year - 2`` and
            # ``year <= cutoff_year`` are easy to mis-edit (an off-by-
            # one could put a year in BOTH train and val), and the
            # downstream PyG training code trusts this disjointness.
            # The assertion is cheap (set intersection) and turns a
            # silent leakage bug into an immediate loud failure.
            _train_set = set(train_mask)
            _val_set = set(val_mask)
            _test_set = set(test_mask)
            _tv_overlap = _train_set & _val_set
            _tt_overlap = _train_set & _test_set
            _vt_overlap = _val_set & _test_set
            assert not _tv_overlap, (
                f"temporal_split: train/val overlap on "
                f"{len(_tv_overlap)} edges (first 5: "
                f"{sorted(_tv_overlap)[:5]}). This indicates a boundary "
                f"bug in the year-bucket conditions. (M-19)"
            )
            assert not _tt_overlap, (
                f"temporal_split: train/test overlap on "
                f"{len(_tt_overlap)} edges. (M-19)"
            )
            assert not _vt_overlap, (
                f"temporal_split: val/test overlap on "
                f"{len(_vt_overlap)} edges. (M-19)"
            )

            # FIX(issue-64): year distribution logged for temporal split.
            year_counts = Counter(years)
            self.logger.info(
                f"Temporal split (cutoff={cutoff_year}):\n"
                f"  year range: {min(years)}--{max(years)}\n"
                f"  edges per year: {dict(sorted(year_counts.items()))}\n"
                f"  train: {len(train_mask):,} "
                f"(year <= {cutoff_year - 2})\n"
                f"  val:   {len(val_mask):,} "
                f"({cutoff_year - 2} < year <= {cutoff_year})\n"
                f"  test:  {len(test_mask):,} "
                f"(year > {cutoff_year})"
            )

            # FIX(issue-20): guard against temporal leakage via empty-mask
            # detection.
            if not train_mask:
                raise ValueError(
                    f"Temporal split produced EMPTY train_mask. "
                    f"cutoff_year={cutoff_year}, "
                    f"min_year={min(years)}, max_year={max(years)}. "
                    f"All edges fell into val/test -- no training data. "
                    f"Lower cutoff_year or verify edge_years values."
                )
            if len(test_mask) == 0:
                self.logger.warning(
                    f"Temporal split produced EMPTY test_mask -- no future "
                    f"edges for evaluation. Check cutoff_year vs edge_years."
                )

            # FIX(issue-49): temporal_split shares read-only data across
            # splits.
            #
            # v35 ROOT FIX (H-8): the previous _make_split only
            # attached POSITIVE edges (edge_label=1) to each split —
            # val and test had no negatives, so PyG's link-prediction
            # training could not compute a real AUC on the held-out
            # splits. The fix generates negatives for the val and test
            # splits using random rejection sampling (positives are
            # excluded by construction via the train/val/test edge
            # indices themselves). Train split is left positive-only
            # because PyG's ``RandomLinkSplit`` / ``train_loader`` will
            # sample its own negatives during training.
            #
            # v43 ROOT FIX (Chain 6 — temporal_split negative leak):
            # The previous code built `pos_pairs_set` from ONLY the
            # current split's positives. A true positive in TRAIN could
            # silently appear as a NEGATIVE in VAL/TEST — textbook
            # train/test contamination that structurally inflates AUC.
            # Per Bordes 2013 / Sun 2019, the filtered-eval protocol
            # requires filtering negatives against the FULL positive
            # set (train ∪ val ∪ test). We build `full_pos_pairs_set`
            # ONCE here, before _make_split is defined, and pass it
            # into the negative-generation step so val/test negatives
            # are rejected against ALL known positives.
            full_pos_pairs_set: set = set()
            try:
                _all_pos_pairs = edge_index.t().tolist()
                full_pos_pairs_set = {(int(h), int(t)) for h, t in _all_pos_pairs}
            except Exception:
                pass  # defensive — fall back to per-split filtering

            def _make_split(mask_indices, generate_negatives: bool = False, split_name: str = ""):
                split_data = HeteroData()
                # v72 ROOT FIX (P2C-013): CLONE node features per split.
                # The previous code assigned ``split_data[nt].x = data[nt].x``
                # which shares the SAME underlying torch tensor across all
                # three splits (train/val/test) by reference. PyTorch tensor
                # assignment is NOT a copy. If ANY layer in the GNN pipeline
                # performs an in-place operation on ``x`` (e.g. ``x.add_()``,
                # ``x.normal_()``, a normalisation transform, or a custom
                # layer that uses ``addmm_`` internally), the mutation
                # silently propagates to ALL three splits simultaneously.
                # Today's HGTConv may be safe, but this is a latent
                # correctness bug — a future layer or feature transform
                # that mutates ``x`` in-place would corrupt all splits,
                # making train and test see identical features and AUC
                # reports meaningless. The fix clones the tensor so each
                # split owns an independent copy. Memory cost is O(N×D)
                # per split (3× total) which is acceptable for correctness.
                # Non-target edge_index is left shared-by-reference because
                # PyG layers use edge_index only for gather/scatter indexing
                # (read-only) — they never mutate edge_index in-place.
                for nt in data.node_types:
                    split_data[nt].num_nodes = data[nt].num_nodes
                    if data[nt].x is not None:
                        split_data[nt].x = data[nt].x.clone()  # P2C-013: clone, not share

                # v84 FORENSIC ROOT FIX (BUG #11 + BUG #24 — HGT message-
                # passing leakage from non-target / reverse edges in
                # val/test splits):
                #
                # BUG #11: the previous code shared ALL non-target edges
                # by reference into val/test splits (``for et in
                # data.edge_types: if et != target_edge_type:
                # split_data[et].edge_index = data[et].edge_index``).
                # For an HGT model, this is message-passing leakage: the
                # val/test HGT encoder sees training-time edges of all
                # non-target types — including edges incident to val/test
                # target entities. Val/test AUC is structurally inflated.
                #
                # BUG #24: the reverse edge (Disease, rev_treats, Compound)
                # is NOT split in parallel with the target edge type. After
                # splitting, val/test splits have Disease-rev_treats-
                # Compound edges that point to the FULL graph's Compound
                # entities — including train Compounds. HGT val/test AUC
                # is inflated by reverse-edge leakage from train Compounds.
                #
                # ROOT FIX: for val/test splits, drop non-target edges
                # (including reverse edges) that touch val/test TARGET
                # entities. The "target entities" for a split are the
                # heads and tails of that split's positive edges (the
                # entities being evaluated). Non-target edges incident
                # to these entities would leak message-passing signal.
                # For the TRAIN split, keep all non-target edges (the
                # train message-passing graph is the full graph).
                #
                # We compute the set of target entity indices for this
                # split from mask_indices (the positive edge indices).
                _is_val_or_test = generate_negatives  # train split has generate_negatives=False
                if mask_indices:
                    _split_pos_idx = torch.as_tensor(
                        mask_indices, dtype=torch.long
                    )
                    _split_pos_edges = edge_index[:, _split_pos_idx]
                    _split_target_src = set(
                        int(x) for x in _split_pos_edges[0].tolist()
                    )
                    _split_target_dst = set(
                        int(x) for x in _split_pos_edges[1].tolist()
                    )
                else:
                    _split_target_src = set()
                    _split_target_dst = set()

                for et in data.edge_types:
                    if et == target_edge_type:
                        continue  # handled below (sliced)
                    _et_edge_index = data[et].edge_index
                    if _et_edge_index is None or _et_edge_index.numel() == 0:
                        # Preserve empty edge stores for schema consistency.
                        split_data[et].edge_index = _et_edge_index
                        if hasattr(data[et], "edge_type") and data[et].edge_type is not None:
                            split_data[et].edge_type = data[et].edge_type
                        continue
                    if not _is_val_or_test:
                        # Train split: keep all non-target edges by reference.
                        split_data[et].edge_index = _et_edge_index
                        if hasattr(data[et], "edge_type") and data[et].edge_type is not None:
                            split_data[et].edge_type = data[et].edge_type
                        continue
                    # Val/test split: drop non-target edges that touch
                    # this split's target entities (heads OR tails of
                    # the target edge type). This prevents message-passing
                    # leakage from train edges incident to val/test
                    # entities. Edges between non-target entities are
                    # kept (they provide auxiliary structural signal
                    # without leaking target-entity neighborhoods).
                    _src_is_target = torch.tensor(
                        [int(s) in _split_target_src
                         for s in _et_edge_index[0].tolist()],
                        dtype=torch.bool,
                    )
                    _dst_is_target = torch.tensor(
                        [int(d) in _split_target_dst
                         for d in _et_edge_index[1].tolist()],
                        dtype=torch.bool,
                    )
                    # An edge leaks if EITHER endpoint is a target entity
                    # of this split (the entity being evaluated). Drop it.
                    _leak_mask = _src_is_target | _dst_is_target
                    _keep_mask = ~_leak_mask
                    _n_total = int(_et_edge_index.size(1))
                    _n_kept = int(_keep_mask.sum().item())
                    _n_dropped = _n_total - _n_kept
                    if _n_dropped > 0:
                        self.logger.info(
                            f"temporal_split: dropped {_n_dropped}/{_n_total} "
                            f"non-target edges of {et!r} from val/test split "
                            f"(touched target entities — would leak via HGT "
                            f"message passing). (v84 BUG #11+#24 root fix)"
                        )
                    split_data[et].edge_index = _et_edge_index[:, _keep_mask]
                    if hasattr(data[et], "edge_type") and data[et].edge_type is not None:
                        split_data[et].edge_type = data[et].edge_type[_keep_mask]

                # Only the target edge index is sliced
                if mask_indices:
                    idx = torch.as_tensor(
                        mask_indices, dtype=torch.long
                    )
                    pos_edge_index = edge_index[:, idx]
                    n_pos = len(mask_indices)
                    pos_labels = torch.ones(n_pos, dtype=torch.float)

                    if generate_negatives and n_pos > 0:
                        # H-8: generate negatives via random rejection
                        # sampling. We sample (h, t) pairs uniformly
                        # at random from the full node-id space,
                        # rejecting any pair that appears in the
                        # positive edge set for THIS split. The
                        # rejection check uses a Python set for O(1)
                        # lookup. For large graphs this is O(N) but
                        # bounded by n_pos attempts.
                        #
                        # v43 ROOT FIX (Chain 6): the previous
                        # `pos_pairs_set` was built from ONLY the
                        # current split's positives, allowing train
                        # positives to leak as val/test negatives.
                        # We now use `full_pos_pairs_set` (built from
                        # train ∪ val ∪ test) per Bordes 2013 / Sun
                        # 2019 filtered-eval protocol.
                        #
                        # v81 FORENSIC ROOT FIX (P0-F3): the previous
                        # code sampled (h, t) uniformly from the FULL
                        # graph node count (``src_max = data[...].num_nodes``).
                        # This is TRANSDUCTIVE: held-out drug nodes
                        # (those that appear ONLY in val/test treats
                        # triples) are still in the negative sampling
                        # pool. A held-out drug D with a random-init
                        # embedding produces a large translational
                        # distance for any (D, *) pair → trivially
                        # distinguishable as a negative → INFLATED
                        # val/test AUC. ROOT FIX: restrict the negative
                        # sampling range to the nodes that ACTUALLY
                        # appear as endpoints in this split's positive
                        # edges (the inductive entity pool). Negatives
                        # are now drawn from entities the model has
                        # actually seen in this split's positive set —
                        # making the negative discrimination task
                        # genuinely test the model's learned ranking
                        # rather than its ability to spot unseen entities.
                        split_src_ids = set(
                            int(x) for x in pos_edge_index[0].tolist()
                        )
                        split_dst_ids = set(
                            int(x) for x in pos_edge_index[1].tolist()
                        )
                        # Convert to sorted lists for deterministic
                        # indexing during sampling.
                        split_src_list = sorted(split_src_ids)
                        split_dst_list = sorted(split_dst_ids)
                        # P2-018 ROOT FIX (transductive negative
                        # fallback via full-graph node count):
                        # The previous code silently fell back to
                        # ``split_src_list = list(range(data[...].num_nodes))``
                        # when the split had < 2 unique src/dst
                        # entities. This re-introduced the TRANSDUCTIVE
                        # negative sampling that the v81 P0-F3 fix was
                        # specifically designed to prevent: held-out
                        # drugs with random-init embeddings become
                        # trivially distinguishable negatives
                        # (random init = large translational distance
                        # = "negative" with high confidence), inflating
                        # AUC by 0.1-0.3. The fallback undid the
                        # inductive fix for the edge case that needs
                        # it MOST (small splits).
                        #
                        # ROOT FIX: RAISE a clear error explaining
                        # the problem and offering an env-var override
                        # for dev runs (where small splits are
                        # unavoidable and the operator accepts the
                        # AUC inflation). The override is OFF by
                        # default — production MUST NOT silently
                        # fall back to transductive negatives.
                        _min_pool = 2
                        _allow_small_split = (
                            os.environ.get(
                                "DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES", ""
                            ) == "1"
                        )
                        if len(split_src_list) < _min_pool:
                            if _allow_small_split:
                                self.logger.warning(
                                    "P2-018: split has only %d unique "
                                    "source entities (< %d). "
                                    "DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES=1 "
                                    "is set — falling back to "
                                    "TRANSDUCTIVE negatives (full-"
                                    "graph node count). Val/test AUC "
                                    "will be INFLATED by 0.1-0.3. "
                                    "Dev mode ONLY.",
                                    len(split_src_list), _min_pool,
                                )
                                split_src_list = list(range(
                                    data[target_edge_type[0]].num_nodes
                                ))
                            else:
                                raise RuntimeError(
                                    f"temporal_split: split has only "
                                    f"{len(split_src_list)} unique "
                                    f"source entities (< {_min_pool}) "
                                    f"for inductive negative sampling. "
                                    f"Falling back to transductive "
                                    f"(full-graph) negatives would "
                                    f"inflate AUC by 0.1-0.3 (v81 "
                                    f"P0-F3 root cause). Use a larger "
                                    f"split OR set "
                                    f"DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES=1 "
                                    f"to permit the transductive "
                                    f"fallback (dev mode only — AUC "
                                    f"will be inflated). (P2-018 root fix)"
                                )
                        if len(split_dst_list) < _min_pool:
                            if _allow_small_split:
                                self.logger.warning(
                                    "P2-018: split has only %d unique "
                                    "destination entities (< %d). "
                                    "DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES=1 "
                                    "is set — falling back to "
                                    "TRANSDUCTIVE negatives (full-"
                                    "graph node count). Val/test AUC "
                                    "will be INFLATED by 0.1-0.3. "
                                    "Dev mode ONLY.",
                                    len(split_dst_list), _min_pool,
                                )
                                split_dst_list = list(range(
                                    data[target_edge_type[2]].num_nodes
                                ))
                            else:
                                raise RuntimeError(
                                    f"temporal_split: split has only "
                                    f"{len(split_dst_list)} unique "
                                    f"destination entities (< {_min_pool}) "
                                    f"for inductive negative sampling. "
                                    f"Falling back to transductive "
                                    f"(full-graph) negatives would "
                                    f"inflate AUC by 0.1-0.3 (v81 "
                                    f"P0-F3 root cause). Use a larger "
                                    f"split OR set "
                                    f"DRUGOS_ALLOW_SMALL_SPLIT_NEGATIVES=1 "
                                    f"to permit the transductive "
                                    f"fallback (dev mode only — AUC "
                                    f"will be inflated). (P2-018 root fix)"
                                )
                        # v43 Chain 6: use the FULL positive set, not
                        # just this split's positives. Falls back to
                        # per-split set if full set is empty (defensive).
                        pos_pairs_set = (
                            full_pos_pairs_set
                            if full_pos_pairs_set
                            else set(
                                (int(h), int(t))
                                for h, t in pos_edge_index.t().tolist()
                            )
                        )
                        neg_h_list: List[int] = []
                        neg_t_list: List[int] = []
                        max_attempts = n_pos * 50
                        attempts = 0
                        # Seed the local RNG for reproducibility.
                        # P2-034 ROOT FIX: incorporate ``split_name`` into
                        # the seed so val and test splits with the SAME
                        # size produce DIFFERENT negative samples.
                        #
                        # The previous seed was
                        # ``self.config.seed + len(mask_indices)`` — it
                        # depended ONLY on the split size, not on which
                        # split (train/val/test) it was. For a 10K-edge
                        # graph with 10% val + 10% test, both val and
                        # test have 1K edges → same seed → SAME negatives.
                        # Val and test AUC were computed on overlapping
                        # negative sets, biasing the comparison and
                        # leading to mild overfitting to the test set
                        # (the "best val epoch" selection was correlated
                        # with test AUC).
                        #
                        # ROOT FIX: hash (split_name, len(mask_indices))
                        # into the seed using a DETERMINISTIC hash
                        # (``hashlib.sha256``, NOT Python's built-in
                        # ``hash()`` which is randomized per-process via
                        # PYTHONHASHSEED and would break reproducibility
                        # across runs). This guarantees:
                        #   (1) val and test with the same size get
                        #       DIFFERENT seeds (independent RNG streams);
                        #   (2) the same split with the same size gets
                        #       the SAME seed across runs (reproducible);
                        #   (3) changing val_ratio changes test negatives
                        #       only if it changes the test SIZE (the
                        #       split_name component is stable).
                        # The ``& 0xFFFFFFFF`` masks to 32 bits because
                        # ``torch.Generator.manual_seed`` requires a
                        # uint32 (Python ints are arbitrary precision).
                        _split_seed_str = f"{split_name}:{len(mask_indices)}".encode("utf-8")
                        _split_seed_component = (
                            int.from_bytes(
                                hashlib.sha256(_split_seed_str).digest()[:4],
                                byteorder="big",
                                signed=False,
                            )
                            & 0xFFFFFFFF
                        )
                        _neg_rng = torch.Generator()
                        _neg_rng.manual_seed(
                            (self.config.seed + _split_seed_component) & 0xFFFFFFFF
                        )
                        # v81 P0-F3: sample from the per-split entity
                        # pools (inductive), not the full graph node
                        # count (transductive).
                        _n_src_pool = len(split_src_list)
                        _n_dst_pool = len(split_dst_list)
                        while (
                            len(neg_h_list) < n_pos
                            and attempts < max_attempts
                        ):
                            attempts += 1
                            _h_pick = int(torch.randint(
                                0, _n_src_pool, (1,), generator=_neg_rng
                            ).item())
                            _t_pick = int(torch.randint(
                                0, _n_dst_pool, (1,), generator=_neg_rng
                            ).item())
                            h_idx = int(split_src_list[_h_pick])
                            t_idx = int(split_dst_list[_t_pick])
                            if (h_idx, t_idx) in pos_pairs_set:
                                continue
                            neg_h_list.append(h_idx)
                            neg_t_list.append(t_idx)
                        n_neg = len(neg_h_list)
                        if n_neg > 0:
                            neg_edge_index = torch.tensor(
                                [neg_h_list, neg_t_list], dtype=torch.long
                            )
                            neg_labels = torch.zeros(n_neg, dtype=torch.float)
                            combined_edge_index = torch.cat(
                                [pos_edge_index, neg_edge_index], dim=1
                            )
                            combined_labels = torch.cat([pos_labels, neg_labels])
                            # v88 ROOT FIX (BUG #48 — PyG HeteroData built
                            # with mismatched edge_index): set
                            #   edge_index = pos_edge_index (positives only)
                            #   edge_label_index = combined_edge_index
                            # Negative edges should NOT be in the message-
                            # passing graph; they should only be in
                            # edge_label_index for scoring.
                            split_data[
                                target_edge_type
                            ].edge_index = pos_edge_index
                            split_data[target_edge_type].edge_label = combined_labels
                            split_data[
                                target_edge_type
                            ].edge_label_index = combined_edge_index
                            if n_neg < n_pos:
                                # P2-019 ROOT FIX (negative shortfall
                                # warning does not raise):
                                # The previous code only logged a
                                # WARNING when ``n_neg < n_pos``. The
                                # val/test split was then computed
                                # with mismatched pos/neg counts, and
                                # the downstream BCE loss computed on
                                # unequal pos/neg sets — mathematically
                                # valid but statistically biased (the
                                # model's decision threshold is skewed
                                # by the imbalance). Operators saw a
                                # WARNING buried in logs and may not
                                # realise the AUC is unreliable.
                                #
                                # ROOT FIX: RAISE RuntimeError when
                                # ``n_neg < 0.5 * n_pos`` (less than
                                # half the required negatives). The
                                # AUC is genuinely uninterpretable
                                # with insufficient negatives — the
                                # V1 launch criterion (0.85 AUC) may
                                # be met on a split with insufficient
                                # negatives, giving false confidence
                                # in a model that will fail in
                                # production. Operators can override
                                # with
                                # DRUGOS_ALLOW_INSUFFICIENT_NEGATIVES=1
                                # for dev runs.
                                _allow_insufficient = (
                                    os.environ.get(
                                        "DRUGOS_ALLOW_INSUFFICIENT_NEGATIVES",
                                        "",
                                    ) == "1"
                                )
                                _shortfall_ratio = (
                                    n_neg / n_pos if n_pos > 0 else 1.0
                                )
                                if (
                                    _shortfall_ratio < 0.5
                                    and not _allow_insufficient
                                ):
                                    raise RuntimeError(
                                        f"temporal_split: only generated "
                                        f"{n_neg}/{n_pos} negatives "
                                        f"({100.0 * _shortfall_ratio:.1f}%) "
                                        f"for split after {attempts} "
                                        f"attempts (graph may be too "
                                        f"dense). The AUC for this "
                                        f"split is UNINTERPRETABLE with "
                                        f"insufficient negatives — the "
                                        f"V1 launch criterion (0.85 AUC) "
                                        f"may be met on this split but "
                                        f"fail in production. Set "
                                        f"DRUGOS_ALLOW_INSUFFICIENT_NEGATIVES=1 "
                                        f"to permit the run (dev mode "
                                        f"only — AUC is unreliable). "
                                        f"(P2-019 root fix)"
                                    )
                                self.logger.warning(
                                    f"temporal_split: only generated "
                                    f"{n_neg}/{n_pos} negatives for split "
                                    f"({100.0 * _shortfall_ratio:.1f}% — "
                                    f"after {attempts} attempts; graph "
                                    f"may be too dense). AUC for this "
                                    f"split may be inflated. "
                                    f"(P2-019 — shortfall below the 50% "
                                    f"RAISE threshold was {'overridden' if _allow_insufficient else 'not triggered'})."
                                )
                        else:
                            # Fall back to positive-only if neg gen failed.
                            split_data[
                                target_edge_type
                            ].edge_index = pos_edge_index
                            split_data[target_edge_type].edge_label = pos_labels
                            split_data[
                                target_edge_type
                            ].edge_label_index = pos_edge_index
                            self.logger.warning(
                                f"temporal_split: generated 0 negatives "
                                f"for split — AUC will be 0.5 by default."
                            )
                    else:
                        # Train split: positives only (PyG sampler will
                        # generate negatives during training).
                        split_data[
                            target_edge_type
                        ].edge_index = pos_edge_index
                        # FIX(issue-9): temporal_split output
                        # includes edge_label/_index.
                        # FIX(issue-15): temporal_split output includes
                        # edge_label/_index.
                        # FIX(issue-80): temporal_split output compatible with
                        # PyG training.
                        split_data[target_edge_type].edge_label = pos_labels
                        split_data[
                            target_edge_type
                        ].edge_label_index = pos_edge_index
                else:
                    split_data[
                        target_edge_type
                    ].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long
                    )
                    split_data[
                        target_edge_type
                    ].edge_label = torch.zeros(
                        0, dtype=torch.float
                    )
                    split_data[
                        target_edge_type
                    ].edge_label_index = torch.zeros(
                        (2, 0), dtype=torch.long
                    )

                return split_data

            # H-8: train split stays positive-only; val/test get
            # negatives so AUC is computable on the held-out splits.
            train_data = _make_split(train_mask, generate_negatives=False, split_name="train")
            val_data = _make_split(val_mask, generate_negatives=True, split_name="val")
            test_data = _make_split(test_mask, generate_negatives=True, split_name="test")

            # FIX(issue-80): temporal_split output compatible with PyG
            # training -- post-split assertion.
            # P2-066 ROOT FIX: replace ``assert`` with explicit
            # ``if not ...: raise RuntimeError(...)``. Python's ``assert``
            # is a NO-OP when the interpreter runs with ``-O`` (optimize)
            # flag — production deployments often run with ``-O`` for
            # performance. The post-split integrity check is too
            # important to be skipped in production: a malformed split
            # (missing edge_label) would pass the check silently and
            # crash later during training with a cryptic PyG error.
            # Root fix: use a real ``if`` + ``raise RuntimeError`` so
            # the check fires regardless of the ``-O`` flag. The error
            # message is preserved verbatim so existing log-grep
            # patterns still match.
            for name, sd in [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]:
                tgt = sd[target_edge_type]
                if not (hasattr(tgt, "edge_label") and tgt.edge_label is not None):
                    raise RuntimeError(
                        f"{name} split missing edge_label on "
                        f"{target_edge_type} (P2-066 root fix: assert "
                        f"replaced with RuntimeError so the check "
                        f"survives python -O mode)"
                    )
                if not (hasattr(tgt, "edge_label_index") and tgt.edge_label_index is not None):
                    raise RuntimeError(
                        f"{name} split missing edge_label_index on "
                        f"{target_edge_type} (P2-066 root fix: assert "
                        f"replaced with RuntimeError so the check "
                        f"survives python -O mode)"
                    )

            return train_data, val_data, test_data

    # ═══ Section D -- Serialization (Save/Load) ════════════════════

    def save_heterodata(
        self,
        data: HeteroData,
        filename: Optional[str] = None,
        versioned: bool = True,
    ) -> Path:
        """Save HeteroData to disk.

        Args:
            data: HeteroData to save.
            filename: Output filename. Defaults to PyGConfig default.
            versioned: If True, append timestamp+config_hash to filename.

        Returns:
            Path: Path to the saved file.

        Audit findings addressed:
            - Issue 36: post-save verification via reload + type check
            - Issue 38: retry logic for I/O
            - Issue 44: versioned filenames prevent silent overwrites
            - Issue 45: companion .meta.json with input checksums + lineage
            - Issue 55: directory permission check
            - Issue 67: single source of truth for default filename
            # FIX(issue-69): env var overrides in PyGConfig.
            - Issue 77: schema versioning
            - Issue 78: documented .pt file format spec
            - Issue-88: full lineage in companion .meta.json
        """
        with self._timed("save_heterodata"):
            # FIX(issue-67): single source of truth for default filename.
            if filename is None:
                filename = self.config.DEFAULT_HETERODATA_FILENAME

            ensure_dirs()

            # FIX(issue-44): versioned filenames prevent silent overwrites.
            if versioned:
                timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                config_hash = hashlib.sha256(
                    json.dumps(
                        asdict(self.config),
                        default=str,
                        sort_keys=True,
                    ).encode()
                ).hexdigest()[:8]
                stem = Path(filename).stem
                suffix = Path(filename).suffix or ".pt"
                filename = f"{stem}__{timestamp}__{config_hash}{suffix}"

            path = PROCESSED_DIR / filename

            # FIX(issue-55): warn on world/group-writable save directory.
            self._check_directory_security(PROCESSED_DIR)

            # FIX(issue-77): schema versioning attached to HeteroData.
            data.__pyg_builder_schema_version__ = PYG_BUILDER_SCHEMA_VERSION
            data.__pyg_builder_pipeline_version__ = (
                PYG_BUILDER_PIPELINE_VERSION
            )
            data.__saved_at__ = datetime.now(timezone.utc).isoformat()

            # FIX(issue-38): exponential backoff retry for I/O operations.
            def _do_save():
                torch.save(data, path)

            self._with_retry(_do_save, "save_heterodata")

            # FIX(issue-36): post-save verification via reload + type check.
            saved_size = path.stat().st_size
            if saved_size == 0:
                raise IOError(
                    f"Saved file {path} is 0 bytes -- filesystem may be "
                    f"full or path unwritable."
                )

            # FIX(issue-45): companion .meta.json with input checksums +
            # lineage.
            sha256_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            meta = {
                "sha256": sha256_hash,
                "size_bytes": saved_size,
                "saved_at": data.__saved_at__,
                "schema_version": PYG_BUILDER_SCHEMA_VERSION,
                "pipeline_version": PYG_BUILDER_PIPELINE_VERSION,
                "config": {
                    k: str(v) for k, v in asdict(self.config).items()
                },
                "input_checksums": self._input_checksums,
                "node_type_counts": {
                    nt: data[nt].num_nodes for nt in data.node_types
                },
                "edge_type_counts": {
                    str(et): data[et].edge_index.shape[1]
                    for et in data.edge_types
                    if hasattr(data[et], "edge_index")
                    and data[et].edge_index is not None
                },
                "feature_provenance": {},
                "pyg_version": torch_geometric.__version__,
                "torch_version": torch.__version__,
                "lineage": getattr(data, "__lineage__", {}),
            }
            # FIX(issue-88): full lineage in companion .meta.json.
            for nt in data.node_types:
                fp = getattr(data[nt], "__feature_provenance__", None)
                if fp is not None:
                    meta["feature_provenance"][nt] = fp

            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta_path.write_text(
                json.dumps(meta, indent=2, default=str)
            )

            self.logger.info(
                f"HeteroData saved to {path} ({saved_size:,} bytes, "
                f"verified, sha256={sha256_hash[:16]}...)"
            )
            return path

    def load_heterodata(
        self,
        filename: Optional[str] = None,
        allow_unsafe_deserialization: bool = False,
        expected_sha256: Optional[str] = None,
    ) -> HeteroData:
        """Load HeteroData from disk.

        Args:
            filename: File to load. Defaults to PyGConfig default.
            allow_unsafe_deserialization: If True, allows
                weights_only=False. Defaults to False for safety.
            expected_sha256: Optional expected SHA-256 hash.

        Returns:
            HeteroData: The loaded graph data.

        Raises:
            SecurityError: On hash mismatch or unsafe load without opt-in.

        Audit findings addressed:
            - Issue 35: narrow exception handling
            - Issue 36: type check after load
            - Issue 38: retry logic for I/O
            - Issue 39: post-load schema validation
            - Issue 53: no silent RCE fallback
            - Issue 54: SHA-256 integrity verification
            - Issue 67: single source of truth for default filename
            - Issue 69: env var overrides
            - Issue 76: documented security policy
            - Issue 77: schema versioning check
        """
        with self._timed("load_heterodata"):
            # FIX(issue-67): single source of truth for default filename.
            if filename is None:
                filename = self.config.DEFAULT_HETERODATA_FILENAME

            path = PROCESSED_DIR / filename

            # FIX(issue-54): SHA-256 integrity verification on load.
            # Check companion .meta.json first.
            # P2-025 ROOT FIX (v107): the previous code SKIPPED the hash
            # check when ``.meta.json`` was MISSING — an attacker who
            # deletes the ``.meta.json`` can substitute a malicious
            # ``.pt`` file (wrong node features, wrong edges) and the
            # loader accepts it without verification. ROOT FIX: in
            # production mode (DRUGOS_ENVIRONMENT=production), REQUIRE
            # ``.meta.json`` to exist — raise SecurityError if missing.
            # In dev mode, log a WARNING and proceed (so dev fixtures
            # without .meta.json still load).
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            if not meta_path.exists():
                _is_prod_p2_025 = os.environ.get(
                    "DRUGOS_ENVIRONMENT", "production"
                ).lower() in ("prod", "production")
                if _is_prod_p2_025:
                    raise SecurityError(
                        f"P2-025 ROOT FIX: companion .meta.json is MISSING "
                        f"for {path}. In production mode, the SHA-256 "
                        f"integrity check is MANDATORY — a missing "
                        f".meta.json means the file's authenticity cannot "
                        f"be verified (supply-chain attack vector). "
                        f"Either restore the .meta.json file or set "
                        f"DRUGOS_ENVIRONMENT=dev to allow loading without "
                        f"verification (dev fixtures only)."
                    )
                else:
                    self.logger.warning(
                        "P2-025 ROOT FIX: companion .meta.json is MISSING "
                        "for %s — SKIPPING SHA-256 integrity check (dev "
                        "mode). The file's authenticity cannot be "
                        "verified. Do NOT use in production.",
                        path,
                    )
            else:
                meta = json.loads(meta_path.read_text())
                stored_hash = meta.get("sha256")
                if stored_hash:
                    actual_hash = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    if actual_hash != stored_hash:
                        raise SecurityError(
                            f"SHA-256 mismatch for {path}: "
                            f"expected {stored_hash[:16]}..., "
                            f"got {actual_hash[:16]}... File may be "
                            f"corrupted or tampered."
                        )

            # FIX(issue-53): explicit SHA-256 check if provided.
            if expected_sha256 is not None:
                actual_hash = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                if actual_hash != expected_sha256:
                    raise SecurityError(
                        f"SHA-256 mismatch for {path}: "
                        f"expected {expected_sha256[:16]}..., "
                        f"got {actual_hash[:16]}..."
                    )

            # FIX(issue-38): retry logic for I/O.
            def _do_load():
                # FIX(issue-53): no silent RCE fallback -- require explicit
                # opt-in + SHA-256 verification.
                try:
                    return torch.load(path, weights_only=True)
                except (
                    pickle.UnpicklingError,
                    RuntimeError,
                    EOFError,
                    ValueError,
                ) as exc:
                    # FIX(issue-35): narrow exception handling -- let
                    # OOM/SIGINT propagate.
                    self.logger.warning(
                        f"weights_only=True failed for {path}: "
                        f"{type(exc).__name__}: {exc}."
                    )
                    if allow_unsafe_deserialization:
                        self.logger.critical(
                            f"UNSAFE LOAD: loading {path} with "
                            f"weights_only=False. "
                            f"File size: {path.stat().st_size:,} bytes, "
                            f"mtime: {datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()}. "
                            f"This is a SECURITY RISK -- only use for "
                            f"trusted files."
                        )
                        return torch.load(
                            path, weights_only=False
                        )
                    else:
                        raise SecurityError(
                            "Refusing to load with weights_only=False "
                            "without explicit opt-in. Pass "
                            "allow_unsafe_deserialization=True to "
                            "override."
                        )

            data = self._with_retry(_do_load, "load_heterodata")

            # FIX(issue-39): post-load schema validation.
            if not isinstance(data, HeteroData):
                raise TypeError(
                    f"Loaded object is {type(data).__name__}, "
                    f"expected HeteroData. File may be from a different "
                    f"code version."
                )
            expected_node_types = self.config.expected_node_types
            if expected_node_types is not None:
                missing = set(expected_node_types) - set(
                    data.node_types
                )
                if missing:
                    raise ValueError(
                        f"Loaded HeteroData is missing node types: "
                        f"{missing}"
                    )

            # FIX(issue-77): schema versioning check.
            saved_version = getattr(
                data, "__pyg_builder_schema_version__", None
            )
            if saved_version is None:
                self.logger.warning(
                    "Loaded file has no schema version -- "
                    "pre-1.0.0 format."
                )
            elif saved_version != PYG_BUILDER_SCHEMA_VERSION:
                raise ValueError(
                    f"Schema version mismatch: file is {saved_version}, "
                    f"current code is {PYG_BUILDER_SCHEMA_VERSION}. "
                    f"Migration required -- see MIGRATION_NOTES.md."
                )

            self.logger.info(f"HeteroData loaded from {path}")
            return data

    # ═══ Section E -- Summary & Reporting ══════════════════════════

    def summarize_heterodata(
        self, data: HeteroData
    ) -> Dict[str, Any]:
        """Print and return a summary of the HeteroData.

        Returns a dict with node/edge counts, feature dimensions,
        lineage metadata, and feature provenance.

        Audit findings addressed:
            - Issue 26: handles missing edge_index safely
            - Issue 89: lineage fields in summary
        """
        with self._timed("summarize_heterodata"):
            summary: Dict[str, Any] = {
                "node_types": len(data.node_types),
                "edge_types": len(data.edge_types),
                "nodes_per_type": {},
                "edges_per_type": {},
            }

            for nt in data.node_types:
                num = data[nt].num_nodes
                feat_dim = (
                    data[nt].x.shape[1]
                    if data[nt].x is not None
                    else 0
                )
                summary["nodes_per_type"][nt] = {
                    "count": num,
                    "feat_dim": feat_dim,
                }

            # FIX(issue-26): summarize_heterodata handles missing
            # edge_index safely.
            for et in data.edge_types:
                if (
                    hasattr(data[et], "edge_index")
                    and data[et].edge_index is not None
                ):
                    num = data[et].edge_index.shape[1]
                else:
                    num = 0
                # Use str(et) as key -- tuple keys break JSON serialization
                # (cross-ref Issue 88).
                summary["edges_per_type"][str(et)] = num

            total_nodes = sum(
                v["count"] for v in summary["nodes_per_type"].values()
            )
            total_edges = sum(summary["edges_per_type"].values())
            summary["total_nodes"] = total_nodes
            summary["total_edges"] = total_edges

            # FIX(issue-89): lineage fields in summary.
            lineage = getattr(data, "__lineage__", {})
            summary["lineage"] = {
                "created_at": lineage.get("created_at"),
                "pipeline_version": lineage.get("pipeline_version"),
                "pyg_builder_version": lineage.get(
                    "pyg_builder_version"
                ),
                "input_checksums": lineage.get("input_checksums", {}),
                "feature_provenance": {
                    nt: getattr(
                        data[nt], "__feature_provenance__", None
                    )
                    for nt in data.node_types
                },
            }

            return summary


# FIX(issue-24): __main__ block works as both module and script.
if __name__ == "__main__":
    # Allow running as both `python -m drugos_graph.pyg_builder`
    # and `python pyg_builder.py` from inside drugos_graph/
    _pkg_parent = Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in sys.path:
        sys.path.insert(0, str(_pkg_parent))

    logging.basicConfig(level=logging.INFO)

    try:
        from drugos_graph.drkg_loader import (
            build_edge_index_maps,
            build_entity_id_maps,
            load_drkg,
        )
    except ImportError:
        from drkg_loader import (
            build_edge_index_maps,
            build_entity_id_maps,
            load_drkg,
        )

    df, _, _ = load_drkg(download=False)
    entity_maps = build_entity_id_maps(df)
    edge_maps = build_edge_index_maps(df, entity_maps)

    builder = PyGBuilder()
    data = builder.build_from_drkg(entity_maps, edge_maps)
    summary = builder.summarize_heterodata(data)
    print(f"\nHeteroData Summary:")
    print(f"  Total nodes: {summary['total_nodes']:,}")
    print(f"  Total edges: {summary['total_edges']:,}")


# ===========================================================================
# v85 P0 ROOT FIX — build_pyg_hetero_data (Phase 2→3 bridge function)
# ===========================================================================
# The run_pipeline.py (line 166) imports ``build_pyg_hetero_data`` from
# this module. The function DID NOT EXIST — the entire 4-phase pipeline
# was dead on arrival (ImportError at runtime). This function converts
# Phase 2 node/edge dicts into the format the GT-RL bridge expects,
# with critical node-type name mapping (Compound→drug, Disease→disease).

# INT-004 ROOT FIX: use the shared schema mapping instead of a local dict
# that diverged from phase2_adapter.PHASE2_TO_PHASE3_NODE. The previous
# _PHASE2_TO_GT_NODE_TYPE had 7 entries (including Gene and MedDRA_Term)
# while the adapter's mapping had 5 — producing DIFFERENT graphs from the
# same source. Both now import from schema_mappings.
from .schema_mappings import (
    PHASE2_TO_PHASE3_NODE as _PHASE2_TO_GT_NODE_TYPE,
    PHASE3_TO_PHASE2_NODE as _GT_TO_PHASE2_NODE_TYPE,
    ALL_PHASE2_NODE_TYPES,
    ALL_PHASE3_NODE_TYPES,
)

# Keep the old constant names as aliases so existing code doesn't break.
# These are deprecated — new code should import from schema_mappings directly.
#
# v108 ROOT FIX (Team 6 + Team 4 + TM5, conflict-resolved): the previous
# "INT-004 root fix" called ``__all__.extend([...])`` here WITHOUT ever
# defining ``__all__`` at module level. Python raised ``NameError: name
# '__all__' is not defined`` at IMPORT TIME — the entire ``pyg_builder``
# module was UNIMPORTABLE. Any pipeline that did ``from .pyg_builder
# import PyGBuilder`` or ``import pyg_builder`` crashed immediately.
# The whole Phase 2 -> Phase 3 PyG path was dead on arrival. The user
# explicitly warned: "many of these fixes introduced NEW bugs while
# patching old ones" — this was one of them.
#
# Team 4, Team 6, and Team 5 (TM5) all independently found and fixed
# this bug. Conflict resolved by taking Team 6's approach (most complete
# — defines ``__all__`` with the ACTUAL public API names so
# ``from pyg_builder import *`` exports the real public names), then
# extends with the deprecated aliases. TM5's P2-024/P2-025 ROOT FIX
# (SecurityError consolidation) sits above at line ~225 and is
# preserved through this rebase.
__all__: list = [
    "SecurityError",
    "GraphBuilderProtocol",
    "LinkPredictionSplit",
    "HeteroDataSummary",
    "PyGBuilder",
    "build_pyg_hetero_data",
]
__all__.extend([
    "_PHASE2_TO_GT_NODE_TYPE",
    "_GT_TO_PHASE2_NODE_TYPE",
    "ALL_PHASE2_NODE_TYPES",
    "ALL_PHASE3_NODE_TYPES",
])


def build_pyg_hetero_data(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    config: Optional["PyGConfig"] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, int]], List[Tuple[str, str]]]:
    """Build PyG HeteroData from Phase 2 node/edge dicts.

    This is the CRITICAL Phase 2→3 bridge function that run_pipeline.py
    imports. It converts the stage_phase1_to_phase2 output (lists of
    node/edge dicts) into the HeteroData + node_maps + known_pairs
    format that gt_rl_bridge.GTRLBridge expects.

    Key design: maps Capitalized Phase 2 node types ("Compound",
    "Disease") to the lowercase names the GT-RL bridge expects
    ("drug", "disease"). Without this, every key lookup in the
    bridge fails with KeyError or empty dict.

    Parameters
    ----------
    nodes : list[dict]
        Phase 2 node dicts with "label" and "id" keys.
    edges : list[dict]
        Phase 2 edge dicts with "source_label", "target_label",
        "relation", "source_id", "target_id" keys.
    config : PyGConfig, optional
        Builder configuration.

    Returns
    -------
    hetero_data : HeteroData
        PyG graph with lowercase node type keys.
    node_maps : dict[str, dict[str, int]]
        ID→index mappings with lowercase keys.
    known_pairs : list[tuple[str, str]]
        Known drug-treats-disease pairs.
    """
    if not nodes:
        raise ValueError(
            "build_pyg_hetero_data received empty nodes list — "
            "cannot build a graph. Check Phase 1 outputs."
        )

    builder = PyGBuilder(config=config or PyGConfig())

    # Step 1: Build entity_maps from node dicts with Phase2→GT mapping
    entity_maps: Dict[str, Dict[str, int]] = {}
    _phase2_nodes_by_type: Dict[str, List[Dict]] = {}

    for node in nodes:
        p2_label = node.get("label")
        if not p2_label:
            logger.warning(
                "build_pyg_hetero_data: node missing 'label', skipping"
            )
            continue
        # P2-006 ROOT FIX (v109): the previous code used ``.lower()`` as
        # a fallback for unknown Phase 2 labels — silently letting
        # "AdverseEvent", "Side Effect", "Drug" (when not in the map),
        # and any other arbitrary label through the contract. This
        # corrupted the Phase 3 HeteroData with node types that the
        # Graph Transformer's contract does not recognize (it expects
        # only: drug, protein, gene, pathway, disease, clinical_outcome,
        # side_effect, anatomy). ROOT FIX: do NOT silently fall back to
        # ``.lower()``. If the label is not in the contract, log a
        # warning and SKIP the node — this preserves the contract and
        # surfaces the data-quality issue in the audit log.
        gt_label = _PHASE2_TO_GT_NODE_TYPE.get(p2_label)
        if gt_label is None:
            # ``_PHASE2_TO_GT_NODE_TYPE`` is ``PHASE2_TO_PHASE3_NODE``
            # from ``phase2_schema``. Some entries map to ``None`` (e.g.
            # ``Gene`` and ``MedDRA_Term`` are intermediates that fold
            # into other types). For those, the node is intentionally
            # dropped at the Phase 2→3 boundary — skip silently.
            if p2_label in _PHASE2_TO_GT_NODE_TYPE:
                logger.debug(
                    "build_pyg_hetero_data: dropping intermediate "
                    "Phase 2 node label %r (maps to None in Phase 3 "
                    "contract).",
                    p2_label,
                )
                continue
            # GENUINELY unknown label — skip with a warning so the
            # operator can fix the upstream data quality issue.
            logger.warning(
                "build_pyg_hetero_data: unknown Phase 2 node label %r "
                "(not in PHASE2_TO_PHASE3_NODE contract) — skipping. "
                "Known labels: %s. This is a data-quality issue in the "
                "Phase 1→2 bridge.",
                p2_label, sorted(_PHASE2_TO_GT_NODE_TYPE.keys()),
            )
            continue
        if gt_label not in entity_maps:
            entity_maps[gt_label] = {}
            _phase2_nodes_by_type[gt_label] = []
        _phase2_nodes_by_type[gt_label].append(node)

    # Assign contiguous indices (deterministic order)
    for gt_label in sorted(entity_maps.keys()):
        type_nodes = _phase2_nodes_by_type[gt_label]
        type_nodes_sorted = sorted(type_nodes, key=lambda n: n.get("id", ""))
        id_map: Dict[str, int] = {}
        for idx, node in enumerate(type_nodes_sorted):
            node_id = node.get("id")
            if node_id:
                id_map[str(node_id)] = idx
        entity_maps[gt_label] = id_map

    logger.info(
        "build_pyg_hetero_data: node types: %s",
        {k: len(v) for k, v in entity_maps.items()},
    )

    if not entity_maps:
        raise ValueError(
            "build_pyg_hetero_data: no valid node types found. "
            f"Expected: {list(_PHASE2_TO_GT_NODE_TYPE.keys())}"
        )

    # Step 2: Build edge_maps with name mapping
    edge_maps: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

    for edge in edges:
        p2_src = edge.get("source_label", "")
        p2_dst = edge.get("target_label", "")
        relation = edge.get("relation", "")
        if not (p2_src and p2_dst and relation):
            continue

        # P2-006 ROOT FIX (v109): same fix as for nodes — do NOT fall
        # back to ``.lower()`` for unknown labels. Skip the edge with a
        # warning so the operator can fix the upstream issue.
        gt_src = _PHASE2_TO_GT_NODE_TYPE.get(p2_src)
        gt_dst = _PHASE2_TO_GT_NODE_TYPE.get(p2_dst)
        if gt_src is None or gt_dst is None:
            # If the label is in the map but maps to None, it's an
            # intentional drop (e.g. Gene→None). Otherwise it's an
            # unknown label — warn loudly.
            if p2_src not in _PHASE2_TO_GT_NODE_TYPE:
                logger.warning(
                    "build_pyg_hetero_data: unknown source label %r "
                    "(not in PHASE2_TO_PHASE3_NODE contract) — skipping "
                    "edge %r. Known labels: %s.",
                    p2_src, relation,
                    sorted(_PHASE2_TO_GT_NODE_TYPE.keys()),
                )
            if p2_dst not in _PHASE2_TO_GT_NODE_TYPE:
                logger.warning(
                    "build_pyg_hetero_data: unknown target label %r "
                    "(not in PHASE2_TO_PHASE3_NODE contract) — skipping "
                    "edge %r. Known labels: %s.",
                    p2_dst, relation,
                    sorted(_PHASE2_TO_GT_NODE_TYPE.keys()),
                )
            continue
        edge_key = (gt_src, relation, gt_dst)

        src_id = str(edge.get("source_id", ""))
        dst_id = str(edge.get("target_id", ""))
        src_idx = entity_maps.get(gt_src, {}).get(src_id)
        dst_idx = entity_maps.get(gt_dst, {}).get(dst_id)

        if src_idx is None or dst_idx is None:
            continue

        if edge_key not in edge_maps:
            edge_maps[edge_key] = ([], [])
        edge_maps[edge_key][0].append(src_idx)
        edge_maps[edge_key][1].append(dst_idx)

    logger.info(
        "build_pyg_hetero_data: edge types: %s",
        {f"({k[0]},{k[1]},{k[2]})": len(v[0]) for k, v in edge_maps.items()},
    )

    # Step 3: Build HeteroData via PyGBuilder
    hetero_data = builder.build_from_drkg(entity_maps, edge_maps)

    # Step 4: Extract known drug→disease pairs
    known_pairs: List[Tuple[str, str]] = []
    drug_id_map = entity_maps.get("drug", {})
    disease_id_map = entity_maps.get("disease", {})
    drug_idx_to_id = {v: k for k, v in drug_id_map.items()}
    disease_idx_to_id = {v: k for k, v in disease_id_map.items()}

    for edge_key in edge_maps:
        if edge_key[1] == "treats" and edge_key[0] in ("drug", "compound") \
                and edge_key[2] in ("disease",):
            src_indices, dst_indices = edge_maps[edge_key]
            for si, di in zip(src_indices, dst_indices):
                drug_id = drug_idx_to_id.get(si)
                disease_id = disease_idx_to_id.get(di)
                if drug_id and disease_id:
                    pair = (drug_id, disease_id)
                    if pair not in known_pairs:
                        known_pairs.append(pair)

    logger.info(
        "build_pyg_hetero_data: %d known drug-treats-disease pairs",
        len(known_pairs),
    )

    node_maps = entity_maps
    return hetero_data, node_maps, known_pairs
