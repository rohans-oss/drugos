"""RecordingGraphBuilder serialization contract.

TASK 323 ROOT FIX (forensic, root-level):
  Previously the ``RecordingGraphBuilder.save()`` and ``.load()`` methods
  in ``phase2/drugos_graph/phase1_bridge.py`` defined the serialization
  format INLINE — the snapshot schema (keys, value types, format version)
  was hardcoded in the writer. Phase 3's reader
  (``graph_transformer.data.phase2_adapter``) had to REVERSE-ENGINEER the
  format from the writer's source code, with no formal contract to verify
  against. When the writer changed a key (e.g. added ``dead_letter`` in
  v108 issue 67), Phase 3 silently broke until someone noticed.

  This module extracts the serialization format into a CONTRACT that both
  sides import. Any change to the format is now a compile-time error on
  both sides — the contract consistency test (Task 330) verifies that
  the writer's snapshot keys match this contract and that the reader
  accepts the same set of keys.

Serialization format (v2)
-------------------------
The RecordingGraphBuilder serializes its in-memory state to a single
file (JSON or Parquet) containing a snapshot dict with the following keys:

    {
      "__version__": "2",                # format version (int as string)
      "format": "json" | "parquet",      # which serializer was used
      "node_loads": List[NodeLoadDict],  # one entry per load_nodes_batch call
      "edge_loads": List[EdgeLoadDict],  # one entry per load_edges_batch call
      "_node_ids_by_label": Dict[str, List[str]],  # for referential integrity checks
      "dead_letter": List[DeadLetterDict],         # records rejected by validation
    }

NodeLoadDict shape:
    {
      "label": str,                          # Phase 2 node label (e.g. "Compound")
      "requested": int,                      # number of nodes submitted
      "accepted": int,                       # number that passed validation
      "nodes": List[Dict[str, Any]],         # the accepted node records
      "source": str,                         # data source (e.g. "chembl")
      "dead_lettered": int,                  # number rejected in this batch
      "dropped_property_keys": Dict[str, int],  # property keys stripped by whitelist
    }

EdgeLoadDict shape:
    {
      "src_label": str,
      "rel_type": str,
      "dst_label": str,
      "requested": int,
      "accepted": int,
      "edges": List[Dict[str, Any]],
      "source": str,
      "dead_lettered": int,
      "dropped_property_keys": Dict[str, int],
    }

DeadLetterDict shape:
    {
      "source": str,
      "reason": str,                         # machine-readable rejection reason
      "record": Dict[str, Any],              # the rejected record
    }

Parquet format
--------------
Parquet doesn't natively support nested Python dicts, so the snapshot
is JSON-serialized to bytes first, then stored as a single-row,
single-column Parquet file with column name ``snapshot_json``. This
preserves the exact JSON schema while gaining Parquet's compression
and type metadata. Phase 3 callers can read either format transparently.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# =============================================================================
# Format version
# =============================================================================
# Bump this when the snapshot schema changes (added/removed/renamed keys).
# Both writer and reader MUST agree on this version — the reader rejects
# snapshots from incompatible versions with a clear error message.
RECORDING_GRAPH_BUILDER_FORMAT_VERSION: str = "2"

# =============================================================================
# Supported serialization formats
# =============================================================================
# The writer MUST support both. The reader MUST auto-detect from the file
# extension (.parquet / .pq -> Parquet, anything else -> JSON).
RECORDING_GRAPH_BUILDER_SUPPORTED_FORMATS: Tuple[str, ...] = ("json", "parquet")

# =============================================================================
# Snapshot keys — the EXACT set of top-level keys in the snapshot dict
# =============================================================================
# Both writer and reader MUST agree on this set. The writer must include
# ALL of these keys (no missing keys). The reader must accept ALL of these
# keys (no extra-key rejection, but unknown keys trigger a WARNING).
RECORDING_GRAPH_BUILDER_SNAPSHOT_KEYS: Tuple[str, ...] = (
    "__version__",            # str — format version
    "format",                 # str — "json" or "parquet"
    "node_loads",             # List[Dict] — one entry per load_nodes_batch call
    "edge_loads",             # List[Dict] — one entry per load_edges_batch call
    "_node_ids_by_label",     # Dict[str, List[str]] — for referential integrity
    "dead_letter",            # List[Dict] — records rejected by validation
)

# Required keys (must be present; missing -> hard error).
REQUIRED_SNAPSHOT_KEYS: Tuple[str, ...] = RECORDING_GRAPH_BUILDER_SNAPSHOT_KEYS

# Per-key value type spec (for runtime validation).
# "any" means the value can be any JSON-serializable type (validated deeper).
SNAPSHOT_KEY_TYPES: Dict[str, str] = {
    "__version__": "str",
    "format": "str",
    "node_loads": "list",
    "edge_loads": "list",
    "_node_ids_by_label": "dict",
    "dead_letter": "list",
}

# =============================================================================
# NodeLoadDict required keys
# =============================================================================
NODE_LOAD_REQUIRED_KEYS: Tuple[str, ...] = (
    "label", "requested", "accepted", "nodes", "source",
    "dead_lettered", "dropped_property_keys",
)

# =============================================================================
# EdgeLoadDict required keys
# =============================================================================
EDGE_LOAD_REQUIRED_KEYS: Tuple[str, ...] = (
    "src_label", "rel_type", "dst_label",
    "requested", "accepted", "edges", "source",
    "dead_lettered", "dropped_property_keys",
)

# =============================================================================
# DeadLetterDict required keys
# =============================================================================
DEAD_LETTER_REQUIRED_KEYS: Tuple[str, ...] = (
    "source", "reason", "record",
)

# =============================================================================
# Parquet single-column name
# =============================================================================
# When format == "parquet", the snapshot is JSON-serialized to bytes and
# stored as a single-row, single-column Parquet file with this column name.
PARQUET_SNAPSHOT_COLUMN_NAME: str = "snapshot_json"


# =============================================================================
# Snapshot validator — used by BOTH writer (pre-save self-check) and
# reader (post-load verification)
# =============================================================================


def validate_recording_graph_builder_snapshot(
    snapshot: Dict[str, Any],
    *,
    strict: bool = True,
) -> List[str]:
    """Validate a RecordingGraphBuilder snapshot dict against the contract.

    Returns a list of error messages. Empty list = valid snapshot.

    Parameters
    ----------
    snapshot : dict
        The snapshot dict to validate.
    strict : bool, default True
        If True, unknown keys trigger errors. If False, unknown keys
        trigger warnings (returned in the error list but with a "WARNING:"
        prefix; callers can choose to ignore).

    Returns
    -------
    list of str
        Empty if valid; otherwise a list of human-readable error messages.
    """
    errors: List[str] = []

    if not isinstance(snapshot, dict):
        return [f"Snapshot must be a dict, got {type(snapshot).__name__}."]

    # Check 1: all required keys present.
    for key in REQUIRED_SNAPSHOT_KEYS:
        if key not in snapshot:
            errors.append(
                f"Missing required snapshot key {key!r}. "
                f"Required keys: {list(REQUIRED_SNAPSHOT_KEYS)}."
            )

    # Check 2: no unknown keys (in strict mode).
    if strict:
        for key in snapshot.keys():
            if key not in REQUIRED_SNAPSHOT_KEYS:
                errors.append(
                    f"WARNING: Unknown snapshot key {key!r}. "
                    f"Known keys: {list(REQUIRED_SNAPSHOT_KEYS)}."
                )

    # Check 3: format version.
    version = snapshot.get("__version__")
    if version is not None and version != RECORDING_GRAPH_BUILDER_FORMAT_VERSION:
        errors.append(
            f"Snapshot version {version!r} does not match contract version "
            f"{RECORDING_GRAPH_BUILDER_FORMAT_VERSION!r}. "
            f"Writer and reader are out of sync — re-save the snapshot with "
            f"the current writer, or update the contract version in "
            f"phase2/contracts/kg_builder_contract.py."
        )

    # Check 4: format value.
    fmt = snapshot.get("format")
    if fmt is not None and fmt not in RECORDING_GRAPH_BUILDER_SUPPORTED_FORMATS:
        errors.append(
            f"Snapshot format {fmt!r} is not supported. "
            f"Supported: {list(RECORDING_GRAPH_BUILDER_SUPPORTED_FORMATS)}."
        )

    # Check 5: node_loads structure.
    node_loads = snapshot.get("node_loads")
    if node_loads is not None:
        if not isinstance(node_loads, list):
            errors.append(f"node_loads must be a list, got {type(node_loads).__name__}.")
        else:
            for i, load in enumerate(node_loads):
                if not isinstance(load, dict):
                    errors.append(f"node_loads[{i}] must be a dict, got {type(load).__name__}.")
                    continue
                for key in NODE_LOAD_REQUIRED_KEYS:
                    if key not in load:
                        errors.append(f"node_loads[{i}] missing key {key!r}.")

    # Check 6: edge_loads structure.
    edge_loads = snapshot.get("edge_loads")
    if edge_loads is not None:
        if not isinstance(edge_loads, list):
            errors.append(f"edge_loads must be a list, got {type(edge_loads).__name__}.")
        else:
            for i, load in enumerate(edge_loads):
                if not isinstance(load, dict):
                    errors.append(f"edge_loads[{i}] must be a dict, got {type(load).__name__}.")
                    continue
                for key in EDGE_LOAD_REQUIRED_KEYS:
                    if key not in load:
                        errors.append(f"edge_loads[{i}] missing key {key!r}.")

    # Check 7: _node_ids_by_label structure.
    node_ids = snapshot.get("_node_ids_by_label")
    if node_ids is not None:
        if not isinstance(node_ids, dict):
            errors.append(
                f"_node_ids_by_label must be a dict, got {type(node_ids).__name__}."
            )

    # Check 8: dead_letter structure.
    dead_letter = snapshot.get("dead_letter")
    if dead_letter is not None:
        if not isinstance(dead_letter, list):
            errors.append(f"dead_letter must be a list, got {type(dead_letter).__name__}.")
        else:
            for i, entry in enumerate(dead_letter):
                if not isinstance(entry, dict):
                    errors.append(f"dead_letter[{i}] must be a dict, got {type(entry).__name__}.")
                    continue
                for key in DEAD_LETTER_REQUIRED_KEYS:
                    if key not in entry:
                        errors.append(f"dead_letter[{i}] missing key {key!r}.")

    return errors


def is_valid_snapshot(snapshot: Dict[str, Any]) -> bool:
    """Return True if ``snapshot`` satisfies the contract (no errors)."""
    return not validate_recording_graph_builder_snapshot(snapshot, strict=True)
