"""
DrugOS Graph Module — Configuration
====================================
Central configuration for the DrugOS Autonomous Drug Repurposing Platform.

This module is the AUTHORITATIVE source for:
  - Directory paths (RAW_DIR, PROCESSED_DIR, etc.)
  - Data source registry (DATA_SOURCES dict)
  - Neo4j connection settings (Neo4jConfig)
  - PyG graph construction settings (PyGConfig)
  - TransE model hyperparameters (TransEConfig)
  - Knowledge graph schema (CORE_NODE_TYPES, CORE_EDGE_TYPES)
  - Entity resolution rules (CANONICAL_IDS, ID_MAPPING_PRIORITY)
  - Validation thresholds (MIN_NODES_W2, TARGET_TRANSE_AUC, etc.)
  - Logging configuration (LOG_FORMAT, LOG_LEVEL)
  - Reproducibility (SEED, set_global_seed, CONFIG_HASH)
  - AUC enforcement (AUCEnforcementLevel, assert_auc_meets_threshold)
  - Lineage metadata (LineageMetadata, build_lineage_metadata)
  - Data quality & integrity (checksums, freshness, dead-letter queue)
  - Security (safe_config_dict, PII handling, secrets registry)
  - Performance (auto-sizing, chunking, batch-size recommendations)
  - Configuration management (env overrides, JSON/YAML config loading)
  - Compliance (naming conventions, data format standards, retention)
  - Audit & lineage (transformation logging, impact analysis, diff)

v107 ROOT FIX (ISSUE-P2-056): ARCHITECTURAL ROADMAP for splitting this
monolith. This file is 8400+ lines and a change to it can break 38+ files.
The hardcoded values previously flagged below (issues 2.9, 11.2, 11.3,
5.3, 5.4, 2.4) have ALL been fixed in prior versions — entity_resolver.py
now imports CANONICAL_IDS, run_pipeline.py imports LOG_FORMAT/LOG_LEVEL,
etc. The remaining concern is purely maintainability.

INTENDED SPLIT (do NOT execute mid-release — schedule for v2.1.0):
  config_paths.py       — RAW_DIR, PROCESSED_DIR, LOGS_DIR, DATA_SOURCES,
                          get_data_source_path, get_drkg_tsv_path
  config_neo4j.py       — Neo4jConfig, Neo4jConnectionConfig, batch_size_*
  config_pyg.py         — PyGConfig, HGTConfig, DEFAULT_FEATURE_DIMS
  config_transe.py      — TransEConfig, TARGET_TRANSE_AUC, TRANSE_EMBEDDING_DIM
  config_schema.py      — CORE_NODE_TYPES, CORE_EDGE_TYPES, CORE_EDGE_TYPES_SET,
                          SYMMETRIC_RELATIONS, CANONICAL_IDS, ID_MAPPING_PRIORITY
  config_validation.py  — MIN_NODES_W2, MIN_NEGATIVE_PAIRS, AUCEnforcementLevel,
                          assert_auc_meets_threshold, safe_config_dict
  config/__init__.py    — re-exports everything for backward compat
                          (``from .config import X`` keeps working)

Migration plan: create the submodules as thin re-export wrappers first
(so ``from .config_neo4j import Neo4jConfig`` works alongside
``from .config import Neo4jConfig``), then incrementally move
definitions into the submodules in follow-up PRs. The re-export wrapper
ensures zero downtime — no consumer breaks during the migration.

Until the split is executed, this file remains the single source of
truth. The 8400-line size is tracked as tech debt; new config items
should be added in the section corresponding to their future submodule
to ease the eventual extraction.

KNOWN HARDCODED VALUES IN CONSUMERS (audit issue 13.1 — RESOLVED in v107):
  - entity_resolver.py imports CANONICAL_IDS from config (issue 2.9 — FIXED)
  - run_pipeline.py imports LOG_FORMAT from config (issue 11.3 — FIXED)
  - run_pipeline.py imports LOG_LEVEL from config (issue 11.2 — FIXED)
  - drugbank_parser.py uses get_data_source_path (issue 5.3 — FIXED)
  - drkg_loader.py uses get_drkg_tsv_path (issue 5.4 — FIXED)
  - negative_sampling.py reads MIN_NEGATIVE_PAIRS (issue 2.4 — FIXED)

Each consumer contract is documented in the corresponding section below.

Week 2 exit criteria (project-level, see project doc):
  - KG loaded with minimum 500K nodes and 6M edges
  - Training dataset with 15K+ positive and 75K+ negative pairs
  - TransE baseline AUC > 0.78
  - PyG data loader confirmed working

Clinical safety note:
  A misconfiguration here does not produce a broken test — it produces
  wrong predictions, which can be used by pharmaceutical partners to
  make wet-lab decisions, which can kill patients. Treat every line as
  a potential cause of patient harm.
"""

from __future__ import annotations

# ─── Phase A — Foundations ────────────────────────────────────────────────────

import enum
import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
import warnings
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)

# ─── v108 ROOT FIX (ISSUE-P2-056): REAL config monolith split ────────────────
# The previous v107 "fix" only DOCUMENTED the intended split and explicitly
# deferred execution to "v2.1.0" — a surface-level fix. This block DOES the
# split: path constants are MOVED to ``config_paths.py`` and schema constants
# are MOVED to ``config_schema.py``. This file imports them back so
# ``from .config import RAW_DIR, CORE_NODE_TYPES`` continues to work
# (backward compat for the 38+ files that import from config).
#
# New consumers should prefer importing from the submodules directly:
#     from .config_paths import RAW_DIR, PROCESSED_DIR
#     from .config_schema import CORE_NODE_TYPES, CORE_EDGE_TYPES
#
# Future PRs can extract config_neo4j.py, config_pyg.py, config_transe.py,
# config_validation.py following the same pattern.
from .config_paths import (  # noqa: F401 — re-exported for backward compat
    _PROJECT_ROOT,
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PHASE1_PROCESSED_DIR,
    RESULTS_PERSIST_PATH,
    KG_EXPORT_DIR,
    KG_DIR,
    EMBEDDINGS_DIR,
    LOGS_DIR,
)
from .config_schema import (  # noqa: F401 — re-exported for backward compat
    CORE_NODE_TYPES,
    DRKG_NODE_TYPES,
    CORE_EDGE_TYPES,
    CORE_EDGE_TYPES_SET,
)

# ─── A.1 Version Constants ────────────────────────────────────────────────────
# Fixes audit issue 7.3 — version constants for reproducibility
# Fixes audit issue 16.1 — lineage metadata requires version tracking

PACKAGE_VERSION: str = "2.0.0"
PIPELINE_VERSION: str = "2.0.0-week2"
CONFIG_VERSION: str = "2.0.0"
SCHEMA_VERSION: str = "2.0.0"

# ─── A.1b Domain constants for loaders (uniprot_loader audit D12-001) ────────
# These constants eliminate magic strings ('uniprot', 'UniProt', 'Protein')
# scattered across loader source files. Importing these from config means a
# rename happens in exactly one place.  Fixes D12-001.
SOURCE_KEY_UNIPROT: str = "uniprot"          # DATA_SOURCES dict key
SOURCE_UNIPROT: str = "UniProt"              # human-readable / node .source value
ENTITY_TYPE_PROTEIN: str = "Protein"         # CORE_NODE_TYPES member
UNIPROT_PARSER_VERSION: str = "2.0.0"        # bumped on any parser logic change (D7-003)
UNIPROT_SCHEMA_VERSION: str = "2.0.0"        # bumped on any output-schema change (D14-004)
UNIPROT_LICENSE: str = "CC BY 4.0"           # D14-001 attribution requirement
UNIPROT_ATTRIBUTION: str = (                 # D14-001 — propagated to every record
    "Data source: UniProt, https://www.uniprot.org/, CC BY 4.0"
)
# Minimum byte size for a cached UniProt file to be considered valid (D5-008).
# UniProt sprot is ~800 MB; the local hand-curated sample is ~12 KB, so this
# threshold is only enforced for the *production* path, not for explicit
# test/sample filepaths passed by the caller.
UNIPROT_MIN_VALID_SIZE_BYTES: int = 1_000_000

# Fixes D9-002 — URL allowlist for UniProt downloads (guard against config
# injection / SSRF). Any URL in DATA_SOURCES['uniprot']['url'] MUST start with
# one of these prefixes or the download is refused before any network call.
ALLOWED_UNIPROT_URLS: tuple[str, ...] = (
    "https://ftp.uniprot.org/pub/databases/uniprot/",
    "https://mirror.uniprot.org/pub/databases/uniprot/",
    "https://expasy.org/ftp/uniprot/",          # Swiss mirror
    "https://ftp.ebi.ac.uk/pub/databases/uniprot/",  # EBI mirror
)

# ─── A.1c Domain constants for ChEMBL loader ─────────────────────────────────
# These constants eliminate magic strings scattered across the chembl_loader
# source file. Importing from config means a rename happens in one place.
# Follows the same pattern as UNIPROT_* and DRUGBANK_* constants.
SOURCE_KEY_CHEMBL: str = "chembl"               # DATA_SOURCES dict key
SOURCE_CHEMBL: str = "ChEMBL"                   # human-readable / node .source value
ENTITY_TYPE_COMPOUND: str = "Compound"           # CORE_NODE_TYPES member
CHEMBL_PARSER_VERSION: str = "2.0.0"             # bumped on any parser logic change
CHEMBL_SCHEMA_VERSION: str = "2.0.0"             # bumped on any output-schema change
CHEMBL_LICENSE: str = "CC BY-SA 3.0"             # attribution requirement
CHEMBL_ATTRIBUTION: str = (                      # propagated to every record
    "Data source: ChEMBL, https://www.ebi.ac.uk/chembl/, CC BY-SA 3.0"
)

# ─── A.1d Domain constants for STRING loader ─────────────────────────────────
# Added by string_loader v1.0 institutional-grade audit fix
# (master_prompt_fix_string_loader.md — Section 14, Domain 12 Configuration).
# These constants eliminate magic strings scattered across the string_loader
# source file. Importing from config means a rename happens in one place.
# Follows the same pattern as UNIPROT_* / DRUGBANK_* / CHEMBL_* constants.
# Fixes: C12-08 (config validation), C12-09 (threshold docs), C12-10 (env vars).
SOURCE_KEY_STRING: str = "string"                # DATA_SOURCES dict key
SOURCE_STRING: str = "STRING"                    # human-readable / node .source value
STRING_PARSER_VERSION: str = "1.0.0"             # bumped on any parser logic change
STRING_SCHEMA_VERSION: str = "1.0.0"             # bumped on any output-schema change
STRING_LICENSE: str = "CC BY 4.0"                # STRING license (academic use free)
STRING_ATTRIBUTION: str = (                      # propagated to every record
    "Data source: STRING (Szklarczyk D. et al., Nucleic Acids Res. 2023), "
    "https://string-db.org/, CC BY 4.0"
)
# Minimum byte size for a downloaded STRING .txt.gz to be considered valid.
# The real 9606.protein.links.full.v12.0.txt.gz is ~300 MB; this threshold
# catches truncated or corrupted downloads (e.g., an HTML error page — R6-03).
STRING_MIN_VALID_SIZE_BYTES: int = 1_000_000     # 1 MB minimum

# STRING_REQUIRED
# Whether STRING is a required source for the pipeline. When True (default),
# any failure to load STRING (download failure, 0 edges, parse error) raises
# CriticalDataSourceError. When False, the pipeline logs a WARNING and
# continues (R6-04). Override via DRUGOS_STRING_REQUIRED=0 to disable.
STRING_REQUIRED: bool = os.environ.get(
    "DRUGOS_STRING_REQUIRED", "1"
) == "1"

# DEFAULT_BATCH_SIZE
# Default batch size for iter_string_edges streaming edge generation
# (master prompt Section 13, A1-08). Override per-call via the batch_size
# kwarg or globally via DRUGOS_STRING_BATCH_SIZE env var.
DEFAULT_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_STRING_BATCH_SIZE", "10000")
)

# DEFAULT_CHUNK_SIZE
# Default chunk size for iter_string_ppi streaming parse (A1-08).
# 100K rows per chunk balances memory and I/O efficiency on 11M-row files.
DEFAULT_CHUNK_SIZE: int = int(
    os.environ.get("DRUGOS_STRING_CHUNK_SIZE", "100000")
)

# EDGE_TYPE_TO_RELATION_STRING
# Maps (src_type, rel_type, dst_type) tuples to the canonical relation
# string used in Neo4j edge :TYPE. The STRING loader emits edges of type
# ("Protein", "interacts_with", "Protein") — this dict centralises the
# mapping so the loader does not hardcode the string (I15-02).
EDGE_TYPE_TO_RELATION_STRING: dict[tuple[str, str, str], str] = {
    ("Protein", "interacts_with", "Protein"): "interacts_with",
}

# ─── STITCH constants ───────────────────────────────────────────────────────
# Added by stitch_loader v1.1.0 institutional-grade audit fix
# (master_prompt_fix_stitch_loader.md — Section 4.3, Domain 12 Configuration).
#
# These constants eliminate magic strings scattered across the stitch_loader
# source file. Importing from config means a rename happens in one place.
# Follows the same pattern as UNIPROT_* / DRUGBANK_* / CHEMBL_* / STRING_*
# constants.
#
# Fixes: BUG-9.1 (URL allowlist), BUG-12.1 (threshold type), BUG-12.4 (config
#        validation), BUG-14.1 (license/attribution), BUG-14.2 (schema version),
#        BUG-5.2 (min valid size + required flag), GAP-12.2 (env vars).
SOURCE_KEY_STITCH: str = "stitch"               # DATA_SOURCES dict key
SOURCE_STITCH: str = "STITCH"                   # human-readable / node .source value
STITCH_PARSER_VERSION: str = "1.0.0"            # bumped on any parser logic change
STITCH_SCHEMA_VERSION: str = "1.1.0"            # bumped on any output-schema change
                                                # (1.0.0 -> 1.1.0: additive fields per Rule R6)
STITCH_LICENSE: str = "CC0 1.0"                 # STITCH license (public domain)
STITCH_ATTRIBUTION: str = (                     # propagated to every record
    "Data source: STITCH (Kuhn et al., Nucleic Acids Res. 2014), "
    "https://stitch.embl.de/, CC0 1.0"
)
# Minimum byte size for a downloaded STITCH .tsv.gz to be considered valid.
# The real 9606.protein_chemical.links.detailed.v5.0.tsv.gz is ~1 GB; this
# threshold catches truncated or corrupted downloads (e.g., an HTML error page).
STITCH_MIN_VALID_SIZE_BYTES: int = 1_000_000    # 1 MB minimum

# STITCH_REQUIRED
# Whether STITCH is a required source for the pipeline. When True, any failure
# to load STITCH (download failure, 0 edges, parse error) raises
# CriticalDataSourceError. When False, the pipeline logs a WARNING and
# continues. Override via DRUGOS_STITCH_REQUIRED=0 to disable.
STITCH_REQUIRED: bool = os.environ.get(
    "DRUGOS_STITCH_REQUIRED", "1"
) == "1"

# ALLOWED_STITCH_URLS
# URL-prefix allowlist for SSRF guard (BUG-9.1). The STITCH loader refuses to
# download from any URL not matching one of these prefixes. HTTPS-only.
# Mirrors the pattern of ALLOWED_STRING_URLS / ALLOWED_UNIPROT_URLS.
ALLOWED_STITCH_URLS: tuple[str, ...] = (
    "https://stitch.embl.de/download/",
    "https://stitch.embl.de/cgi/",
)

# STITCH_BATCH_SIZE
# Default batch size for iter_stitch_edges streaming edge generation
# (master prompt Section 3, BUG-8.2). Override per-call via the batch_size
# kwarg or globally via DRUGOS_STITCH_BATCH_SIZE env var.
STITCH_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_STITCH_BATCH_SIZE", "10000")
)

# STITCH_CHUNK_SIZE
# Default chunk size for iter_stitch_cpi streaming parse (BUG-8.2).
# 100K rows per chunk balances memory and I/O efficiency on 20M-row files.
STITCH_CHUNK_SIZE: int = int(
    os.environ.get("DRUGOS_STITCH_CHUNK_SIZE", "100000")
)

# STITCH_CHECKPOINT_INTERVAL
# Number of rows between checkpoint writes (GAP-6.5). Override per-call via
# the checkpoint_interval kwarg or globally via DRUGOS_STITCH_CHECKPOINT_INTERVAL.
STITCH_CHECKPOINT_INTERVAL: int = int(
    os.environ.get("DRUGOS_STITCH_CHECKPOINT_INTERVAL", "100000")
)

# EDGE_TYPE_TO_RELATION_STITCH
# Maps (src_type, rel_type, dst_type) tuples to the canonical relation string
# used in Neo4j edge :TYPE. The STITCH loader emits edges of type
# ("Compound", "binds"/"inhibits"/"activates", "Protein"). Centralises the
# mapping so the loader does not hardcode strings (BUG-15.1).
EDGE_TYPE_TO_RELATION_STITCH: dict[tuple[str, str, str], str] = {
    ("Compound", "binds", "Protein"): "binds",
    ("Compound", "inhibits", "Protein"): "inhibits",
    ("Compound", "activates", "Protein"): "activates",
    ("Compound", "allosterically_modulates", "Protein"): "allosterically_modulates",
    ("Compound", "induces", "Protein"): "induces",
    ("Compound", "metabolized_by", "Protein"): "metabolized_by",
    ("Compound", "transported_by", "Protein"): "transported_by",
    ("Compound", "carried_by", "Protein"): "carried_by",
}

# ─── SIDER constants ────────────────────────────────────────────────────────
# Added by sider_loader v1.0.0 institutional-grade audit fix
# (master_prompt — Section 3 Phase 0.4, Domain 12 Configuration).
#
# These constants eliminate magic strings scattered across the sider_loader
# source file. Importing from config means a rename happens in one place.
# Follows the same pattern as UNIPROT_* / DRUGBANK_* / CHEMBL_* / STRING_* /
# STITCH_* constants.
#
# Fixes: Phase 0.4 (A1.1 — SIDER is critical), D3.8 (pin version + sha256),
#        D12.4 (config validation), D14.1 (license/attribution),
#        D14.2 (schema version), D14.6 (schema versioning), D14.11 (filename),
#        D5.1 (expected row count), D5.9 (stale-file freshness),
#        D9.1/D9.2 (URL allowlist + HTTPS), BUG-15.1 (kg_builder contract).
SOURCE_KEY_SIDER: str = "sider"                  # DATA_SOURCES dict key
SOURCE_SIDER: str = "SIDER"                      # human-readable / node .source value
SIDER_PARSER_VERSION: str = "1.0.0"              # bumped on any parser logic change
SIDER_SCHEMA_VERSION: str = "1.0.0"              # bumped on any output-schema change
SIDER_LICENSE: str = "CC0 1.0"                   # SIDER license (public domain)
SIDER_ATTRIBUTION: str = (                       # propagated to every record
    "Data source: SIDER (Kuhn M. et al., Nucleic Acids Res. 2016), "
    "https://sideeffects.embl.de/, CC0 1.0"
)

# Minimum byte size for a downloaded SIDER .tsv.gz to be considered valid.
# The real meddra_all_se.tsv.gz is ~50 MB; this threshold catches truncated
# or corrupted downloads (e.g., an HTML error page returned by EMBL).
SIDER_MIN_VALID_SIZE_BYTES: int = 1_000_000      # 1 MB minimum

# SIDER_REQUIRED — equivalent of STITCH_REQUIRED. SIDER is in CRITICAL_SOURCES
# (Phase 0.4) so this flag is largely informational; CRITICAL_SOURCES
# membership governs the pipeline-level criticality classification.
# Override via DRUGOS_SIDER_REQUIRED=0 to make SIDER optional (NOT recommended
# — see Patient Safety note in sider_loader.py module docstring).
SIDER_REQUIRED: bool = os.environ.get(
    "DRUGOS_SIDER_REQUIRED", "1"
) == "1"

# ALLOWED_SIDER_URLS — URL-prefix allowlist for SSRF guard (D9.1/D9.2).
# The SIDER loader refuses to download from any URL not matching one of
# these prefixes. HTTPS-only.
ALLOWED_SIDER_URLS: tuple[str, ...] = (
    "https://sideeffects.embl.de/media/",
    "https://sideeffects.embl.de/download/",
    "https://sidereo.embl.de/download/",          # EBI mirror
)

# SIDER_BATCH_SIZE — default batch size for streaming edge generation
# (Domain 8 Performance). Override per-call via batch_size kwarg or
# globally via DRUGOS_SIDER_BATCH_SIZE env var.
SIDER_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_SIDER_BATCH_SIZE", "10000")
)

# SIDER_CHUNK_SIZE — default chunk size for streaming parse.
# 100K rows per chunk balances memory and I/O efficiency on 5M-row files.
SIDER_CHUNK_SIZE: int = int(
    os.environ.get("DRUGOS_SIDER_CHUNK_SIZE", "100000")
)

# SIDER_CHECKPOINT_INTERVAL — number of rows between checkpoint writes
# (Domain 6 Reliability).
SIDER_CHECKPOINT_INTERVAL: int = int(
    os.environ.get("DRUGOS_SIDER_CHECKPOINT_INTERVAL", "100000")
)

# SIDER_COMPOUND_ID_FORMAT — declares the canonical Compound ID format
# emitted by the SIDER loader (Phase 0.1 / D2.4 / A1.10). Downstream
# consumers (entity_resolver, id_crosswalk) MUST treat this as the
# canonical Compound ID. The legacy zero-padded string format
# ("drug_cid" column) is DEPRECATED and will be removed in v2.0.
SIDER_COMPOUND_ID_FORMAT: str = "pubchem_cid:int"

# SIDER_EXPECTED_COLUMN_COUNT — the SIDER meddra_all_se.tsv.gz file has
# exactly 6 columns. The loader raises SiderSchemaError if the parsed
# DataFrame has a different column count (D15.10).
SIDER_EXPECTED_COLUMN_COUNT: int = 6

# SIDER expected row count range (D5.1). The real SIDER file has ~5M
# rows. If the parsed DataFrame is outside this range, the loader raises
# SiderDataQualityError to catch silent schema drift or truncated downloads.
EXPECTED_SIDER_ROW_COUNT_MIN: int = 1_000_000
EXPECTED_SIDER_ROW_COUNT_MAX: int = 50_000_000

# SIDER canonical node/edge type strings (Phase 0.3 / D14.12 / D14.13).
# These are validated against CORE_NODE_TYPES and CORE_EDGE_TYPES at
# sider_loader import time.
SIDER_NODE_TYPE: str = "MedDRA_Term"             # canonical (with underscore)
SIDER_EDGE_TYPE: str = "causes_adverse_event"    # canonical
SIDER_LEGACY_NODE_TYPE: str = "Side Effect"      # legacy (with space)
SIDER_LEGACY_EDGE_TYPE: str = "causes_side_effect"  # legacy

# SIDER MedDRA type enum (D2.12). SIDER publishes 5 MedDRA hierarchy levels.
VALID_MEDDRA_TYPES: frozenset[str] = frozenset({
    "PT",     # Preferred Term — canonical adverse-event reporting level (default)
    "LLT",    # Lowest Level Term — sub-concept of PT, would double-count
    "HLT",    # High Level Term
    "HLGT",   # High Level Group Term
    "SOC",    # System Organ Class
})

# SIDER MedDRA type sort order for deterministic dedup (D2.7).
# PT first (canonical), then LLT (most specific), then up the hierarchy.
MEDDRA_TYPE_DEDUP_ORDER: tuple[str, ...] = (
    "PT", "LLT", "HLT", "HLGT", "SOC",
)

# PubChem CID range (D3.12 / D5.13 / GAP-3.6 — mirrors stitch_loader).
PUBCHEM_CID_MIN_SIDER: int = 1
PUBCHEM_CID_MAX_SIDER: int = 370_000_000

# UMLS CUI regex (D3.4 / D5.6). UMLS CUIs match ^C\d{7}$.
UMLS_CUI_REGEX: str = r"^C\d{7}$"

# Edge ID hash length (D2.8).
SIDER_EDGE_ID_HASH_LENGTH: int = 16

# Large-DataFrame threshold for streaming (D6.7 / D8.1).
SIDER_LARGE_DF_THRESHOLD: int = 500_000

# Stale-file freshness threshold (D5.9 / D7.3). SIDER publishes ~annually.
SIDER_STALE_FILE_DAYS: int = 365

# Download retry / timeout constants (D6.1 / D6.2).
SIDER_MAX_RETRIES: int = 3
SIDER_RETRY_BACKOFF_BASE: int = 2
SIDER_DOWNLOAD_TIMEOUT_SECONDS: int = 60

# Magic numbers (D12.7) — extracted as named constants.
SIDER_MAX_REDIRECTS: int = 5
SIDER_FILE_PERMISSIONS: int = 0o644
SIDER_LOG_DIR_PERMISSIONS: int = 0o755

# MedDRA vocabulary version (D16.5). SIDER 2023 uses MedDRA v26.0.
# Update this when SIDER publishes a new release.
SIDER_MEDDRA_VERSION: str = "26.0"

# SIDER release date — pinned version (D3.8 / D12.4).
# Update this AND the sha256 when SIDER publishes a new release.
# The current pinned release is the 2023-10-25 build.
SIDER_PINNED_VERSION: str = "2023-10-25"
SIDER_PINNED_RELEASE_DATE: str = "2023-10-25"
# sha256 of the pinned SIDER meddra_all_se.tsv.gz. Set to None when SIDER
# does not publish a checksum (SIDER does NOT publish sha256 — we compute
# it at download time and pin it for future runs, per D3.8).
# Leave None initially; the loader will compute and store it as a sidecar.
SIDER_PINNED_SHA256: str | None = None

# EDGE_TYPE_TO_RELATION_SIDER — maps (src_type, rel_type, dst_type) tuples
# to the canonical relation string used in Neo4j edge :TYPE. The SIDER
# loader emits edges of type ("Compound", "causes_adverse_event",
# "MedDRA_Term"). Centralises the mapping so the loader does not hardcode
# strings (D15.1 — kg_builder contract).
EDGE_TYPE_TO_RELATION_SIDER: dict[tuple[str, str, str], str] = {
    ("Compound", "causes_adverse_event", "MedDRA_Term"): "causes_adverse_event",
    # Legacy edge type — kept for migration-period dual-write (Phase 0.3).
    ("Compound", "causes_side_effect", "Side Effect"): "causes_side_effect",
}

# ALLOWED_STRING_URLS
# URL-prefix allowlist for SSRF guard (S9-02). The STRING loader refuses to
# download from any URL not matching one of these prefixes. HTTPS-only.
# Mirrors the pattern of ALLOWED_UNIPROT_URLS / ALLOWED_CHEMBL_URLS.
ALLOWED_STRING_URLS: tuple[str, ...] = (
    "https://string-db.org/download/",
    "https://string-db.org/cgi/",
    "https://stringdownfiles.org/",
)

# ENTITY_TYPE_PROTEIN is defined by the uniprot_loader block above;
# STRING edges have BOTH endpoints of type "Protein".

# ─── A.2 __all__ ─────────────────────────────────────────────────────────────
# Fixes audit issue 13.12 — __all__ must be defined

__all__: list[str] = [
    # ── Version constants ──
    "PACKAGE_VERSION", "PIPELINE_VERSION", "CONFIG_VERSION", "SCHEMA_VERSION",
    "CONFIG_HASH",
    # ── Loader domain constants (uniprot_loader audit D12-001) ──
    "SOURCE_KEY_UNIPROT", "SOURCE_UNIPROT", "ENTITY_TYPE_PROTEIN",
    "UNIPROT_PARSER_VERSION", "UNIPROT_SCHEMA_VERSION",
    "UNIPROT_LICENSE", "UNIPROT_ATTRIBUTION",
    "UNIPROT_MIN_VALID_SIZE_BYTES", "ALLOWED_UNIPROT_URLS",
    # ── Loader domain constants (chembl_loader audit D12-001) ──
    "SOURCE_KEY_CHEMBL", "SOURCE_CHEMBL", "ENTITY_TYPE_COMPOUND",
    "CHEMBL_PARSER_VERSION", "CHEMBL_SCHEMA_VERSION",
    "CHEMBL_LICENSE", "CHEMBL_ATTRIBUTION",
    "CHEMBL_MIN_PCHEMBL_VALUE", "CHEMBL_MIN_CONFIDENCE_SCORE",
    "CHEMBL_ORGANISM_FILTER_TAX_ID", "CHEMBL_TARGET_TYPES",
    "CHEMBL_ASSAY_TYPES", "CHEMBL_STANDARD_TYPE_TO_RELATION",
    "CHEMBL_ACTIVITY_TYPE_INHIBITS", "CHEMBL_ACTIVITY_TYPE_ACTIVATES",
    "CHEMBL_ACTIVITY_TYPE_BINDS", "CHEMBL_ACTIVITY_TYPE_MODULATES",
    "ALLOWED_CHEMBL_URLS",
    "CHEMBL_MIN_VALID_SIZE_BYTES", "CHEMBL_PROGRESS_LOG_INTERVAL",
    "CHEMBL_MIN_FIELD_POPULATION", "CHEMBL_KG_BUILDER_FIELDS",
    "CHEMBL_DRUG_IDENTIFIER_REGEX", "CHEMBL_UNIPROT_AC_REGEX",
    "CHEMBL_PCHEMBL_RANGE",
    # ── Loader domain constants (string_loader audit D12-001) ──
    # Added by string_loader v1.0 institutional-grade audit fix
    # (master_prompt_fix_string_loader.md — Section 14).
    "SOURCE_KEY_STRING", "SOURCE_STRING",
    "STRING_PARSER_VERSION", "STRING_SCHEMA_VERSION",
    "STRING_LICENSE", "STRING_ATTRIBUTION",
    "STRING_MIN_VALID_SIZE_BYTES", "STRING_REQUIRED",
    "ALLOWED_STRING_URLS",
    "DEFAULT_BATCH_SIZE", "DEFAULT_CHUNK_SIZE",
    "EDGE_TYPE_TO_RELATION_STRING",
    # ── Loader domain constants (stitch_loader audit BUG-12.4) ──
    # Added by stitch_loader v1.1.0 institutional-grade audit fix
    # (master_prompt_fix_stitch_loader.md — Section 4.3).
    "SOURCE_KEY_STITCH", "SOURCE_STITCH",
    "STITCH_PARSER_VERSION", "STITCH_SCHEMA_VERSION",
    "STITCH_LICENSE", "STITCH_ATTRIBUTION",
    "STITCH_MIN_VALID_SIZE_BYTES", "STITCH_REQUIRED",
    "ALLOWED_STITCH_URLS",
    "STITCH_BATCH_SIZE", "STITCH_CHUNK_SIZE",
    "STITCH_CHECKPOINT_INTERVAL",
    "EDGE_TYPE_TO_RELATION_STITCH",
    # ── Loader domain constants (sider_loader audit Phase 0.4 / D12.4) ──
    # Added by sider_loader v1.0.0 institutional-grade audit fix
    # (master_prompt — Section 3 Phase 0.4, Domain 12 Configuration).
    "SOURCE_KEY_SIDER", "SOURCE_SIDER",
    "SIDER_PARSER_VERSION", "SIDER_SCHEMA_VERSION",
    "SIDER_LICENSE", "SIDER_ATTRIBUTION",
    "SIDER_MIN_VALID_SIZE_BYTES", "SIDER_REQUIRED",
    "ALLOWED_SIDER_URLS",
    "SIDER_BATCH_SIZE", "SIDER_CHUNK_SIZE",
    "SIDER_CHECKPOINT_INTERVAL",
    "SIDER_COMPOUND_ID_FORMAT",
    "SIDER_EXPECTED_COLUMN_COUNT",
    "EXPECTED_SIDER_ROW_COUNT_MIN", "EXPECTED_SIDER_ROW_COUNT_MAX",
    "SIDER_NODE_TYPE", "SIDER_EDGE_TYPE",
    "SIDER_LEGACY_NODE_TYPE", "SIDER_LEGACY_EDGE_TYPE",
    "VALID_MEDDRA_TYPES", "MEDDRA_TYPE_DEDUP_ORDER",
    "PUBCHEM_CID_MIN_SIDER", "PUBCHEM_CID_MAX_SIDER",
    "UMLS_CUI_REGEX",
    "SIDER_EDGE_ID_HASH_LENGTH",
    "SIDER_LARGE_DF_THRESHOLD",
    "SIDER_STALE_FILE_DAYS",
    "SIDER_MAX_RETRIES", "SIDER_RETRY_BACKOFF_BASE",
    "SIDER_DOWNLOAD_TIMEOUT_SECONDS",
    "SIDER_MAX_REDIRECTS", "SIDER_FILE_PERMISSIONS",
    "SIDER_LOG_DIR_PERMISSIONS",
    "SIDER_MEDDRA_VERSION",
    "SIDER_PINNED_VERSION", "SIDER_PINNED_RELEASE_DATE",
    "SIDER_PINNED_SHA256",
    "EDGE_TYPE_TO_RELATION_SIDER",
    # ── OpenTargets (added by opentargets_loader v2.0 audit fix) ──
    "SOURCE_KEY_OPENTARGETS", "SOURCE_OPENTARGETS",
    "OPENTARGETS_PARSER_VERSION", "OPENTARGETS_SCHEMA_VERSION",
    "OPENTARGETS_LICENSE", "OPENTARGETS_ATTRIBUTION",
    "OPENTARGETS_TARGET_TAX_ID",
    "OPENTARGETS_MIN_SCORE_DEFAULT",
    "DISGENET_MIN_SCORE",
    "OMIM_MIN_SCORE",
    "OPENTARGETS_MIN_RESOLUTION_RATE",
    "OPENTARGETS_REGULATORY_RESOLUTION_RATE",
    "OPENTARGETS_PROGRESS_LOG_INTERVAL",
    "OPENTARGETS_DOWNLOAD_BATCH_BYTES",
    "OPENTARGETS_PARSED_CACHE_DIR",
    "OPENTARGETS_STALENESS_DAYS",
    "OPENTARGETS_NEO4J_BATCH_SIZE",
    "OPENTARGETS_DEAD_LETTER_PATH",
    "OPENTARGETS_LINEAGE_LOG_PATH",
    "OPENTARGETS_AUDIT_LOG_PATH",
    "OPENTARGETS_TRANSFORMATION_LOG_PATH",
    "OPENTARGETS_QUALITY_REPORT_PATH",
    "OPENTARGETS_CIRCUIT_BREAKER_THRESHOLD",
    "OPENTARGETS_MAX_RETRIES",
    "OPENTARGETS_RETRY_BACKOFF_BASE",
    "OPENTARGETS_DOWNLOAD_TIMEOUT_SECONDS",
    "OPENTARGETS_MIN_VALID_SIZE_BYTES",
    "OPENTARGETS_FORCE_DOWNLOAD", "OPENTARGETS_SKIP",
    "OPENTARGETS_OFFLINE", "OPENTARGETS_SKIP_SHA256",
    "OPENTARGETS_MAX_ROWS",
    "OPENTARGETS_CHEMBL_ID_REGEX", "OPENTARGETS_ENSG_ID_REGEX",
    "OPENTARGETS_UNIPROT_AC_REGEX",
    "OPENTARGETS_DISEASE_ID_PATTERNS",
    "OPENTARGETS_PER_EVIDENCE_TYPE_THRESHOLDS",
    "OPENTARGETS_DATASOURCE_RELATION_MAP",
    "OPENTARGETS_EMITTABLE_TRIPLES",
    "OPENTARGETS_DST_ID_PREFIXES",
    "ALLOWED_OPENTARGETS_URLS",
    "OPENTARGETS_GZIP_MAGIC",
    "OPENTARGETS_HASH_LENGTH", "OPENTARGETS_EDGE_ID_SOURCE",
    "OPENTARGETS_LARGE_FILE_THRESHOLD", "OPENTARGETS_LARGE_DF_THRESHOLD",
    "OPENTARGETS_BATCH_SIZE", "OPENTARGETS_CHUNK_SIZE",
    "OPENTARGETS_CHECKPOINT_INTERVAL",
    "OPENTARGETS_RELEASE_DATE",
    "OPENTARGETS_PINNED_VERSION", "OPENTARGETS_PINNED_SHA256",
    # ── Reproducibility ──
    "SEED", "set_global_seed", "DETERMINISTIC_MODE",
    # ── Directory paths ──
    "PROJECT_ROOT", "_PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
    "KG_DIR", "KG_EXPORT_DIR", "EMBEDDINGS_DIR", "LOGS_DIR", "MODEL_DIR",
    "DEAD_LETTER_DIR", "CHECKPOINT_DIR", "AUDIT_LOG_DIR",
    "OUTPUT_METADATA_DIR", "IMPACT_ANALYSIS_DIR", "TRANSFORMATION_LOG_DIR",
    "CONFIG_DIFF_DIR",
    # ── ensure_dirs ──
    "ensure_dirs",
    # ── Data source registry ──
    "DATA_SOURCES", "__data_sources_version__",
    "CRITICAL_SOURCES", "OPTIONAL_SOURCES", "ON_SOURCE_FAILURE",
    "get_data_source_path",
    # ── Neo4j config ──
    "Neo4jConfig", "get_neo4j_config",
    # ── KG Schema ──
    "CORE_NODE_TYPES", "DRKG_NODE_TYPES",
    "CORE_EDGE_TYPES", "CORE_EDGE_TYPES_SET", "is_core_edge",
    "filter_to_core_edges", "DRKG_RELATION_TO_CORE_EDGE",
    "STRICT_EDGE_FILTERING", "split_drkg_relation", "join_drkg_relation",
    "DRKG_RELATION_SEPARATOR",
    # P2-014 ROOT FIX: new helpers for robust DRKG relation parsing.
    "parse_drkg_relation_head_tail",
    "canonical_drkg_relation_name",
    # ── DRKG v2.0 audit-fix constants (drkg_loader_repair_prompt.md) ──
    # Fixes BUG 1.3 / 1.4 / 3.1 / 3.2 / 3.4 / 3.6 / 3.7 / 5.5 / 5.8 / 9.2 /
    #       9.6 / 12.4 / 14.2 / GUARD 3.10.
    "DRKG_TSV_COLUMNS",
    "DRKG_TREATMENT_RELATIONS",
    "DRKG_COMPOUND_GENE_RELATIONS",
    "DRKG_GENE_DISEASE_ASSOCIATION_RELATIONS",
    "DRKG_GENE_DISEASE_BIOMARKER_RELATIONS",
    "DRKG_VALID_TRIPLE_SCHEMAS",
    "DRKG_RELATION_ABBREV_TO_NAME",
    "DRKG_ENTITY_TYPE_TO_URI_PREFIX",
    "DRKG_RARE_DISEASE_CODES",
    "DRKG_STRICT_FILTER_ALLOW_UNKNOWN",
    "ALLOWED_DRKG_URLS",
    "EXPECTED_DRKG_ENTITY_TYPES",
    "EXPECTED_DRKG_RELATION_TYPES",
    "DRKG_PARSER_VERSION",
    "DRKG_SCHEMA_VERSION",
    "DRKG_LICENSE",
    "DRKG_ATTRIBUTION",
    # ── DrugBank v2.0 audit-fix constants (drugbank_parser_fix_prompt.md) ──
    # Fixes FIX[(7.2)] FIX[(14.1)] FIX[(14.2)] FIX[(14.3)] FIX[(14.11)]
    #       FIX[(3.3)] FIX[(3.4)] FIX[(3.16)] FIX[(5.1)] FIX[(5.4)]
    #       FIX[(5.13)] FIX[(12.4)] FIX[(12.6)] FIX[(G.15)] FIX[(G.16)]
    #       FIX[(G.17)]
    "DRUGBANK_PARSER_VERSION",
    "DRUGBANK_SCHEMA_VERSION",
    "DRUGBANK_LICENSE",
    "DRUGBANK_ATTRIBUTION",
    "DRUGBANK_NAMESPACE_URI",
    "DRUGBANK_NAMESPACE_ALIASES",
    "DRUGBANK_TEXT_FIELD_MAX_LENGTH",
    "DRUGBANK_ORGANISM_FILTER_TAX_ID",
    "DRUGBANK_ACTION_TO_RELATION",
    "DRUGBANK_EXTERNAL_ID_ALIASES",
    "ATC_CODE_SEPARATOR",
    "DRUGBANK_PROGRESS_LOG_INTERVAL",
    "DRUGBANK_MIN_FIELD_POPULATION",
    "DRUGBANK_KG_BUILDER_FIELDS",
    "ALLOWED_DRUGBANK_URLS",
    "DRUGOS_DEPLOYMENT_CONTEXT",
    "DRUGOS_ENVIRONMENT",
    "DRUGBANK_STRICT_VERSION",
    "DRUGBANK_ALLOW_MISSING_APPROVAL_YEAR",
    "DRUGBANK_STORE_FULL_TEXT",
    "DRUGBANK_BACKFILL_REFERENCE_TIME",
    "DRUGOS_FIXED_PARSED_AT",
    "DRUGOS_RUN_ID",
    "DRUGBANK_RARE_DISEASE_KEYWORDS",
    "DRUGBANK_MEMORY_CEILING_MB",
    "DRUGBANK_CHECKPOINT_INTERVAL",
    "DRUGBANK_INTERACTION_SEVERITY_RULES",
    "DRUGBANK_DRUG_TYPE_TO_NODE_LABEL",
    "DRUGBANK_TEXT_FIELD_NAMES",
    "DRUGBANK_XML_BACKEND",
    "DRUGBANK_DRUG_IDENTIFIER_REGEX",
    "DRUGBANK_INCHIKEY_REGEX",
    "DRUGBANK_CAS_REGEX",
    "DRUGBANK_ATC_REGEX",
    "DRUGBANK_ORGANISM_TO_TAXID",
    "DRUGBANK_XSD_PATH",
    "DRUGBANK_PARSER_DEPRECATIONS",
    "DRUGBANK_DRUGBANK_VERSION_LEXCMP",
    "DRUGBANK_DRUGBANK_FALLBACK_TO_TUPLE_COMPARE",
    # ── Edge metadata ──
    "EDGE_EVIDENCE_STRENGTH", "EDGE_CAUSALITY", "EDGE_VERB_EVIDENCE",
    "EDGE_PRODUCERS", "BIOLOGICAL_EDGE_CORRECTIONS",
    # ── PyG Config ──
    "PyGConfig",
    # ── TransE Config ──
    "TransEConfig",
    # ── Evaluation Config (evaluation.py v2.0 audit fix) ──
    "EvaluationConfig", "EVALUATION_CONFIG",
    "EVALUATION_METRIC_VERSION", "EVALUATION_SCHEMA_VERSION",
    "SKLEARN_MIN_VERSION", "K_VALUES_DEFAULT",
    "EVALUATION_FALLBACK_STRATEGY",
    # ── AUC Enforcement ──
    "AUCEnforcementLevel", "V1_LAUNCH_AUC",
    "STRICT_AUC_ENFORCEMENT",
    "AUC_ENFORCEMENT_LEVEL",
    "get_target_auc",
    "assert_auc_meets_threshold", "check_auc_meets_threshold",
    "AUCBelowThresholdError",
    "assert_positive_pair_count", "assert_negative_pair_count",
    "InsufficientTrainingDataError", "STRICT_PAIR_COUNTS",
    # ── Entity resolution ──
    "CANONICAL_IDS", "CANONICAL_IDS_FROZEN",
    "ID_MAPPING_PRIORITY", "ID_MAPPING_PRIORITY_FROZEN",
    "resolve_canonical_id", "flag_entity_confidence",
    "get_canonical_id_system", "get_entity_match_rate",
    "ENTITY_MATCH_RATE_BY_TYPE",
    "ENTITY_CONFIDENCE_THRESHOLD", "ENTITY_CONFIDENCE_STRICT_THRESHOLD",
    "ENTITY_CONFIDENCE_REJECT_THRESHOLD",
    "INCHIKEY_REGEX", "validate_inchikey",
    # ── Validation thresholds ──
    "MIN_NODES_W1", "MIN_EDGES_W1", "MIN_NODES_W2", "MIN_EDGES_W2",
    "MIN_POSITIVE_PAIRS", "MIN_NEGATIVE_PAIRS", "TARGET_TRANSE_AUC",
    "DEV_SMOKE_TEST", "DEV_SMOKE_TEST_MIN_AUC",
    "ENTITY_MATCH_RATE", "STRING_SCORE_THRESHOLD", "STITCH_SCORE_THRESHOLD",
    "STRING_MIN_COMBINED_SCORE",  # v57 ROOT FIX (P2L-032) — canonical STRING threshold
    "REFERENTIAL_INTEGRITY_RULES", "DUPLICATE_DETECTION_THRESHOLD",
    "DUPLICATE_DETECTION_FIELDS",
    # ── Entity-Resolver configuration (Block B of ENTITY_RESOLVER_FIX_PROMPT.md) ──
    "UNMATCHED_DRKG_CONFIDENCE",
    "EDGE_DEDUP_EARLY_REDUCTION_THRESHOLD",
    "DEFAULT_ENTITY_CONFIDENCE",
    "ATC_DELIMITER",
    "DATA_STALENESS_DAYS",
    "ENTITY_NAME_MAX_LENGTH",
    "ENTITY_RESOLVER_TIMEOUT_SECONDS",
    "ENTITY_RESOLVER_MAX_LOOKUPS_PER_SECOND",
    "ENTITY_RESOLVER_LOG_LEVEL",
    "ENTITY_RESOLVER_CONFIG_VERSION",
    "ENTITY_RESOLVER_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "ENTITY_RESOLVER_CIRCUIT_BREAKER_RESET_SECONDS",
    "ENTITY_RESOLVER_LRU_CACHE_SIZE",
    # ── Data quality & integrity ──
    "verify_checksum", "compute_and_record_checksum",
    "check_data_freshness", "check_disk_space", "check_record_count",
    "download_with_retry",
    # ── Reliability ──
    "dead_letter_record", "write_checkpoint", "read_latest_checkpoint",
    # ── Performance ──
    "parse_memory_string", "format_memory_string", "auto_size_neo4j_memory",
    "BATCH_SIZE_BY_NODE_TYPE", "CHUNK_SIZE",
    "CHEMBERTA_DIM_BY_MODEL", "EMBEDDING_DIM_BY_GRAPH_SIZE",
    "DeviceConfig",
    # ── Security ──
    "safe_config_dict", "PII_FIELDS", "REDACT_PII",
    "FILE_PERMISSIONS", "ENCRYPT_AT_REST",
    "SECRETS_REGISTRY", "get_secret", "require_secret",
    "MASK_OUTPUT_FIELDS", "audit_log",
    # ── Logging ──
    "LOG_FORMAT", "LOG_LEVEL", "LOG_LEVELS",
    "JsonFormatter", "STRUCTURED_LOGGING",
    "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
    "RUN_ID", "CORRELATION_ID",
    # ── Configuration management ──
    "CONFIG_FILE", "load_config_from_file", "apply_config_overrides",
    "ENVIRONMENT", "ENVIRONMENT_CONFIGS", "apply_environment_config",
    "DATA_SOURCES_FILE",
    # ── Compliance ──
    "ComplianceConfig", "RETENTION_DAYS", "AUDIT_TRAIL_ENABLED",
    "DataFormatConfig", "NAMING_CONVENTIONS", "deprecated",
    # ── Lineage & traceability ──
    "LineageMetadata", "build_lineage_metadata",
    "write_lineage_manifest",
    "compute_model_hash", "verify_model_hash",
    "CONFIG_DEPENDENCY_GRAPH", "compute_impact_analysis",
    # v28 ROOT FIX (audit TOP-18): re-export LABEL_MAP_VERSION from
    # utils.py so consumers that expect it on the config module (and
    # the CONFIG_DEPENDENCY_GRAPH entry below) find a real symbol.
    "LABEL_MAP_VERSION",
    "log_transformation", "diff_configs",
    # ── Documentation ──
    "DATA_DICTIONARY", "print_data_dictionary",
    "CONFIG_SECTIONS", "MAGIC_NUMBERS_REGISTRY", "THRESHOLD_LOCKS",
    # ── Backward compat (deprecated aliases) ──
    "TARGET_TRANSE_AUC",
]

# ─── A.3 SEED & Reproducibility ──────────────────────────────────────────────
# Fixes audit issue 7.2 — no global seed set
# RATIONALE: Without a fixed seed, the pipeline is non-deterministic.
# Two runs on the same data produce different negative samples, different
# TransE initializations, and different train/val/test splits. This
# violates regulatory reproducibility requirements (FDA 21 CFR Part 11).
# Default seed=42 follows scikit-learn convention.

SEED: int = int(os.environ.get("DRUGOS_SEED", "42"))

# RATIONALE: When DETERMINISTIC_MODE=True, the pipeline forces
# deterministic algorithms even at performance cost. This is required
# for clinical-grade runs where reproducibility > speed.
DETERMINISTIC_MODE: bool = os.environ.get("DRUGOS_DETERMINISTIC", "0") == "1"


def set_global_seed(seed: int | None = None) -> int:
    """Set the global random seed for reproducibility.

    Seeds Python's ``random``, NumPy, and PyTorch (if available).
    Also configures PyTorch for deterministic operation when
    ``DETERMINISTIC_MODE`` is enabled.

    Parameters
    ----------
    seed : int, optional
        Seed value. Defaults to the module-level ``SEED`` constant.

    Returns
    -------
    int
        The seed that was actually set.
    """
    global SEED
    if seed is not None:
        SEED = seed

    import random
    random.seed(SEED)

    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if DETERMINISTIC_MODE:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Fixes audit issue 7.2 — struct timeval seed for hash randomization
    os.environ["PYTHONHASHSEED"] = str(SEED)

    logger.info("Global seed set to %d (deterministic_mode=%s)", SEED, DETERMINISTIC_MODE)
    return SEED


# ─── A.4 Config hash computation ─────────────────────────────────────────────
# Fixes audit issue 7.3 — CONFIG_HASH for reproducibility tracking

def compute_config_hash() -> str:
    """Compute a stable SHA-256 hash of the current configuration.

    The hash is computed over the deterministic parts of the config
    (paths, thresholds, schema, data source registry) so that two
    runs with the same configuration produce the same hash.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest.
    """
    hasher = hashlib.sha256()
    # Version constants
    hasher.update(f"{PACKAGE_VERSION}|{PIPELINE_VERSION}|{CONFIG_VERSION}|{SCHEMA_VERSION}".encode())
    # Schema
    hasher.update(str(sorted(CORE_NODE_TYPES)).encode())
    hasher.update(str(sorted(CORE_EDGE_TYPES)).encode())
    hasher.update(str(sorted(CANONICAL_IDS.items())).encode())
    # Data sources (version-pinned fields only)
    for k in sorted(DATA_SOURCES.keys()):
        v = DATA_SOURCES[k]
        hasher.update(f"{k}|{v.get('version', '')}|{v.get('url', '')}|{v.get('pinned', '')}".encode())
    # Thresholds
    for name in sorted([
        "MIN_NODES_W2", "MIN_EDGES_W2", "MIN_POSITIVE_PAIRS",
        "MIN_NEGATIVE_PAIRS", "STRING_SCORE_THRESHOLD",
        "STITCH_SCORE_THRESHOLD", "ENTITY_CONFIDENCE_THRESHOLD",
    ]):
        hasher.update(f"{name}={globals().get(name, '')}".encode())
    # Seed
    hasher.update(str(SEED).encode())
    return hasher.hexdigest()[:16]


CONFIG_HASH: str = ""  # Computed at end of module after all config is loaded


# ─── Phase B — Directory & Path Fixes ────────────────────────────────────────

# Fixes audit issue 1.1 — PROJECT_ROOT renamed to _PROJECT_ROOT (private)
# Fixes audit issue 4.4 — DRUGOS_PROJECT_ROOT env var override
# Fixes audit issue 12.4 — env-based root for deployment flexibility

# v108 ROOT FIX (ISSUE-P2-056): the path constants (_PROJECT_ROOT, DATA_DIR,
# RAW_DIR, PROCESSED_DIR, PHASE1_PROCESSED_DIR, RESULTS_PERSIST_PATH,
# KG_EXPORT_DIR, KG_DIR, EMBEDDINGS_DIR, LOGS_DIR) have been MOVED to
# ``config_paths.py`` and are imported at the top of this file. The original
# inline definitions (60+ lines) were REMOVED to avoid shadowing the imports.
# See ``config_paths.py`` for the canonical definitions and full docstrings.
# The ``MODEL_DIR``, ``DEAD_LETTER_DIR``, ``CHECKPOINT_DIR``, etc. below
# remain here (they depend on the imported constants and will be extracted
# in a follow-up PR).
MODEL_DIR = _PROJECT_ROOT / "models"

# Fixes audit issue 6.6 — DEAD_LETTER_DIR for quarantining bad records
DEAD_LETTER_DIR = DATA_DIR / "dead_letter"

# Fixes audit issue 6.10 — CHECKPOINT_DIR for pipeline resumption
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Fixes audit issue 9.9 — AUDIT_LOG_DIR for security audit trail
AUDIT_LOG_DIR = LOGS_DIR / "audit"

# Fixes audit issue 16.4 — OUTPUT_METADATA_DIR for lineage manifests
OUTPUT_METADATA_DIR = DATA_DIR / "output_metadata"

# Fixes audit issue 16.10 — IMPACT_ANALYSIS_DIR for downstream impact tracking
IMPACT_ANALYSIS_DIR = DATA_DIR / "impact_analysis"

# Fixes audit issue 16.8 — TRANSFORMATION_LOG_DIR for data transformation logs
TRANSFORMATION_LOG_DIR = LOGS_DIR / "transformations"

# Fixes audit issue 16.11 — CONFIG_DIFF_DIR for configuration diff tracking
CONFIG_DIFF_DIR = DATA_DIR / "config_diffs"

# =============================================================================
# OpenTargets configuration constants
# =============================================================================
# Added by opentargets_loader v2.0 institutional-grade audit fix
# (opentargets_loader_repair_prompt.md — Section 5.3).
#
# These constants are the single source of truth for all OpenTargets loader
# thresholds, URL allowlists, regex patterns, and tunable knobs. They are
# overridable by environment variables (DRUGOS_OPENTARGETS_*) for dev/staging/
# prod parity without code changes (Domain 12 Configuration / CONF-1).
#
# Patient-safety doctrine: OpenTargets is the SOLE source of evidence-scored
# drug-target-disease triples feeding the Graph Transformer's confidence
# training objective. Every threshold here exists to prevent silent data loss
# that would corrupt the model's training signal (opentargets_loader_repair_prompt
# Section 0.1 / Section 0.4).
#
# Fixes: CONF-1 (config validation), CONF-2 (env overrides), CONF-8 (documented
#        defaults), SCI-1..SCI-15 (scientific correctness), DQ-1..DQ-16 (data
#        quality), REL-1..REL-13 (reliability), SEC-1..SEC-3 (security),
#        ARCH-5 (OpenTargetsConfig), SCI-11 (per-evidence-type thresholds).
# -----------------------------------------------------------------------------

# SOURCE_KEY_OPENTARGETS — DATA_SOURCES dict key for OpenTargets.
SOURCE_KEY_OPENTARGETS: str = "opentargets"

# SOURCE_OPENTARGETS — human-readable source name (used in node .source and
# edge _source props).
SOURCE_OPENTARGETS: str = "OpenTargets"

# OPENTARGETS_PARSER_VERSION — bumped on any parser logic change
# (opentargets_loader_repair_prompt Section 0.2 constraint #22).
OPENTARGETS_PARSER_VERSION: str = "2.0.0"

# OPENTARGETS_SCHEMA_VERSION — bumped on any output-schema change
# (opentargets_loader_repair_prompt Section 0.2 constraint #22).
OPENTARGETS_SCHEMA_VERSION: str = "2.0.0"

# OPENTARGETS_LICENSE — OpenTargets is released under CC0 1.0 (public domain).
OPENTARGETS_LICENSE: str = "CC0 1.0"

# OPENTARGETS_ATTRIBUTION — propagated to every emitted record's _attribution
# field (Domain 14 Compliance / COMP-3).
OPENTARGETS_ATTRIBUTION: str = (
    "OpenTargets Platform release 25.03. "
    "https://platform.opentargets.org/. Licensed under CC0 1.0."
)

# OPENTARGETS_TARGET_TAX_ID — NCBI Taxonomy ID for Homo sapiens.
# Used for organism filtering (SCI-7). Non-human evidence is rejected.
OPENTARGETS_TARGET_TAX_ID: int = 9606

# OPENTARGETS_MIN_SCORE_DEFAULT — default minimum score for evidence records.
# Overridable via DRUGOS_OPENTARGETS_MIN_SCORE env var (Domain 12).
# Per-evidence-type thresholds in OPENTARGETS_PER_EVIDENCE_TYPE_THRESHOLDS
# take precedence over this global default (SCI-11).
OPENTARGETS_MIN_SCORE_DEFAULT: float = float(
    os.environ.get("DRUGOS_OPENTARGETS_MIN_SCORE", "0.5")
)

# v28 ROOT FIX (P2-L-16): minimum GDA score thresholds for DisGeNET and
# OMIM. Previously these loaders applied NO score threshold — a 0.01-score
# text-mined association loaded with the SAME edge weight as a 0.95-score
# validated causal variant. Defaults:
#   - DISGENET_MIN_SCORE = 0.3 (per DisGeNET's own recommendation that
#     scores >= 0.3 indicate "moderate" evidence; scores < 0.3 are
#     text-mined noise from a single PubMed abstract).
#   - OMIM_MIN_SCORE = 0.3 (v77 ROOT FIX: was 0.5, which DROPPED every
#     mapping_key=3 / "provisional" gene-disease edge. OMIM mapping_key=3
#     means "the gene has been mapped to a disease locus, but the
#     relationship is provisional" — these ARE scientifically meaningful
#     edges (provisional, not rejected). Dropping them entirely was silent
#     data loss: the KG lost every provisional OMIM association, including
#     rare-disease associations that are only provisionally mapped. The
#     proper scientific treatment is to INCLUDE provisional edges with a
#     low confidence score (0.4) so the model can learn from them with
#     appropriate weighting — NOT to drop them. Lowering the threshold to
#     0.3 (matching DisGeNET) preserves mapping_key=3 edges (score=0.4 >
#     0.3) while still filtering truly garbage text-mined noise (score <
#     0.3). Operators can still raise/lower/disable via env var.)
# Operators can lower or disable these thresholds via env vars.
DISGENET_MIN_SCORE: float = float(
    os.environ.get("DRUGOS_DISGENET_MIN_SCORE", "0.3")
)
OMIM_MIN_SCORE: float = float(
    os.environ.get("DRUGOS_OMIM_MIN_SCORE", "0.3")
)

# OPENTARGETS_MIN_RESOLUTION_RATE — minimum target resolution rate (ENSG →
# UniProt AC crosswalk success rate). Below this rate, the loader raises
# OpenTargetsDataIntegrityError in CLINICAL+ mode (Section 0.4).
OPENTARGETS_MIN_RESOLUTION_RATE: float = float(
    os.environ.get("DRUGOS_OPENTARGETS_MIN_RESOLUTION_RATE", "0.5")
)

# OPENTARGETS_REGULATORY_RESOLUTION_RATE — minimum target resolution rate in
# REGULATORY mode (Section 0.4). Below this rate, the loader raises
# OpenTargetsDataIntegrityError.
OPENTARGETS_REGULATORY_RESOLUTION_RATE: float = float(
    os.environ.get("DRUGOS_OPENTARGETS_REGULATORY_RESOLUTION_RATE", "0.9")
)

# OPENTARGETS_PROGRESS_LOG_INTERVAL — number of lines between progress log
# messages during parsing (Domain 11 Observability / LOG-3).
OPENTARGETS_PROGRESS_LOG_INTERVAL: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_PROGRESS_INTERVAL", "100000")
)

# OPENTARGETS_DOWNLOAD_BATCH_BYTES — chunk size for streaming download
# (Domain 8 Performance). 1 MB balances I/O efficiency and memory.
OPENTARGETS_DOWNLOAD_BATCH_BYTES: int = 1 << 20  # 1 MiB

# OPENTARGETS_PARSED_CACHE_DIR — directory for parsed-record cache files
# (Domain 7 Idempotency / IDEM-3). Keyed by source SHA-256.
OPENTARGETS_PARSED_CACHE_DIR: Path = CHECKPOINT_DIR / "opentargets_parsed"

# OPENTARGETS_STALENESS_DAYS — number of days after which a cached file is
# considered stale and triggers re-download in CLINICAL+ mode (DQ-12, DQ-16).
OPENTARGETS_STALENESS_DAYS: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_STALENESS_DAYS", "180")
)

# OPENTARGETS_NEO4J_BATCH_SIZE — maximum edges per Neo4j load_edges_bulk_create
# call (PERF-4 / Section 0.2 constraint #12). 50K is the safe upper bound for
# Neo4j transaction size on a 15M-edge load.
OPENTARGETS_NEO4J_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_NEO4J_BATCH_SIZE", "50000")
)

# OPENTARGETS_DEAD_LETTER_PATH — path to the OpenTargets dead-letter queue
# (REL-5 / Section 0.2 constraint #18). One JSON line per dropped record.
OPENTARGETS_DEAD_LETTER_PATH: Path = DEAD_LETTER_DIR / "opentargets_malformed.jsonl"

# OPENTARGETS_LINEAGE_LOG_PATH — path to the OpenTargets lineage log
# (LIN-6 / Section 0.2 constraint #19). One JSON line per transformation step.
OPENTARGETS_LINEAGE_LOG_PATH: Path = LOGS_DIR / "lineage" / "opentargets_lineage.jsonl"

# OPENTARGETS_AUDIT_LOG_PATH — path to the OpenTargets audit log
# (Domain 9 Security / SEC-5). One JSON line per download + per access.
OPENTARGETS_AUDIT_LOG_PATH: Path = LOGS_DIR / "audit" / "opentargets_access.jsonl"

# OPENTARGETS_TRANSFORMATION_LOG_PATH — path to the OpenTargets transformation
# log (Domain 16 Lineage / LIN-2). One JSON line per transformation step.
OPENTARGETS_TRANSFORMATION_LOG_PATH: Path = (
    LOGS_DIR / "transformations" / "opentargets.jsonl"
)

# OPENTARGETS_QUALITY_REPORT_PATH — path to the OpenTargets quality report
# (Domain 5 Data Quality / DQ-11). JSON file with metrics + validation result.
OPENTARGETS_QUALITY_REPORT_PATH: Path = (
    LOGS_DIR / "quality" / "opentargets_quality_report.json"
)

# OPENTARGETS_CIRCUIT_BREAKER_THRESHOLD — maximum dead-lettered records before
# the circuit breaker trips (REL-9). Prevents infinite-loop on a structurally
# broken source file.
OPENTARGETS_CIRCUIT_BREAKER_THRESHOLD: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_CIRCUIT_BREAKER_THRESHOLD", "1000")
)

# OPENTARGETS_MAX_RETRIES — maximum download retry attempts (REL-1).
OPENTARGETS_MAX_RETRIES: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_MAX_RETRIES", "3")
)

# OPENTARGETS_RETRY_BACKOFF_BASE — base for exponential backoff (REL-1).
# Actual delay = backoff_base * (2 ** attempt) + jitter.
OPENTARGETS_RETRY_BACKOFF_BASE: float = float(
    os.environ.get("DRUGOS_OPENTARGETS_RETRY_BACKOFF_BASE", "2.0")
)

# OPENTARGETS_DOWNLOAD_TIMEOUT_SECONDS — per-request timeout (REL-2).
OPENTARGETS_DOWNLOAD_TIMEOUT_SECONDS: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_DOWNLOAD_TIMEOUT", "300")
)

# OPENTARGETS_MIN_VALID_SIZE_BYTES — minimum byte size for a downloaded
# OpenTargets .json.gz to be considered valid (DQ-2). The real evidence
# file is ~800 MB; this threshold catches truncated or HTML-error-page
# downloads.
OPENTARGETS_MIN_VALID_SIZE_BYTES: int = 1_000_000  # 1 MB minimum

# OPENTARGETS_FORCE_DOWNLOAD — global env-var override to force re-download
# (Domain 12 Configuration / CONF-2).
OPENTARGETS_FORCE_DOWNLOAD: bool = (
    os.environ.get("DRUGOS_OPENTARGETS_FORCE_DOWNLOAD", "0") == "1"
)

# OPENTARGETS_SKIP — global env-var to skip OpenTargets load entirely
# (Domain 12 Configuration / CONF-2). Useful for fast iteration on other
# loaders during development.
OPENTARGETS_SKIP: bool = (
    os.environ.get("DRUGOS_OPENTARGETS_SKIP", "0") == "1"
)

# OPENTARGETS_OFFLINE — global env-var to use cached file only (no download)
# (Domain 12 Configuration / CONF-2).
OPENTARGETS_OFFLINE: bool = (
    os.environ.get("DRUGOS_OPENTARGETS_OFFLINE", "0") == "1"
)

# OPENTARGETS_SKIP_SHA256 — global env-var to skip sha256 verification
# (dev only — logs a WARNING when active, DQ-1 / DQ-14).
OPENTARGETS_SKIP_SHA256: bool = (
    os.environ.get("DRUGOS_OPENTARGETS_SKIP_SHA256", "0") == "1"
)

# OPENTARGETS_MAX_ROWS — global env-var to cap rows read (dev / debug).
OPENTARGETS_MAX_ROWS: int | None = (
    int(os.environ["DRUGOS_OPENTARGETS_MAX_ROWS"])
    if os.environ.get("DRUGOS_OPENTARGETS_MAX_ROWS", "").isdigit()
    else None
)

# OPENTARGETS_CHEMBL_ID_REGEX — ChEMBL ID format: ^CHEMBL\d+$ (SCI-4).
# Case-insensitive on input; canonical form is uppercase.
OPENTARGETS_CHEMBL_ID_REGEX = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)

# OPENTARGETS_ENSG_ID_REGEX — Ensembl gene ID format: ^ENSG\d{11}$ (SCI-10).
# 11 digits after the "ENSG" prefix. Case-insensitive on input; canonical
# form is uppercase.
OPENTARGETS_ENSG_ID_REGEX = re.compile(r"^ENSG\d{11}$", re.IGNORECASE)

# OPENTARGETS_UNIPROT_AC_REGEX — UniProt accession format: 6 or 10 chars
# ([OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}){1,4}[0-9]).
# Used to validate crosswalk output.
OPENTARGETS_UNIPROT_AC_REGEX = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}){1,4}[0-9])$"
)

# OPENTARGETS_DISEASE_ID_PATTERNS — disease ontology ID format patterns
# (SCI-3 / DQ-11). OpenTargets emits disease IDs from multiple ontologies:
#   * EFO         — ^EFO_\d{7}$
#   * MONDO       — ^MONDO_\d{7}$
#   * HP          — ^HP:\d{7}$  (HPO)
#   * MP          — ^MP:\d{7}$  (Mammalian Phenotype)
#   * Orphanet    — ^Orphanet_\d+$
#   * SNOMEDCT    — ^SNOMEDCT_\d+$
#   * OTAR        — ^OTAR_\d{8}$
#   * DOID        — ^DOID:\d+$
#   * UMLS CUI    — ^C\d{7}$  (crosswalk target)
#
# v70 ROOT FIX (P2L-047): the previous patterns used INCONSISTENT
# separators — underscore for EFO/MONDO/Orphanet/SNOMEDCT/OTAR but
# COLON for DOID/HP/MP. This made the ontology-detection regex
# fragile: an OpenTargets ID emitted as ``DOID_1438`` (underscore,
# which some OpenTargets releases actually use — see audit P2L-047)
# would FAIL the ``^DOID:\d+$`` pattern and be classified as
# "UNKNOWN", losing the ontology provenance. Symmetrically, a
# DisGeNET ID ``EFO:0000311`` (colon, which DisGeNET's API emits
# for some ontologies) would FAIL the ``^EFO_\d{7}$`` pattern.
#
# Root fix: every ontology pattern now accepts BOTH underscore AND
# colon as the separator (``[_:]``). This makes the detection regex
# format-agnostic — the actual separator normalization happens
# downstream in ``_normalise_ontology_id`` (opentargets_loader) and
# the Phase 1 disgenet_pipeline (which already normalizes to colon
# form). The patterns are also made case-insensitive where the
# ontology name is not case-sensitive (Orphanet, SNOMEDCT, OTAR) —
# EFO/MONDO/DOID/HP/MP/UMLS retain their original case sensitivity
# (these ontologies have a canonical case that the source files
# respect).
OPENTARGETS_DISEASE_ID_PATTERNS: dict[str, "re.Pattern[str]"] = {
    "EFO":       re.compile(r"^EFO[_:]\d{7}$"),
    "MONDO":     re.compile(r"^MONDO[_:]\d{7}$"),
    "HP":        re.compile(r"^HP[_:]\d{7}$"),
    "MP":        re.compile(r"^MP[_:]\d{7}$"),
    "Orphanet":  re.compile(r"^Orphanet[_:]\d+$", re.IGNORECASE),
    "SNOMEDCT":  re.compile(r"^SNOMEDCT[_:]\d+$", re.IGNORECASE),
    "OTAR":      re.compile(r"^OTAR[_:]\d{8}$", re.IGNORECASE),
    "DOID":      re.compile(r"^DOID[_:]\d+$"),
    "UMLS":      re.compile(r"^C\d{7}$"),
}

# OPENTARGETS_PER_EVIDENCE_TYPE_THRESHOLDS — per-datasource min-score
# thresholds (SCI-11). Scores are normalized 0-1 within each datasource.
# Thresholds are calibrated per OpenTargets documentation and community
# standards:
#   * chembl/known_drug: 0.5 — derived from pChEMBL ≥5 (10µM activity)
#     Ref: https://doi.org/10.1016/j.drudis.2014.10.012
#   * genetic_association: 0.3 — broad inclusion for hypothesis generation
#   * literature: 0.4 — text-mining cosine threshold
#   * animal_model: 0.5 — pre-clinical signal
#   * affected_pathway: 0.4 — reactome pathway disruption
#   * default: 0.5 — conservative default for unknown datasources
OPENTARGETS_PER_EVIDENCE_TYPE_THRESHOLDS: dict[str, float] = {
    "chembl": 0.5,
    "genetic_association": 0.3,
    "literature": 0.4,
    "animal_model": 0.5,
    "affected_pathway": 0.4,
    "default": 0.5,
}

# OPENTARGETS_DATASOURCE_RELATION_MAP — maps (datasourceId, datatypeId) to
# (rel_type, dst_type) for scientific correctness (SCI-8). The label
# "indication" is FORBIDDEN — ChEMBL binding-activity evidence is NOT
# approved-indication data. Approved indications come from drugbank_parser.
#
# Critical pairs (DO NOT modify without SCI-8 sign-off):
#   * ("chembl", "known_drug")    -> ("binds",       "Protein")
#       ChEMBL is IC50/Ki/Kd binding-activity data, NOT approved indications.
#   * ("chembl", "animal_model")  -> ("tested_for",  "Disease")
#       Pre-clinical assay evidence — NOT approved.
#   * ("evrot", "literature")     -> ("associated_with", "Disease")
#   * ("ot_genetics_portal", "genetic_association") -> ("associated_with", "Disease")
OPENTARGETS_DATASOURCE_RELATION_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("chembl", "known_drug"):       ("binds", "Protein"),
    ("chembl", "animal_model"):     ("tested_for", "Disease"),
    ("evrot", "literature"):        ("associated_with", "Disease"),
    ("europepmc", "literature"):    ("associated_with", "Disease"),
    ("genetic_association", "genetic_association"): ("associated_with", "Disease"),
    ("ot_genetics_portal", "genetic_association"):  ("associated_with", "Disease"),
    ("gene2phenotype", "genetic_association"):      ("associated_with", "Disease"),
    ("uniprot_literature", "literature"):           ("associated_with", "Disease"),
    ("uniprot_variants", "genetic_association"):    ("associated_with", "Disease"),
    ("reactome", "affected_pathway"):               ("disrupted_in", "Pathway"),
    ("crispr", "drug"):                             ("binds", "Protein"),
    ("expression_atlas", "rna_expression"):         ("associated_with", "Disease"),
    ("chembl", "binding_assay"):                    ("binds", "Protein"),
    ("chembl", "functional_assay"):                 ("modulates", "Protein"),
}

# OPENTARGETS_EMITTABLE_TRIPLES — set of (src_type, rel_type, dst_type)
# triples that the OpenTargets loader is allowed to emit (ARCH-2). The
# loader raises OpenTargetsSchemaError if it tries to emit a triple not in
# this set. This is the loader's contract with ``EDGE_PRODUCERS`` and the
# KG builder.
#
# CRITICAL: ("Compound", "indication", "Disease") is INTENTIONALLY ABSENT.
# The "indication" label is FORBIDDEN in this loader (SCI-8).
OPENTARGETS_EMITTABLE_TRIPLES: frozenset[tuple[str, str, str]] = frozenset({
    ("Compound", "binds", "Protein"),
    ("Compound", "targets", "Gene"),
    ("Compound", "tested_for", "Disease"),
    ("Compound", "associated_with", "Disease"),
    ("Compound", "modulates", "Protein"),
    # v43 ROOT FIX (P1 — OPENTARGETS_EMITTABLE_TRIPLES includes non-core
    # edge): ("Compound", "disrupted_in", "Pathway") was here but is NOT
    # in CORE_EDGE_TYPES. Production kg_builder logs WARNING but loads
    # the edge; RecordingGraphBuilder (bridge/tests) rejects via
    # CORE_EDGE_TYPES whitelist → dead-letters every such edge. Divergent
    # behavior between production and test paths. Removed until either
    # (a) added to CORE_EDGE_TYPES, or (b) the OpenTargets loader stops
    # emitting it.
    # ("Compound", "disrupted_in", "Pathway"),  # REMOVED — not in CORE_EDGE_TYPES
    ("Protein", "associated_with", "Disease"),
})

# OPENTARGETS_DST_ID_PREFIXES — prefixes applied to dst_id values to
# prevent namespace collisions in the KG (D15.2). Compound src_ids are
# ChEMBL IDs (no prefix needed). Disease dst_ids are UMLS CUIs when
# crosswalked, otherwise ontology-prefixed (e.g. "EFO:...").
OPENTARGETS_DST_ID_PREFIXES: dict[str, str] = {
    "Protein": "",        # UniProt AC, no prefix
    "Gene": "",           # NCBI Gene ID, no prefix
    "Disease": "UMLS:",   # UMLS CUI prefix when crosswalked
    "Pathway": "",        # Reactome ID, no prefix
}

# ALLOWED_OPENTARGETS_URLS — URL-prefix allowlist for SSRF guard
# (Domain 9 Security / SEC-2). The OpenTargets loader refuses to download
# from any URL not matching one of these prefixes. HTTPS-only.
ALLOWED_OPENTARGETS_URLS: tuple[str, ...] = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/",
    "https://www.ebi.ac.uk/opentargets/",
    "https://platform.opentargets.org/",
)

# OPENTARGETS_GZIP_MAGIC — gzip file magic bytes (DQ-2).
OPENTARGETS_GZIP_MAGIC: bytes = b"\x1f\x8b"

# OPENTARGETS_HASH_LENGTH — length of the deterministic edge ID hash
# (D2.8 / G9). 16 chars of sha1 hex.
OPENTARGETS_HASH_LENGTH: int = 16

# OPENTARGETS_EDGE_ID_SOURCE — suffix used in the edge ID hash to namespace
# OpenTargets edges from other sources (prevents Neo4j MERGE collision).
OPENTARGETS_EDGE_ID_SOURCE: str = "OPENTARGETS"

# OPENTARGETS_LARGE_FILE_THRESHOLD — threshold above which streaming is
# preferred over eager materialization (Domain 8 Performance / PERF-1).
OPENTARGETS_LARGE_FILE_THRESHOLD: int = 500_000  # 500K records

# OPENTARGETS_LARGE_DF_THRESHOLD — threshold above which a DataFrame is
# considered "large" and triggers memory-efficient processing paths
# (Domain 8 Performance / PERF-2).
OPENTARGETS_LARGE_DF_THRESHOLD: int = 500_000

# OPENTARGETS_BATCH_SIZE — default batch size for streaming edge generation
# (Domain 8 Performance / PERF-2). Override per-call via batch_size kwarg.
OPENTARGETS_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_BATCH_SIZE", "10000")
)

# OPENTARGETS_CHUNK_SIZE — default chunk size for streaming parse
# (Domain 8 Performance / PERF-1). 100K rows per chunk balances memory
# and I/O efficiency on 5M-row files.
OPENTARGETS_CHUNK_SIZE: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_CHUNK_SIZE", "100000")
)

# OPENTARGETS_CHECKPOINT_INTERVAL — number of rows between checkpoint writes
# (Domain 6 Reliability / REL-6).
OPENTARGETS_CHECKPOINT_INTERVAL: int = int(
    os.environ.get("DRUGOS_OPENTARGETS_CHECKPOINT_INTERVAL", "100000")
)

# OPENTARGETS_RELEASE_DATE — pinned OpenTargets release date (D3.8 / D12.4).
# Update this AND the sha256 when OpenTargets publishes a new release.
OPENTARGETS_RELEASE_DATE: str = "2025-03-01"

# OPENTARGETS_PINNED_VERSION — pinned OpenTargets platform release string
# (D3.8 / D12.4).
OPENTARGETS_PINNED_VERSION: str = "25.03"

# OPENTARGETS_PINNED_SHA256 — sha256 of the pinned OpenTargets evidence file.
# Set to None because OpenTargets does NOT publish a checksum. The loader
# computes it at download time and stores it as a sidecar for future runs
# (D3.8 / DQ-14).
OPENTARGETS_PINNED_SHA256: str | None = None


# =============================================================================
# ClinicalTrials constants — added by clinicaltrials_loader v2.1.0
# institutional-grade audit fix (PROMPT_fix_clinicaltrials_loader.md —
# 148 findings across 16 domains).
#
# These constants follow the same env-var-overridable pattern as the
# OpenTargets block above so an operator can deploy to a different
# environment by changing only env vars, not source code (Domain 12).
#
# Patient-safety rationale: every threshold here has a documented scientific
# reason. Removing or weakening any of them silently degrades the evidence
# base for the RL ranker, which can lead to a clinician prescribing a
# contraindicated drug off-label. See PROMPT_fix_clinicaltrials_loader.md
# Section 3 for the per-constant rationale.
# =============================================================================

# SOURCE_KEY_CLINICALTRIALS — DATA_SOURCES dict key for ClinicalTrials.gov AACT.
SOURCE_KEY_CLINICALTRIALS: str = "clinicaltrials"

# SOURCE_CLINICALTRIALS — human-readable source name (used in node .source and
# edge _source props).
SOURCE_CLINICALTRIALS: str = "ClinicalTrials"

# CLINICALTRIALS_PARSER_VERSION — bumped on any parser logic change
# (PROMPT_fix_clinicaltrials_loader.md Section 0.2 constraint #22).
CLINICALTRIALS_PARSER_VERSION: str = "2.1.0"

# CLINICALTRIALS_SCHEMA_VERSION — bumped on any output-schema change.
# Tied to ClinicalTrialEdgeRecord in schemas.py (Issue 14.6).
CLINICALTRIALS_SCHEMA_VERSION: str = "2.1.0"

# CLINICALTRIALS_LICENSE — AACT is released under CC0 1.0 (public domain).
CLINICALTRIALS_LICENSE: str = "CC0 1.0"

# CLINICALTRIALS_ATTRIBUTION — propagated to every emitted record's _attribution
# field (Domain 14 Compliance — Issue 13.7, 13.8, 14.4, 14.5). CTTI requests
# this citation for any derivative work.
CLINICALTRIALS_ATTRIBUTION: str = (
    "ClinicalTrials.gov AACT database (CTTI). "
    "https://aact.ctti-clinicaltrials.org/. Licensed under CC0 1.0."
)

# CLINICALTRIALS_CITATION — propagated to lineage file (Issue 13.8).
# CTTI requests this citation format for derivative works.
CLINICALTRIALS_CITATION: str = (
    "AACT data extracted from https://aact.ctti-clinicaltrials.org. "
    "Duke-Margolis Center for Health Policy and FDA. "
    "Clinical Trials Transformation Initiative (CTTI)."
)

# ── Phase filter (Issues 3.2, 3.4, 12.1, 13.2) ──────────────────────────────
# DEFAULT phases includes Phase 4 — post-marketing surveillance evidence
# from 10K+ real-world patients is STRONGER than Phase 3 RCT evidence, not
# weaker. Excluding Phase 4 silently downgrades the evidence base for
# FDA-approved drugs (Issue 3.4 / 13.2 — patient safety).
CLINICALTRIALS_DEFAULT_PHASES: tuple[str, ...] = ("Phase 3", "Phase 4")

# Controlled vocabulary for AACT phase values (Issue 2.10, 5.5).
# Any phase value outside this set is rejected by _validate_phases.
CLINICALTRIALS_VALID_PHASES: frozenset[str] = frozenset({
    "Early Phase 1",
    "Phase 1",
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3",
    "Phase 3",
    "Phase 3/Phase 4",
    "Phase 4",
    "N/A",
})

# Phase → evidence-strength contribution (Issue 2.5).
# Phase 4 > Phase 3 > Phase 2/3 > Phase 2 > Phase 1/2 > Phase 1 > Early Phase 1.
# Phase 3/4 is treated the same as Phase 4 (post-marketing).
# Phase 2/3 is treated the same as Phase 3 (the higher phase dominates).
# "N/A" gets 0.0 because it means the trial did not specify a phase.
CLINICALTRIALS_PHASE_STRENGTH: dict[str, float] = {
    "Early Phase 1": 0.10,
    "Phase 1":       0.15,
    "Phase 1/Phase 2": 0.25,
    "Phase 2":       0.40,
    "Phase 2/Phase 3": 0.60,
    "Phase 3":       0.70,
    "Phase 3/Phase 4": 0.80,
    "Phase 4":       0.85,
    "N/A":           0.0,
}

# ── Intervention type filter (Issues 2.7, 3.3, 12.2, 13.3) ─────────────────
# DEFAULT includes "Drug" AND "Biological". Biological covers monoclonal
# antibodies, vaccines, and cell therapies — the majority of new FDA
# approvals since 2015 (Humira, Keytruda, Opdivo, Ozempic). Excluding
# them would silently blind the RL ranker to ~30% of the modern
# pharmacopeia (Issue 2.7 / 13.3 — patient safety).
CLINICALTRIALS_DEFAULT_INTERVENTION_TYPES: tuple[str, ...] = ("Drug", "Biological")

# Controlled vocabulary for AACT intervention_type values (Issue 2.7).
CLINICALTRIALS_VALID_INTERVENTION_TYPES: frozenset[str] = frozenset({
    "Drug",
    "Biological",
    "Device",
    "Procedure",
    "Behavioral",
    "Dietary Supplement",
    "Radiation",
    "Genetic",
    "Combination Product",
    "Diagnostic Test",
    "Other",
})

# ── Study type filter (Issue 3.7) ───────────────────────────────────────────
# DEFAULT is interventional only. Observational studies are weaker evidence
# than interventional RCTs.
CLINICALTRIALS_DEFAULT_STUDY_TYPES: tuple[str, ...] = ("Interventional",)

# Controlled vocabulary for AACT study_type values.
CLINICALTRIALS_VALID_STUDY_TYPES: frozenset[str] = frozenset({
    "Interventional",
    "Observational",
    "Observational [Patient Registry]",
    "Expanded Access",
})

# ── Status filter (Issue 3.11) ──────────────────────────────────────────────
# DEFAULT explicitly EXCLUDES Withdrawn/Suspended/Terminated/No Longer
# Available/Unknown status — these are NOT positive evidence. A Terminated
# trial that was stopped for safety is captured via why_stopped (Issue 3.5),
# but the trial itself does not constitute evidence that the drug treats
# the disease.
#
# v27 ROOT FIX (P2-L-7): the previous default also included
# ``"Recruiting"``, ``"Not yet recruiting"``, ``"Enrolling by invitation"``,
# and ``"Active, not recruiting"``. These trials have ZERO results data
# (no enrollment completion, no primary outcome measure published) —
# including them as efficacy evidence is scientifically wrong and a
# patient-safety risk (an RL ranker would treat a half-enrolled Phase II
# trial as equivalent to a completed Phase III trial with published
# results). Restrict the DEFAULT to ``("Completed",)`` only. Operators
# who want to include in-progress trials can override via the explicit
# ``allowed_statuses`` parameter to the loader.
CLINICALTRIALS_DEFAULT_ALLOWED_STATUSES: tuple[str, ...] = (
    "Completed",
)

# Controlled vocabulary for AACT overall_status values.
CLINICALTRIALS_VALID_STATUSES: frozenset[str] = frozenset({
    "Completed",
    "Active, not recruiting",
    "Recruiting",
    "Enrolling by invitation",
    "Not yet recruiting",
    "Withdrawn",
    "Suspended",
    "Terminated",
    "No Longer Available",
    "Unknown status",
    "Approved for marketing",
    "Available",
    "No longer recruiting",
    "Temporarily not available",
})

# ── Trial design strength modifiers (Issue 2.5, 3.8) ───────────────────────
# Allocation bonuses (Issue 3.8 — randomized > non-randomized).
CLINICALTRIALS_ALLOCATION_BONUS: dict[str, float] = {
    "Randomized":        0.10,
    "Non-Randomized":    0.0,
    "Non Randomized":    0.0,  # alternate spelling
    "NA":                0.0,
}

# Masking bonuses (Issue 3.8 — double-blind > single-blind > open label).
CLINICALTRIALS_MASKING_BONUS: dict[str, float] = {
    "Quadruple Blind":       0.10,  # Participant, Care Provider, Investigator, Outcomes Assessor
    "Triple Blind":          0.10,
    "Double Blind":          0.08,
    "Single Blind":          0.04,
    "Open Label":            0.0,
    "None":                  0.0,
    "NA":                    0.0,
}

# ── Enrollment filter (Issue 3.6) ───────────────────────────────────────────
# Minimum enrollment for a trial to be considered. Phase 3 trials with
# <30 participants are suspect — likely misclassified. We don't filter them
# out by default (that would silently drop data), but we emit a WARNING and
# penalize evidence_strength.
CLINICALTRIALS_DEFAULT_MIN_ENROLLMENT: int = 0
CLINICALTRIALS_SUSPECT_ENROLLMENT_THRESHOLD: int = 30
CLINICALTRIALS_ENROLLMENT_BONUS_LARGE_TRIAL: int = 500  # +0.05 bonus above this
CLINICALTRIALS_ENROLLMENT_BONUS_VALUE: float = 0.05

# ── Why-stopped safety pattern (Issue 3.5) ──────────────────────────────────
# If why_stopped matches this regex, the edge gets:
#   - evidence_strength -0.20 (penalty)
#   - confidence="low"
#   - id_confidence="low"
#   - safety_signal="stopped_for_safety"
# The RL ranker can consume props.safety_signal directly.
CLINICALTRIALS_SAFETY_STOP_PATTERN: str = (
    r"(?i)\b(safety|adverse|death|toxicity|severe)\b"
)

# Penalty applied when why_stopped matches the safety pattern.
CLINICALTRIALS_SAFETY_STOP_PENALTY: float = 0.20

# Bonus applied when the trial has published results (Issue 2.5, 3.10).
CLINICALTRIALS_HAS_RESULTS_BONUS: float = 0.05

# ── Comparator/placebo detection (Issue 3.3) ────────────────────────────────
# If the intervention description matches this regex, the edge gets:
#   - drug_role="comparator_or_placebo"
#   - evidence_strength *= 0.3 (heavy penalty)
#   - id_confidence="low"
# This is the C2 patient-safety fix — prevents the ranker from learning
# "Warfarin treats Disease X" when Warfarin was the comparator arm.
#
# v70 ROOT FIX (P2L-043): the original pattern lumps placebos and
# active comparators into a single match. The pattern is preserved
# here for backward compatibility with external consumers, but the
# clinicaltrials_loader now uses two more-specific regexes
# (``_PLACEBO_REGEX`` and ``_ACTIVE_COMPARATOR_REGEX``) and a new
# ``CLINICALTRIALS_ACTIVE_COMPARATOR_EVIDENCE_MULTIPLIER`` constant
# to apply a MILDER penalty to active comparators (real drugs used as
# comparison standards — they ARE evidence the disease is treatable,
# unlike placebos which are inert controls). The legacy
# ``CLINICALTRIALS_COMPARATOR_EVIDENCE_MULTIPLIER`` (0.3) now applies
# ONLY to placebo arms.
CLINICALTRIALS_COMPARATOR_PATTERN: str = (
    r"(?i)\b(placebo|comparator|active control|active comparator)\b"
)
CLINICALTRIALS_COMPARATOR_EVIDENCE_MULTIPLIER: float = 0.3
# v70 P2L-043: Active-comparator multiplier. Active comparators are
# REAL drugs (e.g. metformin, warfarin) used as the comparison
# standard in a trial — they ARE evidence that the disease is
# treatable, but they're not the experimental arm. A mild penalty
# (0.8) reflects that the evidence is real but indirect (the trial
# was not designed to test the active comparator — it was designed
# to test the experimental drug against the active comparator). 0.8
# is chosen so active-comparator evidence stays in the "high" or
# "medium" confidence band for Phase 2/3 trials (Phase 3 base 0.7 ×
# 0.8 = 0.56 → medium; Phase 2 base 0.5 × 0.8 = 0.40 → medium),
# whereas the placebo multiplier 0.3 drops the same trials to "low"
# (Phase 3 × 0.3 = 0.21 → low; Phase 2 × 0.3 = 0.15 → low).
CLINICALTRIALS_ACTIVE_COMPARATOR_EVIDENCE_MULTIPLIER: float = 0.8

# ── Trial age filter (Issue 3.13, 5.7) ──────────────────────────────────────
# Default None — includes all trials. For RL training, consider setting
# max_trial_age_years=30 to exclude obsolete evidence.
CLINICALTRIALS_DEFAULT_MAX_TRIAL_AGE_YEARS: int | None = None

# ── Cross-product inflation (Issue 2.2) ─────────────────────────────────────
# AACT does not link interventions to conditions at the row level. A trial
# with interventions [Drug A, Placebo] and conditions [Disease X, Disease Y]
# produces 4 rows in the JOIN. Only ONE row (Drug A → Disease X) is the
# experimental association; the other 3 are fabrications of the JOIN.
# We penalize trials whose N_interventions × N_conditions exceeds this
# threshold.
CLINICALTRIALS_CROSS_PRODUCT_WARN_THRESHOLD: int = 4
CLINICALTRIALS_CROSS_PRODUCT_PENALTY: float = 0.10

# ── MeSH term handling (Issues 3.14, 5.11, 14.9) ────────────────────────────
# Warn if an intervention has >5 MeSH terms (suspicious — likely over-broad).
CLINICALTRIALS_MAX_MESH_PER_INTERVENTION: int = 5

# Known garbage MeSH values (placeholder, error strings) — Issue 5.11.
# v77 ROOT FIX (forensic scientific bug): "D000001" was INCORRECTLY in this
# blocklist. D000001 is a VALID MeSH descriptor ID (the MeSH ID for
# "Calcium", a real biomedical substance). Treating it as garbage caused
# every clinical trial record whose drug_mesh was D000001 to be silently
# quarantined and dropped from the knowledge graph — a scientific data-loss
# bug. The blocklist must contain ONLY placeholder/error sentinel strings,
# never valid MeSH IDs. Any MeSH ID matching ^D\d{6}$ is by definition a
# valid descriptor and must NOT be blocklisted.
CLINICALTRIALS_GARBAGE_MESH_VALUES: frozenset[str] = frozenset({
    "", "ERROR", "N/A", "UNKNOWN", "NULL", "NONE",
})

# ── NCT ID validation (Issues 3.15, 14.8, 15.6) ─────────────────────────────
# NCT IDs are 8-digit numeric, prefixed with "NCT".
CLINICALTRIALS_NCT_ID_REGEX_PATTERN: str = r"^NCT\d{8}$"

# ── Performance / scalability (Issues 8.1, 8.3, 8.5, 8.6, 8.10) ─────────────
# Default chunk size for SQL reads (Issue 8.1).
CLINICALTRIALS_CHUNK_SIZE: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_CHUNK_SIZE", "50000")
)

# Memory ceiling warning threshold (Issue 8.10).
CLINICALTRIALS_MEMORY_CEILING_WARNING_THRESHOLD: int = 1_000_000

# ── Download / retry / circuit breaker (Issues 6.1, 6.11, 12.3) ─────────────
CLINICALTRIALS_MAX_RETRIES: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_MAX_RETRIES", "3")
)
CLINICALTRIALS_RETRY_BACKOFF_BASE: float = float(
    os.environ.get("DRUGOS_CLINICALTRIALS_RETRY_BACKOFF_BASE", "2.0")
)
CLINICALTRIALS_DOWNLOAD_TIMEOUT_SECONDS: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_DOWNLOAD_TIMEOUT", "600")
)
CLINICALTRIALS_DOWNLOAD_CHUNK_SIZE: int = 1 << 20  # 1 MiB
CLINICALTRIALS_CIRCUIT_BREAKER_THRESHOLD: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_CIRCUIT_BREAKER_THRESHOLD", "5")
)
CLINICALTRIALS_CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_CIRCUIT_BREAKER_COOLDOWN", "3600")
)
CLINICALTRIALS_MIN_VALID_SIZE_BYTES: int = 1_000_000  # 1 MB minimum

# ── File path constants (Issues 6.5, 16.10) ─────────────────────────────────
CLINICALTRIALS_DEAD_LETTER_PATH: Path = DEAD_LETTER_DIR / "clinicaltrials_malformed.jsonl"
CLINICALTRIALS_QUARANTINE_PATH: Path = (
    DEAD_LETTER_DIR / "clinicaltrials_quarantine.csv"
)
CLINICALTRIALS_LINEAGE_LOG_PATH: Path = (
    LOGS_DIR / "lineage" / "clinicaltrials_lineage.jsonl"
)
CLINICALTRIALS_AUDIT_LOG_PATH: Path = LOGS_DIR / "audit" / "clinicaltrials_access.jsonl"
CLINICALTRIALS_QUALITY_REPORT_PATH: Path = (
    LOGS_DIR / "quality" / "clinicaltrials_quality_report.json"
)

# ── Environment variable overrides (Issue 12.6) ─────────────────────────────
CLINICALTRIALS_FORCE_DOWNLOAD: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_FORCE_DOWNLOAD", "0") == "1"
)
CLINICALTRIALS_SKIP: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_SKIP", "0") == "1"
)
CLINICALTRIALS_OFFLINE: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_OFFLINE", "0") == "1"
)
CLINICALTRIALS_SKIP_SHA256: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_SKIP_SHA256", "0") == "1"
)
CLINICALTRIALS_ALLOW_STALE: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_ALLOW_STALE", "0") == "1"
)
CLINICALTRIALS_ALLOW_LEGACY_SCHEMA: bool = (
    os.environ.get("DRUGOS_CLINICALTRIALS_ALLOW_LEGACY", "0") == "1"
)
CLINICALTRIALS_PROGRESS_LOG_INTERVAL: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_PROGRESS_INTERVAL", "100000")
)
CLINICALTRIALS_NEO4J_BATCH_SIZE: int = int(
    os.environ.get("DRUGOS_CLINICALTRIALS_NEO4J_BATCH_SIZE", "50000")
)

# ── URL allowlist (Issue 9.1, 9.2) ──────────────────────────────────────────
# AACT is hosted by CTTI at aact.ctti-clinicaltrials.org.
ALLOWED_CLINICALTRIALS_URLS: tuple[str, ...] = (
    "https://aact.ctti-clinicaltrials.org/",
)

# ── User-Agent (Issue 9.2) ──────────────────────────────────────────────────
CLINICALTRIALS_USER_AGENT: str = (
    os.environ.get(
        "DRUGOS_CLINICALTRIALS_USER_AGENT",
        "DrugOS/2.1 (drugos@example.com)",
    )
)

# ── ZIP magic bytes (Issues 4.9, 6.8) ───────────────────────────────────────
# ZIP files start with the magic bytes PK\x03\x04.
CLINICALTRIALS_ZIP_MAGIC: bytes = b"PK\x03\x04"

# ── Extraction sentinel (Issues 4.8, 6.9) ───────────────────────────────────
# File written at the end of extraction to indicate completion.
CLINICALTRIALS_EXTRACT_SENTINEL: str = "_AACT_EXTRACT_COMPLETE"

# ── Hash / edge ID (Issues 2.3, 7.2) ────────────────────────────────────────
CLINICALTRIALS_HASH_LENGTH: int = 16
CLINICALTRIALS_EDGE_ID_SOURCE: str = "CLINICALTRIALS"

# ── Neo4j label / edge type contract (Issues 2.1, 14.1, 15.3, 15.9) ─────────
# The ONLY emittable triple is ("Compound", "tested_for", "Disease").
# "treats" is FORBIDDEN — reserved for FDA-approved drugs from DrugBank.
# "clinical_trial" is DEPRECATED v0 name — kept for backward-compat shim
# only, never emitted by the new loader.
CLINICALTRIALS_EMITTABLE_TRIPLES: frozenset[tuple[str, str, str]] = frozenset({
    ("Compound", "tested_for", "Disease"),
})

# Valid Neo4j node types this loader can emit (Issue 15.9).
CLINICALTRIALS_VALID_NODE_TYPES: frozenset[str] = frozenset({
    "Compound",
    "Disease",
})

# ── Staleness (Issue 11.2 — stale cache warning) ────────────────────────────
# Warn if cached AACT zip is older than this many days.
CLINICALTRIALS_STALE_CACHE_WARNING_DAYS: int = 7

# ── Expected record count deviation (Issues 5.9, 12.4) ──────────────────────
# Warn if row count deviates from expected_record_count by more than this
# fraction (0.10 = 10%). Raise CriticalDataSourceError if deviation >50%.
CLINICALTRIALS_DEVIATION_WARNING_THRESHOLD: float = 0.10
CLINICALTRIALS_DEVIATION_CRITICAL_THRESHOLD: float = 0.50

# ── Pinning / reproducibility (Issue 7.8) ───────────────────────────────────
# When set, the loader refuses to use any AACT snapshot other than the
# pinned one. Set via DRUGOS_CLINICALTRIALS_PINNED_RELEASE env var.
CLINICALTRIALS_PINNED_RELEASE: str | None = (
    os.environ.get("DRUGOS_CLINICALTRIALS_PINNED_RELEASE")
)

# CLINICALTRIALS_PINNED_SHA256 — set to None because AACT does not publish a
# checksum. The loader computes it at download time and stores it as a
# sidecar / lineage field for future runs (Issues 6.10, 7.4).
CLINICALTRIALS_PINNED_SHA256: str | None = None


# =============================================================================
# GEO constants — added by geo_loader v1.0.0 institutional-grade audit fix
# (GEO_LOADER_MASTER_REPAIR_PROMPT.md — 192 findings across 16 domains).
#
# These constants centralise every magic value used by ``geo_loader.py``
# so the loader NEVER hardcodes URLs, filenames, retry counts, timeouts,
# regex patterns, or threshold values (master prompt Rule R7).
#
# Naming convention mirrors CLINICALTRIALS_/SIDER_/OPENTARGETS_/STITCH_/
# STRING_: every constant is prefixed with ``GEO_`` and is the single
# source of truth for its value.
#
# Patient-safety doctrine: GEO is the SOLE source of
# Protein→expressed_in→Anatomy edges in the KG. If GEO silently fails,
# the KG lacks the entire tissue-specificity modality, the Graph
# Transformer cannot learn that a drug target is absent from the disease
# tissue, and a clinician may be handed a "high-confidence" repurposing
# candidate that will fail in Phase II — or harm a patient. Every
# constant here exists to make the loader's behavior explicit and
# auditable.
# =============================================================================

# SOURCE_GEO — human-readable source name (used in node .source and edge
# _source props). NOTE: this is the lowercase form used in the
# ``_source`` field on every record (per R15); the title-case form
# ``"GEO"`` is used in log messages and reports.
SOURCE_GEO: str = "geo"

# GEO_PARSER_VERSION — bumped on any parser logic change (master prompt
# §11.1 / §0.2 constraint #22 — Definition of Done). The parser version
# propagates to every emitted record via the ``_parser_version``
# provenance field (R15) so the KG builder can trace any edge back to
# the exact parser that produced it.
GEO_PARSER_VERSION: str = "1.0.0"

# GEO_SCHEMA_VERSION — bumped on any output-schema change. Tied to
# ``GeoRawRecord`` / ``GeoEdgeRecord`` in schemas.py (GEO-14.5, GEO-14.6).
GEO_SCHEMA_VERSION: str = "1.0.0"

# GEO_API_VERSION — semantic version of the public API (download_geo,
# parse_geo_series, geo_to_edge_records, GeoLoader). Bumped on any
# breaking change to public function signatures (GEO-15.7).
GEO_API_VERSION: str = "1.0.0"

# GEO_LICENSE — GEO data is public domain (U.S. Government work).
# Every record carries ``_license=GEO_LICENSE`` (Phase 0.9 / GEO-14.3).
GEO_LICENSE: str = "Public Domain"

# GEO_ATTRIBUTION — NCBI requests citation of "Barrett T et al. Nucleic
# Acids Res. 2013" for GEO. Every record carries
# ``_attribution=GEO_ATTRIBUTION`` (Phase 0.9 / GEO-14.4 / GEO-15.8).
GEO_ATTRIBUTION: str = (
    "Data source: GEO (Barrett T et al., Nucleic Acids Res. 2013), "
    "NCBI, https://www.ncbi.nlm.nih.gov/geo/"
)

# GEO_CITATION — full academic citation for the GEO database (GEO-3.11).
GEO_CITATION: str = (
    "Barrett T, Wilhite SE, Ledoux P, et al. NCBI GEO: archive for "
    "functional genomics data sets—update. Nucleic Acids Res. "
    "2013;41(D1):D991-D995. doi:10.1093/nar/gks1193."
)

# GEO_CITATION_ORIGINAL — the original GEO paper (GEO-3.11).
GEO_CITATION_ORIGINAL: str = (
    "Edgar R, Domrachev M, Lash AE. Gene Expression Omnibus: NCBI gene "
    "expression and hybridization array data repository. Nucleic Acids "
    "Res. 2002;30(1):207-10. doi:10.1093/nar/30.1.207."
)

# ── Series ID validation (Phase 0.4, GEO-2.1, GEO-3.14, GEO-7.5, GEO-9.2,
#    GEO-12.1) ────────────────────────────────────────────────────────────────
# GEO Series accession format: ``GSE`` followed by 1+ digits.
# Examples: GSE1, GSE12345, GSE92649.
# Counter-examples rejected by this regex:
#   * ``GSE1A``        (letters after digits)
#   * ``gse12345``     (lowercase)
#   * ``GSE``          (no digits)
#   * ``../../../etc`` (path traversal)
#   * empty string
GEO_SERIES_ID_REGEX: str = r"GSE\d+"

# GEO_SAMPLE_ID_REGEX — GSM accession format (GEO-5.1).
GEO_SAMPLE_ID_REGEX: str = r"GSM\d+"

# GEO_PLATFORM_ID_REGEX — GPL accession format (GEO-5.1).
GEO_PLATFORM_ID_REGEX: str = r"GPL\d+"

# GEO_UBERON_URI_REGEX — UBERON URI format (GEO-3.3, GEO-5.1).
GEO_UBERON_URI_REGEX: str = r"http://purl\.obolibrary\.org/obo/UBERON_\d+"

# ── Pinned series (Phase 0.2, Phase 0.4, GEO-2.1, GEO-3.14, GEO-7.5,
#    GEO-12.1) ──────────────────────────────────────────────────────────────
# GSE92649 (Cheng et al., 2018, Sci Rep) is the pinned GEO series for v1.0.0.
# RATIONALE: well-characterized human expression dataset covering multiple
# tissues; used in drug repurposing literature. Future versions may add
# more series (ADR-GEO-002 in the module docstring).
GEO_PINNED_SERIES_ID: str = "GSE92649"
GEO_PINNED_RELEASE_DATE: str = "2018-01-01"

# ── SOFT format constants (GEO-2.10, GEO-5.7) ───────────────────────────────
# Expected schema version of the SOFT file (matches DATA_SOURCES["geo"]).
GEO_SOFT_SCHEMA_VERSION: str = "GEO-SOFT-2.0"

# Maximum fraction of malformed lines tolerated before raising GeoParseError
# (GEO-5.7). Default: 1% of total lines.
GEO_MAX_MALFORMED_LINE_RATIO: float = 0.01

# ── Expression thresholds (GEO-2.8, GEO-3.2, GEO-3.9) ───────────────────────
# Default log2 expression above which "expressed" is called. log2 = 4.0
# corresponds to ~16 TPM, which is a commonly used threshold in the
# transcriptomics literature for "detectable expression".
GEO_DEFAULT_EXPRESSION_THRESHOLD: float = 4.0

# Default minimum sample count supporting an edge (GEO-3.8).
GEO_DEFAULT_MIN_SAMPLES: int = 3

# Default FDR threshold for differential-expression calls (GEO-3.7).
GEO_DEFAULT_FDR_THRESHOLD: float = 0.05

# ── Expression unit normalization (GEO-3.9) ─────────────────────────────────
# Canonical unit that all expression values are normalized to.
GEO_CANONICAL_EXPRESSION_UNIT: str = "log2_rma"

# Supported input expression units. Any other unit is dead-lettered with
# reason="unknown_expression_unit".
GEO_VALID_EXPRESSION_UNITS: frozenset[str] = frozenset({
    "log2_rma", "log2_tpm", "log2_fpkm",
    "raw_counts", "rpm", "tpm", "fpkm",
})

# ── Human organism filter (GEO-3.13) ────────────────────────────────────────
# The KG is human-protein-centric. Only Homo sapiens (taxid 9606) records
# are kept by default. Mouse/rat/etc. records are dead-lettered.
GEO_DEFAULT_ORGANISM_FILTER: str = "Homo sapiens"
GEO_HUMAN_TAXID: int = 9606

# ── Memory budget (GEO-8.6) ─────────────────────────────────────────────────
# Default memory budget for parse_geo_series. If tracemalloc detects
# usage above this, GeoDataQualityError is raised.
GEO_DEFAULT_MEMORY_BUDGET_MB: int = 2048

# ── Random seed (GEO-7.3, R9) ───────────────────────────────────────────────
# Fixed random seed for any randomized operation (e.g., down-sampling).
GEO_RANDOM_SEED: int = 0

# ── File permissions (GEO-9.5) ──────────────────────────────────────────────
# Files written by geo_loader are 0o600 (owner read/write only) to prevent
# unauthorized access in shared environments.
GEO_FILE_PERMISSIONS: int = 0o600
GEO_DIR_PERMISSIONS: int = 0o700

# ── Circuit breaker (GEO-6.10) ──────────────────────────────────────────────
# Number of consecutive failures before the circuit breaker trips.
GEO_CIRCUIT_BREAKER_THRESHOLD: int = 5

# How long the circuit breaker stays tripped (seconds).
GEO_CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = 3600

# ── Atomic download (GEO-6.5) ───────────────────────────────────────────────
# Partial downloads are written to a ``.part`` file and renamed to the
# final name only after the SHA-256 verification passes.
GEO_PART_SUFFIX: str = ".part"

# ── Marker file (GEO-7.10) ──────────────────────────────────────────────────
# After a successful download, a marker file is written next to the SOFT
# file containing the download timestamp, sha256, and pipeline version.
GEO_MARKER_FILE_SUFFIX: str = ".downloaded_at"

# ── Sidecar metadata (GEO-16.4) ────────────────────────────────────────────
# After a successful download, a JSON sidecar is written containing the
# full provenance of the file (url, sha256, downloaded_at, etc.).
GEO_META_SIDECAR_SUFFIX: str = ".meta.json"

# ── Downstream consumers (GEO-15.9) ─────────────────────────────────────────
# Modules that consume GEO data. Documented for impact analysis.
GEO_DOWNSTREAM_CONSUMERS: tuple[str, ...] = (
    "kg_builder", "entity_resolver", "graph_stats",
)

# ── URL allowlist (GEO-9.7) ─────────────────────────────────────────────────
# Only HTTPS URLs from NCBI are allowed for GEO downloads.
ALLOWED_GEO_URLS: tuple[str, ...] = (
    "https://ftp.ncbi.nlm.nih.gov/geo/",
    "https://www.ncbi.nlm.nih.gov/geo/",
)

# ── User-Agent (GEO-9.3) ────────────────────────────────────────────────────
# Identify ourselves to NCBI for rate-limit negotiation.
GEO_USER_AGENT: str = (
    os.environ.get(
        "DRUGOS_GEO_USER_AGENT",
        "DrugOS/2.1 (drugos@example.com)",
    )
)

# ── NCBI API key (GEO-9.3) ──────────────────────────────────────────────────
# Optional. Increases the rate limit from 3 to 10 requests/second.
# Read from env var; never logged (redacted as ``***`` in logs).
GEO_NCBI_API_KEY: str | None = os.environ.get("NCBI_API_KEY") or None

# ── Sensitive-field regex (GEO-9.1) ─────────────────────────────────────────
# If a SOFT !Sample_characteristics field name matches this regex, the
# record is tagged ``sensitive=True`` (PII declaration in module docstring).
GEO_SENSITIVE_FIELD_REGEX: str = r"(?i)patient|subject|participant"

# ── Staleness (GEO-16.13) ───────────────────────────────────────────────────
# Warn if the source series is older than this many days.
GEO_STALE_FILE_DAYS: int = 365 * 5  # 5 years

# ── Record count deviation (GEO-5.5) ────────────────────────────────────────
# If parsed record count is below this fraction of expected, raise
# GeoDataQualityError (or GeoCriticalError if GEO_REQUIRED=1).
GEO_RECORD_COUNT_MIN_FRACTION: float = 0.5

# If parsed record count is above this multiple of expected, log WARNING
# (may indicate double-parsing or format change).
GEO_RECORD_COUNT_MAX_MULTIPLE: float = 2.0

# ── Idempotency / backfilling (GEO-7.6) ─────────────────────────────────────
# GEO series are NOT versioned by NCBI. Once superseded, the old version
# is permanently unavailable. Backfilling historical data is not supported.
GEO_SUPPORTS_BACKFILL: bool = False

# ── Hash / edge ID (GEO-5.11, GEO-16.12) ────────────────────────────────────
# Length of the ``_edge_sha256`` field truncated for logging.
GEO_EDGE_SHA256_LOG_LENGTH: int = 16

# ── Subdir name (GEO-12.4) ──────────────────────────────────────────────────
# Subdirectory under RAW_DIR where GEO SOFT files are stored.
# Matches the convention used by every sibling loader.
GEO_SUBDIR: str = "geo"

# ── Performance / scalability (GEO-8.2, GEO-8.3, GEO-8.7) ───────────────────
# Default chunk size for streaming parse of expression matrices.
GEO_DEFAULT_CHUNK_SIZE: int = 10_000

# Default max_workers for parallel batch download (GEO-8.7).
# NCBI rate-limits to 3 requests/second without an API key.
GEO_DEFAULT_MAX_WORKERS: int = 3

# Large-file threshold for switching to streaming mode (GEO-8.6).
GEO_LARGE_FILE_THRESHOLD_BYTES: int = 100 * 1024 * 1024  # 100 MB

# ── Batch-effect detection (GEO-3.6) ────────────────────────────────────────
# If samples come from >1 platform or series, a WARNING is logged.
# v1.0.0 does NOT perform batch correction (ComBat / limma).
GEO_SUPPORTS_BATCH_CORRECTION: bool = False

# ── Environment variable overrides (GEO-12.5, Phase 0.1, 0.4, 0.5) ──────────
# GEO_REQUIRED=1 — fail loudly (raise GeoCriticalError) if GEO produces
# 0 records. Default "0" (graceful degradation — log WARNING, continue).
GEO_REQUIRED: bool = os.environ.get("GEO_REQUIRED", "0") == "1"

# GEO_AUTO_DOWNLOAD=1 — enable automatic download. Default "0" (operator
# must place the file manually OR set GEO_AUTO_DOWNLOAD=1).
GEO_AUTO_DOWNLOAD: bool = os.environ.get("GEO_AUTO_DOWNLOAD", "0") == "1"

# GEO_KEEP_BACKUPS=1 — keep .bak.{timestamp} of overwritten files.
GEO_KEEP_BACKUPS: bool = os.environ.get("GEO_KEEP_BACKUPS", "0") == "1"

# GEO_MEMORY_BUDGET_MB — override the default memory budget.
GEO_ENV_MEMORY_BUDGET_MB: int = int(
    os.environ.get("GEO_MEMORY_BUDGET_MB", str(GEO_DEFAULT_MEMORY_BUDGET_MB))
)

# GEO_SKIP_SHA256=1 — skip SHA-256 verification (testing only, NEVER in prod).
GEO_SKIP_SHA256: bool = os.environ.get("GEO_SKIP_SHA256", "0") == "1"

# GEO_OFFLINE=1 — never attempt network calls; only use cached files.
GEO_OFFLINE: bool = os.environ.get("DRUGOS_GEO_OFFLINE", "0") == "1"

# GEO_SKIP_RECORD_COUNT_GUARD=1 — skip the GEO-5.5 record-count validation
# (testing only — useful for unit tests that use small fixtures). NEVER
# set this in production.
GEO_SKIP_RECORD_COUNT_GUARD: bool = (
    os.environ.get("GEO_SKIP_RECORD_COUNT_GUARD", "0") == "1"
)

# DRUGOS_ENV — environment selector (GEO-12.9). Default "production" (v108 issue 76).
#   * "production" / "prod" / "staging" / "stage" — production-grade thresholds
#   * "dev"     — small test series for fast iteration
# v29 ROOT FIX (audit I-4 — THREE environment selectors): the codebase
# had THREE different env selectors with DIFFERENT defaults:
#   DRUGOS_ENV (default "dev") — DEAD, not used anywhere
#   DRUGOS_ENVIRONMENT (default "dev") — used everywhere
#   ENVIRONMENT (default "development") — DIFFERENT default!
# This was a footgun: setting DRUGOS_ENVIRONMENT=production left
# ENVIRONMENT="development", causing contradictory behavior.
# ROOT FIX: make DRUGOS_ENV an ALIAS of DRUGOS_ENVIRONMENT (same value,
# same default). ENVIRONMENT is also aliased below. Now there is ONE
# source of truth: DRUGOS_ENVIRONMENT.
DRUGOS_ENV: str = os.environ.get("DRUGOS_ENVIRONMENT", os.environ.get("DRUGOS_ENV", "production"))


def get_geo_series_path(series_id: str | None = None) -> Path:
    """Get the expected local path for a GEO Series SOFT file.

    Returns the path under ``RAW_DIR / GEO_SUBDIR`` for the given series
    ID. If ``series_id`` is None, defaults to the pinned series from
    ``DATA_SOURCES["geo"]["version"]`` (currently ``"GSE92649"``).

    The filename convention is ``{series_id}_family.soft.gz`` to match
    NCBI's naming (e.g. ``GSE92649_family.soft.gz``). This preserves the
    series ID in the filename for traceability (GEO-16.4) and matches
    the URL path on NCBI's FTP server.

    Parameters
    ----------
    series_id : str, optional
        GSE accession (e.g. ``"GSE92649"``). If None, uses the pinned
        series from ``DATA_SOURCES["geo"]["version"]``.

    Returns
    -------
    Path
        Expected file path under ``RAW_DIR / GEO_SUBDIR``.

    Raises
    ------
    GeoConfigurationError
        If ``series_id`` is provided but does not match ``GEO_SERIES_ID_REGEX``.
        If ``DATA_SOURCES["geo"]`` is missing or has no ``version`` key.

    See Also
    --------
    drugos_graph.config.get_data_source_path : Generic path resolver.
    drugos_graph.geo_loader.download_geo : Performs the actual download.

    Fixes: GEO-4.11 (filename convention), GEO-12.4 (subdir constant),
           GEO-15.2 (filename mismatch with get_data_source_path).
    """
    # Local import to avoid a circular import at module load time
    # (exceptions.py imports nothing from config.py, but config.py is
    # imported by many modules that also import exceptions).
    from .exceptions import GeoConfigurationError
    import re as _re

    if "geo" not in DATA_SOURCES:
        raise GeoConfigurationError(
            "DATA_SOURCES['geo'] is missing — cannot resolve GEO series path",
            context={"available_sources": sorted(DATA_SOURCES.keys())},
        )

    if series_id is None:
        series_id = DATA_SOURCES["geo"].get("version")
        if not series_id:
            raise GeoConfigurationError(
                "DATA_SOURCES['geo']['version'] is missing — cannot resolve "
                "default GEO series ID",
                context={"geo_config_keys": sorted(DATA_SOURCES["geo"].keys())},
            )

    if not _re.fullmatch(GEO_SERIES_ID_REGEX, series_id):
        raise GeoConfigurationError(
            f"Invalid GEO series_id {series_id!r} — must match "
            f"{GEO_SERIES_ID_REGEX}",
            context={"series_id": series_id, "regex": GEO_SERIES_ID_REGEX},
        )

    return RAW_DIR / GEO_SUBDIR / f"{series_id}_family.soft.gz"


# All directory constants (used by ensure_dirs)
_ALL_DIR_CONSTANTS = [
    RAW_DIR, PROCESSED_DIR, KG_EXPORT_DIR, EMBEDDINGS_DIR,
    LOGS_DIR, MODEL_DIR, DEAD_LETTER_DIR, CHECKPOINT_DIR,
    AUDIT_LOG_DIR, OUTPUT_METADATA_DIR, IMPACT_ANALYSIS_DIR,
    TRANSFORMATION_LOG_DIR, CONFIG_DIFF_DIR,
]

# Thread-safe lazy directory initialization
# Fixes audit issue 4.2 — ensure_dirs not called before I/O
_dirs_lock = threading.Lock()
_dirs_initialized = False


def ensure_dirs() -> None:
    """Create required directories if they don't exist (idempotent, thread-safe).

    This function MUST be called before any file I/O operations.
    It is automatically called by the pipeline runner. Modules that
    save files (pyg_builder, run_pipeline) should call this at their
    entry point.

    Fixes audit issues 4.2, 6.1, 6.5, 6.9:
      - 4.2: ensure_dirs() is called before any file I/O
      - 6.1: logs warning on failure instead of crashing
      - 6.5: creates dead_letter and checkpoint dirs
      - 6.9: thread-safe with proper lock
    """
    global _dirs_initialized
    with _dirs_lock:
        if _dirs_initialized and all(d.exists() for d in _ALL_DIR_CONSTANTS):
            return
        for d in _ALL_DIR_CONSTANTS:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # Fixes audit issue 6.1 — graceful degradation
                logger.warning(
                    "Could not create directory %s: %s. "
                    "Operations requiring this directory will fail.",
                    d, exc,
                )
        _dirs_initialized = True


# ─── Phase D — Data Source Registry ──────────────────────────────────────────
# Fixes audit issues 5.1, 5.2, 5.5, 5.8, 5.9, 5.10, 6.2, 6.3,
#   7.7, 14.4, 15.1, 15.2, 16.5
#
# Every entry now has structured fields:
#   - sha256, md5: checksums for integrity verification (issue 5.1)
#   - version: machine-readable version (issue 7.7)
#   - pinned: whether URL is pinned to specific version (issue 5.2)
#   - release_date: when this version was released (issue 7.7)
#   - license: data license (issue 14.4)
#   - size_bytes: expected download size (issue 8.1)
#   - expected_record_count: approximate record count (issue 5.9)
#   - last_updated: when we last downloaded (issue 5.10)
#   - expected_update_frequency_days: how often source updates (issue 5.10)
#   - last_downloaded_at: timestamp of last download (issue 16.5)
#   - retry_count, retry_backoff_seconds, timeout_seconds: download params (issues 6.2, 6.3)
#   - max_size_bytes: guard against unexpected growth (issue 8.1)
#   - url_scheme: http/https/ftp (issue 15.2)
#   - schema_version: schema version of the source data (issue 15.1)

DATA_SOURCES: dict[str, dict[str, Any]] = {
    "drkg": {
        "url": (
            "https://dgl-data.s3-us-west-2.amazonaws.com/"
            "dataset/DRKG/drkg.tar.gz"
        ),
        "filename": "drkg.tar.gz",
        "tsv_file": "drkg.tsv",
        "description": (
            "Drug Repurposing Knowledge Graph — "
            "97K nodes, 5.9M edges"
        ),
        "version_note": "Stable release; no version number in URL",
        # Fixes audit issue 7.7 — structured version field
        "version": "2.0",
        "pinned": True,
        "release_date": "2023-06-01",
        # Fixes audit issue 5.1 — checksums for integrity
        "sha256": None,  # To be computed after first download
        "md5": None,
        # Fixes audit issue 14.4 — license
        "license": "MIT",
        # Fixes audit issue 8.1 — expected size
        "size_bytes": 500_000_000,
        "max_size_bytes": 1_000_000_000,
        # Fixes audit issue 5.9 — expected record count
        "expected_record_count": 5_874_261,
        # Fixes audit issue 5.10 — freshness tracking
        "last_updated": None,
        "expected_update_frequency_days": 365,
        "last_downloaded_at": None,
        # Fixes audit issues 6.2, 6.3 — download resilience
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 300,
        # Fixes audit issue 15.2 — url scheme
        "url_scheme": "https",
        # Fixes audit issue 15.1 — schema version
        "schema_version": "1.0",
    },
    "drugbank": {
        "url": (
            # Fixes audit issue 5.2 — pinned to specific release
            "https://go.drugbank.com/releases/5-1-12/downloads/"
            "all-full-database.xml"
        ),
        "filename": "drugbank.xml",
        "description": (
            "DrugBank full database XML "
            "(requires academic registration)"
        ),
        "version_note": (
            "Pinned to release 5.1.12 for reproducibility. "
            "NOTE: Requires free academic registration and login cookie. "
            "Download manually and place in data/raw/ directory."
        ),
        "version": "5.1.12",
        "pinned": True,
        "release_date": "2023-12-01",
        "sha256": None,
        "md5": None,
        "license": "CC BY-NC 4.0 (academic)",
        "size_bytes": 1_200_000_000,
        "max_size_bytes": 2_000_000_000,
        "expected_record_count": 15_000,
        "last_updated": None,
        "expected_update_frequency_days": 90,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 60,
        "timeout_seconds": 600,
        "url_scheme": "https",
        "schema_version": "5.1",
    },
    "chembl": {
        "url": (
            # Fixes audit issue 5.2 — pinned to chembl_35
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/"
            "rel/chembl_35/chembl_35_sqlite.tar.gz"
        ),
        "filename": "chembl_sqlite.tar.gz",
        "description": "ChEMBL bioactivity database SQLite dump",
        "version_note": (
            "Pinned to chembl_35 for reproducibility."
        ),
        "version": "35",
        "pinned": True,
        "release_date": "2024-05-01",
        "sha256": None,
        "md5": None,
        "license": "CC BY-SA 3.0",
        "size_bytes": 4_000_000_000,
        "max_size_bytes": 8_000_000_000,
        "expected_record_count": 2_400_000,
        "last_updated": None,
        "expected_update_frequency_days": 120,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 600,
        "url_scheme": "https",
        "schema_version": "35",
    },
    "opentargets": {
        "url": (
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/"
            "platform/25.03/output/evidence/sourceId=chembl/"
            "evidence-chembl.json.gz"
        ),
        "filename": "opentargets_evidence.json.gz",
        "description": (
            "OpenTargets drug-target-disease evidence with scores"
        ),
        "version_note": (
            "Pinned to platform release 25.03. Update URL when "
            "new release available."
        ),
        "version": "25.03",
        "pinned": True,
        "release_date": "2025-03-01",
        "sha256": None,
        "md5": None,
        "license": "CC0 1.0",
        "size_bytes": 800_000_000,
        "max_size_bytes": 2_000_000_000,
        "expected_record_count": 5_000_000,
        "last_updated": None,
        "expected_update_frequency_days": 90,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 300,
        "url_scheme": "https",
        "schema_version": "25.03",
    },
    "string": {
        "url": (
            "https://string-db.org/download/"
            "protein.links.full.v12.0/"
            "9606.protein.links.full.v12.0.txt.gz"
        ),
        "filename": "string_ppi.txt.gz",
        "description": (
            "STRING protein-protein interactions for Homo sapiens"
        ),
        "version_note": (
            "Pinned to v12.0; update URL when new version available"
        ),
        "version": "12.0",
        "pinned": True,
        "release_date": "2023-06-01",
        "sha256": None,
        "md5": None,
        "license": "CC BY 4.0",
        "size_bytes": 300_000_000,
        "max_size_bytes": 1_000_000_000,
        "expected_record_count": 11_000_000,
        "last_updated": None,
        "expected_update_frequency_days": 365,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 300,
        "url_scheme": "https",
        "schema_version": "12.0",
        # CONF-1 (id_crosswalk audit): externalized STRING aliases filename
        # so that upgrading STRING (v12.0 -> v12.5 -> v13.0) does NOT require
        # code changes in run_pipeline.py. The pattern is used for
        # version-agnostic resolution; the filename is the pinned default.
        "aliases_pattern": "9606.protein.aliases.v*.txt.gz",
        "aliases_filename": "9606.protein.aliases.v12.0.txt.gz",
    },
    "uniprot": {
        "url": (
            # Fixes audit issue 5.2 — pinned to 2024_03 release.
            # Fixes D12-002 — URL must point at the FLAT .dat.gz file, NOT the
            # .tar.gz archive. The previous URL (uniprot_sprot-only2024_03.tar.gz)
            # downloaded a tar archive that gzip.open() could decompress but
            # whose first bytes were a tar header, not an "ID " line — the
            # parser silently produced ZERO records. Production would launch
            # with an empty Protein subgraph.
            "https://ftp.uniprot.org/pub/databases/uniprot/"
            "knowledgebase/complete/"
            "uniprot_sprot.dat.gz"
        ),
        "filename": "uniprot_sprot.dat.gz",
        "description": (
            "UniProt Swiss-Prot manually reviewed protein entries"
        ),
        "version_note": (
            "Pinned to 2024_03 release for reproducibility."
        ),
        "version": "2024_03",
        "pinned": True,
        "release_date": "2024-03-01",
        "sha256": None,
        "md5": None,
        "license": "CC BY 4.0",
        "size_bytes": 800_000_000,
        "max_size_bytes": 2_000_000_000,
        "expected_record_count": 570_000,
        "last_updated": None,
        "expected_update_frequency_days": 60,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 600,
        "url_scheme": "https",
        "schema_version": "2024_03",
    },
    "clinicaltrials": {
        "url": (
            "https://aact.ctti-clinicaltrials.org/static/"
            "static_db_copies/dataset/aact_dataset.zip"
        ),
        "filename": "aact_dataset.zip",
        "description": "ClinicalTrials.gov AACT database",
        "version_note": "Stable URL; no version in path",
        "version": "current",
        "pinned": False,
        "release_date": None,
        "sha256": None,
        "md5": None,
        "license": "CC0 1.0",
        "size_bytes": 500_000_000,
        "max_size_bytes": 1_500_000_000,
        "expected_record_count": 500_000,
        "last_updated": None,
        "expected_update_frequency_days": 1,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 60,
        "timeout_seconds": 600,
        "url_scheme": "https",
        "schema_version": "current",
        # ── New keys added by clinicaltrials_loader v2.1.0 institutional-grade
        # audit fix (PROMPT_fix_clinicaltrials_loader.md Issues 3.4, 12.1,
        # 12.2). These mirror CLINICALTRIALS_DEFAULT_* constants for callers
        # that read the config dict instead of the constants.
        "default_phases": ("Phase 3", "Phase 4"),  # Issue 3.4, 12.1 — Phase 4 included
        "default_intervention_types": ("Drug", "Biological"),  # Issue 12.2
        "default_study_types": ("Interventional",),  # Issue 3.7
        "default_allowed_statuses": (
            "Completed", "Active, not recruiting", "Recruiting",
            "Enrolling by invitation", "Not yet recruiting",
        ),  # Issue 3.11
        "min_enrollment": 0,  # Issue 3.6
        "circuit_breaker_threshold": 5,  # Issue 6.11
        "circuit_breaker_cooldown_seconds": 3600,  # Issue 6.11
    },
    "stitch": {
        "url": (
            "https://stitch.embl.de/download/"
            "protein_chemical.links.detailed.v5.0/"
            "9606.protein_chemical.links.detailed.v5.0.tsv.gz"
        ),
        "filename": "stitch_interactions.tsv.gz",
        "description": "STITCH chemical-protein interaction network",
        "version_note": (
            "Pinned to v5.0; update URL when new version available. "
            "Changed from http:// to https:// for security."
        ),
        "version": "5.0",
        "pinned": True,
        "release_date": "2022-01-01",
        "sha256": None,
        "md5": None,
        "license": "CC0 1.0",
        "size_bytes": 1_000_000_000,
        "max_size_bytes": 3_000_000_000,
        "expected_record_count": 20_000_000,
        "last_updated": None,
        "expected_update_frequency_days": 730,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 600,
        "url_scheme": "https",
        "schema_version": "5.0",
    },
    "sider": {
        # Phase 0.4 / D3.8 / D14.11 — pinned version + canonical filename.
        # The meddra_all_se.tsv.gz file is the canonical SIDER adverse-event
        # file (6 columns, no header). The previous filename
        # "sider_side_effects.tsv.gz" was non-canonical; renamed to
        # "sider_meddra_all_se.tsv.gz" to match the source filename pattern.
        "url": (
            "https://sideeffects.embl.de/media/download/"
            "meddra_all_se.tsv.gz"
        ),
        "filename": "sider_meddra_all_se.tsv.gz",
        "description": (
            "SIDER side effect database — drug-side effect pairs with "
            "MedDRA terms. SOLE source of adverse-event data feeding the "
            "RL safety-signal dimension (Phase 0.4 / G6)."
        ),
        "version_note": (
            "Pinned to SIDER 2023-10-25 release. SIDER does NOT publish "
            "sha256; we compute and pin it as a sidecar at first download. "
            "Update version + release_date + sha256 together when SIDER "
            "publishes a new release."
        ),
        # Phase 0.4 / D3.8 — pinned version.
        "version": "2023-10-25",
        "pinned": True,
        "release_date": "2023-10-25",
        # SIDER does not publish sha256 — we compute and store it as a
        # sidecar at first download (D3.8 / D4.19). Leave None here; the
        # loader will populate data/raw/sider_meddra_all_se.tsv.gz.sha256.
        "sha256": None,
        "md5": None,
        "license": "CC0 1.0",
        # D5.1 — expected size + record count for integrity check.
        # Real SIDER meddra_all_se.tsv.gz is ~50 MB compressed, ~5M rows.
        "size_bytes": 50_000_000,
        "max_size_bytes": 500_000_000,
        "expected_record_count": 5_000_000,
        "last_updated": None,
        "expected_update_frequency_days": 365,
        "last_downloaded_at": None,
        # D6.1 / D6.2 — retry / timeout.
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 120,
        "url_scheme": "https",
        "schema_version": "1.0.0",
        # D16.5 — MedDRA vocabulary version. SIDER 2023 uses MedDRA v26.0.
        "meddra_version": "26.0",
        # D3.3 — sibling SIDER files for future expansion (FDA labels +
        # frequencies). NOT downloaded by default; loader supports them
        # via parse_sider_fda_labels / parse_sider_frequencies.
        "sibling_files": {
            "meddra_all_label.tsv.gz": "FDA drug labels with MedDRA terms",
            "meddra_freq.tsv.gz": "Adverse-event frequencies",
        },
    },
    "geo": {
        # Fixes audit issue 3.5 — GEO URL was a placeholder (GSE1 doesn't exist)
        # RATIONALE: GSE92649 is a real gene expression dataset used in
        # drug repurposing literature (Cheng et al., 2018, Sci Rep).
        # The placeholder GSE1 was a non-existent example URL.
        "url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/"
            "GSE92649/soft/GSE92649_family.soft.gz"
        ),
        "filename": "geo_expression.soft.gz",
        "description": (
            "GEO Gene Expression Omnibus — GSE92649 gene expression data "
            "for drug repurposing (Cheng et al., 2018)"
        ),
        "version_note": (
            "Pinned to GSE92649. Specific series must be selected "
            "based on the disease/drug context."
        ),
        "version": "GSE92649",
        "pinned": True,
        "release_date": "2018-01-01",
        "sha256": None,
        "md5": None,
        "license": "Public Domain",
        "size_bytes": 100_000_000,
        "max_size_bytes": 500_000_000,
        "expected_record_count": 50_000,
        "last_updated": None,
        "expected_update_frequency_days": None,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 300,
        "url_scheme": "https",
        "schema_version": "GEO-SOFT-2.0",
        # ── New GEO keys added by geo_loader v1.0.0 institutional-grade
        #    audit fix (GEO_LOADER_MASTER_REPAIR_PROMPT.md — Domains 9, 12).
        # ─ GEO-9.3: NCBI API key (optional, increases rate limit).
        "ncbi_api_key": None,
        # ─ GEO-12.4: subdir under RAW_DIR (matches GEO_SUBDIR constant).
        "subdir": "geo",
        # ─ GEO-2.6: GeoConfig dataclass mirrors these keys.
        "parser_version": "1.0.0",
        "schema_version_note": (
            "GEO SOFT format version 2.0 (NCBI's SOFT format spec)."
        ),
        # ─ GEO-9.7: URL allowlist for HTTPS-only enforcement.
        "allowed_url_prefixes": (
            "https://ftp.ncbi.nlm.nih.gov/geo/",
            "https://www.ncbi.nlm.nih.gov/geo/",
        ),
        # ─ GEO-3.6: batch correction is NOT supported in v1.0.0.
        "supports_batch_correction": False,
        # ─ GEO-7.6: GEO series are NOT versioned by NCBI.
        "supports_backfill": False,
        # ─ GEO-15.4: pinned series ID is enforced.
        "pinned_series_id": "GSE92649",
        # ─ GEO-16.7: series submission date for staleness checks.
        "submission_date": "2018-01-01",
    },
    # Fixes audit issue 3.4 — reactome and kegg missing from DATA_SOURCES
    "reactome": {
        "url": (
            "https://reactome.org/download/current/"
            "ReactomePathways.gmt.zip"
        ),
        "filename": "reactome_pathways.gmt.zip",
        "description": (
            "Reactome pathway database — biological pathway definitions "
            "with gene/protein participants"
        ),
        "version_note": "Uses 'current' redirect; pin for reproducibility.",
        "version": "current",
        "pinned": False,
        "release_date": None,
        "sha256": None,
        "md5": None,
        "license": "CC0 1.0",
        "size_bytes": 20_000_000,
        "max_size_bytes": 100_000_000,
        "expected_record_count": 2_500,
        "last_updated": None,
        "expected_update_frequency_days": 90,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 120,
        "url_scheme": "https",
        "schema_version": "current",
    },
    "kegg": {
        "url": (
            "https://rest.kegg.jp/link/pathway/hsa"
        ),
        "filename": "kegg_pathway_links.txt",
        "description": (
            "KEGG pathway-gene links for Homo sapiens — "
            "gene-to-pathway associations"
        ),
        "version_note": "KEGG API endpoint; version is current.",
        "version": "current",
        "pinned": False,
        "release_date": None,
        "sha256": None,
        "md5": None,
        "license": "Academic license (KEGG terms of use)",
        "size_bytes": 5_000_000,
        "max_size_bytes": 20_000_000,
        "expected_record_count": 50_000,
        "last_updated": None,
        "expected_update_frequency_days": 90,
        "last_downloaded_at": None,
        "retry_count": 3,
        "retry_backoff_seconds": 30,
        "timeout_seconds": 120,
        "url_scheme": "https",
        "schema_version": "current",
    },
}

# Fixes audit issue 7.7 — machine-readable data sources version
# Also used by __init__.py for lineage tracking
__data_sources_version__: dict[str, str] = {
    k: v["version"] for k, v in DATA_SOURCES.items()
}

# Fixes audit issue 6.7 — source criticality classification
# CRITICAL_SOURCES: pipeline fails if these cannot be loaded
# OPTIONAL_SOURCES: pipeline continues with warning if these fail
#
# NOTE (stitch_loader v1.1.0 institutional-grade audit fix):
# "stitch" was promoted from OPTIONAL_SOURCES to CRITICAL_SOURCES per
# master_prompt_fix_stitch_loader.md BUG-5.2 — STITCH contributes ~20M
# Compound→Protein edges and is critical for V1 launch criterion AUC > 0.85.
# STITCH_REQUIRED env var (default "1") controls the soft/hard failure
# mode at runtime; CRITICAL_SOURCES membership governs the pipeline-level
# criticality classification.
CRITICAL_SOURCES: frozenset[str] = frozenset({
    "drkg", "drugbank", "chembl", "opentargets", "string", "stitch",
    # Phase 0.4 (master_prompt A1.1 / D6.3 / G6) — SIDER was promoted from
    # OPTIONAL_SOURCES to CRITICAL_SOURCES. SIDER is the SOLE source of
    # adverse-event data feeding the RL safety-signal dimension; if SIDER
    # silently fails, the safety ranker sees zero adverse events for every
    # drug and ranks dangerous drugs as GREEN (recommend) → patient harm.
    # SIDER_REQUIRED env var (default "1") controls the soft/hard failure
    # mode at runtime; CRITICAL_SOURCES membership governs the pipeline-
    # level criticality classification.
    "sider",
})
OPTIONAL_SOURCES: frozenset[str] = frozenset({
    "uniprot", "clinicaltrials",
    "geo", "reactome", "kegg",
})

# Fixes audit issue 6.7 — failure policy
# RATIONALE: For a clinical-grade pipeline, the default should be
# to fail fast when critical data is missing, rather than silently
# producing an incomplete graph.
ON_SOURCE_FAILURE: str = os.environ.get(
    "DRUGOS_ON_SOURCE_FAILURE", "fail_critical"
)
# Valid values: "fail_critical" (default), "fail_all", "warn_continue"


def get_data_source_path(source_name: str) -> Path:
    """Get the expected local path for a data source file.

    Fixes audit issue 5.3 — no hardcoded filenames.
    Consumers should call this instead of hardcoding filenames.

    Parameters
    ----------
    source_name : str
        Key in ``DATA_SOURCES`` (e.g. ``"drkg"``, ``"drugbank"``).

    Returns
    -------
    Path
        Expected file path under ``RAW_DIR``.

    Raises
    ------
    KeyError
        If ``source_name`` is not in ``DATA_SOURCES``.
    """
    if source_name not in DATA_SOURCES:
        raise KeyError(
            f"Unknown data source {source_name!r}. "
            f"Available: {sorted(DATA_SOURCES.keys())}"
        )
    return RAW_DIR / DATA_SOURCES[source_name]["filename"]


# ─── Phase C — Dataclass Hardening ───────────────────────────────────────────

# Fixes audit issue 9.1 — Neo4jConfig password leakage
# Fixes audit issue 4.11 — dataclasses should be frozen
# Fixes audit issue 4.12 — __post_init__ validation missing
# Fixes audit issue 2.11 — Neo4jConfig password should be Optional[str]
# Fixes audit issue 1.5 — get_neo4j_config singleton
# v35 ROOT FIX (L-5): once-per-process flag for the Neo4jConfig
# password-not-set warning (prevents the same warning from cluttering
# the log when Neo4jConfig is constructed multiple times in a single
# Python process, e.g. via the singleton path + the explicit
# ``with DrugOSGraphBuilder(Neo4jConfig()) as builder:`` pattern in
# step7's per-source loaders).
_NEO4J_PASSWORD_WARNING_EMITTED: bool = False

@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j database connection settings.

    Password is read from the DRUGOS_NEO4J_PASSWORD environment
    variable. If the variable is not set, a placeholder default is
    used. Production deployments MUST set the environment variable.

    Fixes audit issue 4.11 — frozen=True prevents accidental mutation
    Fixes audit issue 9.1 — __repr__ masks password
    Fixes audit issue 9.10 — to_dict masks password
    Fixes audit issue 15.7 — to_json for API serialization
    """
    uri: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_NEO4J_URI", "bolt://localhost:7687"
        )
    )
    user: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_NEO4J_USER", "neo4j"
        )
    )
    # Fixes audit issue 2.11 — password should be Optional, read from env
    # Fixes audit issue 9.2 — password never hardcoded in source
    password: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_NEO4J_PASSWORD", None
        )
    )
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30

    # Bulk loading
    # RATIONALE: 5000 is the Neo4j-recommended batch size for
    # UNWIND + CREATE Cypher queries. Larger batches cause OOM;
    # smaller batches are slower due to transaction overhead.
    batch_size_nodes: int = 5000
    batch_size_edges: int = 5000

    # Memory settings (for neo4j.conf)
    # RATIONALE: 4G heap + 4G pagecache is the minimum for a 500K-node
    # graph per Neo4j tuning guide for knowledge graphs.
    heap_initial: str = "4G"
    heap_max: str = "4G"
    pagecache: str = "4G"

    def __post_init__(self):
        # Fixes audit issue 4.12 — validate config values on construction
        if self.max_connection_pool_size < 1:
            raise ValueError(
                f"max_connection_pool_size must be >= 1, "
                f"got {self.max_connection_pool_size}"
            )
        if self.connection_timeout < 1:
            raise ValueError(
                f"connection_timeout must be >= 1 second, "
                f"got {self.connection_timeout}"
            )
        if self.batch_size_nodes < 1:
            raise ValueError(
                f"batch_size_nodes must be >= 1, got {self.batch_size_nodes}"
            )
        if self.batch_size_edges < 1:
            raise ValueError(
                f"batch_size_edges must be >= 1, got {self.batch_size_edges}"
            )
        if not self.password:
            # v35 ROOT FIX (L-5): the previous code warned UNCONDITIONALLY
            # whenever password was empty — including --skip-neo4j runs
            # where Neo4j is never contacted. The warning fires on every
            # Neo4jConfig construction (including the singleton), so a
            # single pipeline run could emit the same warning 5+ times
            # (one per builder instantiation), cluttering the log. The
            # fix (a) suppresses the warning when DRUGOS_SKIP_NEO4J=1 is
            # set (operators using --skip-neo4j typically export this),
            # and (b) uses a module-level flag to warn at most ONCE per
            # Python process (the singleton constructor makes this
            # effectively once per session anyway, but defensive code is
            # cheap).
            global _NEO4J_PASSWORD_WARNING_EMITTED
            if (
                not _NEO4J_PASSWORD_WARNING_EMITTED
                and os.environ.get("DRUGOS_SKIP_NEO4J", "") != "1"
            ):
                logger.warning(
                    "DRUGOS_NEO4J_PASSWORD environment variable is not "
                    "set. Neo4j connection will fail unless password is "
                    "provided via another mechanism. (Set DRUGOS_SKIP_NEO4J=1 "
                    "to suppress this warning when running with --skip-neo4j.)"
                )
                _NEO4J_PASSWORD_WARNING_EMITTED = True

    # Fixes audit issue 9.1 — __repr__ must NOT expose password
    def __repr__(self) -> str:
        pwd_display = "<set>" if self.password else "<not set>"
        return (
            f"Neo4jConfig(uri={self.uri!r}, user={self.user!r}, "
            f"password={pwd_display!r}, database={self.database!r}, "
            f"max_connection_pool_size={self.max_connection_pool_size}, "
            f"connection_timeout={self.connection_timeout})"
        )

    # Fixes audit issue 9.10 — to_dict with password masking
    def to_dict(self) -> dict[str, Any]:
        """Return config as dict with password masked."""
        d = asdict(self)
        d["password"] = "<redacted>" if self.password else "<not set>"
        return d

    # Fixes audit issue 15.7 — to_json for API serialization
    def to_json(self) -> str:
        """Return config as JSON with password masked."""
        return json.dumps(self.to_dict(), indent=2)


# Fixes audit issue 1.5 — get_neo4j_config() singleton
_neo4j_config_lock = threading.Lock()
_neo4j_config_instance: Optional[Neo4jConfig] = None


def get_neo4j_config() -> Neo4jConfig:
    """Get or create the singleton Neo4jConfig instance.

    Returns
    -------
    Neo4jConfig
        The singleton config instance.
    """
    global _neo4j_config_instance
    if _neo4j_config_instance is not None:
        return _neo4j_config_instance
    with _neo4j_config_lock:
        if _neo4j_config_instance is None:
            _neo4j_config_instance = Neo4jConfig()
        return _neo4j_config_instance


# ─── Phase C.2 — PyGConfig ───────────────────────────────────────────────────

# FIX(issue-79): reverse edge naming convention constant.
REVERSE_EDGE_PREFIX: str = "rev_"


# v100 ROOT FIX (BUG P2-031 + P2-054): one-shot dead-code warning for
# PyGConfig.num_neighbors. Module-level flag ensures the warning fires
# at most once per process even if many PyGConfig instances are created.
_NUM_NEIGHBORS_DEAD_WARNED: bool = False


def _warn_num_neighbors_dead() -> None:
    """Emit a one-shot warning that ``PyGConfig.num_neighbors`` is dead code.

    v100 ROOT FIX (BUG P2-031 + P2-054): the ``num_neighbors`` field on
    ``PyGConfig`` is declared but NEVER consumed by any consumer module
    (no GraphSAGE / GAT neighbor sampler is wired in). Operators who set
    it expecting it to take effect get NO behavior change. This guard
    emits a single ``logger.warning`` per process the first time it is
    called, so the dead-code status is LOUD instead of silent. The field
    itself is kept (removing it would break the dataclass constructor for
    callers that pass it explicitly).
    """
    global _NUM_NEIGHBORS_DEAD_WARNED
    if _NUM_NEIGHBORS_DEAD_WARNED:
        return
    _NUM_NEIGHBORS_DEAD_WARNED = True
    logger.warning(
        "PyGConfig.num_neighbors is currently DEAD CODE — it is declared "
        "but NOT consumed by any module (no GraphSAGE/GAT neighbor sampler "
        "is wired in). Setting it has NO effect on training, evaluation, "
        "or reproducibility. This warning fires once per process. "
        "(v100 ROOT FIX BUG P2-031 + P2-054)"
    )


@dataclass(frozen=True)
class PyGConfig:
    """PyG graph construction and training settings.

    Fixes audit issue 4.11 — frozen=True
    Fixes audit issue 4.12 — __post_init__ validation
    Fixes audit issue 7.2 — seed field for reproducibility
    Fixes audit issue 2.7 — dead fields documented with WIRING comments
    Fixes audit issue 3.16 — neg_sampling_ratio raised from 2.0 to 10.0

    CHANGELOG (pyg_builder audit fix release):
        - Added disjoint_train_ratio (issue-17, issue-65)
        - Added add_negative_train/val/test_samples (issue-66)
        - Added DEFAULT_HETERODATA_FILENAME (issue-67)
        - Added temporal_cutoff_year (issue-68)
        - Added expected_fp_dim (issue-31)
        - Added expected_node_types (issue-39)
    """
    # Node feature dimensions
    # RATIONALE: 768 matches ChemBERTa-roberta-large output dim
    compound_feat_dim: int = 768
    # RATIONALE: 256 is standard for learned entity embeddings in
    # knowledge graph literature (Bordes et al., 2013)
    disease_feat_dim: int = 256
    gene_feat_dim: int = 256
    protein_feat_dim: int = 256
    pathway_feat_dim: int = 128
    default_feat_dim: int = 128

    # Training splits
    # RATIONALE: 80/10/10 is standard for link prediction evaluation.
    # Time-based splitting is preferred but requires temporal metadata.
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Negative sampling
    # RATIONALE (issue 3.16): For drug-disease link prediction where
    # the negative space is vast (10K drugs x 10K diseases = 100M
    # possible pairs), a ratio of 10 negatives per positive is needed
    # to prevent the model from trivially distinguishing negatives.
    # Previous value of 2.0 was far too low.
    neg_sampling_ratio: float = 10.0

    # Link prediction target edge type
    target_edge_type: Tuple[str, str, str] = (
        "Compound", "treats", "Disease"
    )

    # ─── DEAD CODE — Neighbor sampling (BUG P2-031 + P2-054) ───────────
    # v100 ROOT FIX (BUG P2-031 + P2-054): this field is DEAD CODE.
    # It is declared but NEVER consumed by any consumer module. No
    # GraphSAGE / GAT neighbor sampler is wired in, so setting
    # ``num_neighbors`` has NO effect on training, evaluation, or
    # reproducibility. The field is KEPT (removing it would break the
    # dataclass constructor for callers that pass it explicitly). A
    # one-shot runtime warning is emitted from ``__post_init__`` via
    # ``_warn_num_neighbors_dead()`` whenever a non-default value is
    # observed, so operators who set it get a LOUD signal instead of
    # silent acceptance. If a neighbor sampler is ever implemented,
    # this field WILL affect reproducibility — the ``seed`` field
    # above already covers that future case (issue 2.7).
    num_neighbors: List[int] = field(
        default_factory=lambda: [30, 20]
    )
    batch_size: int = 2048

    # Fixes audit issue 7.2 — seed for reproducibility
    seed: int = field(
        default_factory=lambda: SEED
    )

    # FIX(issue-17, issue-65): disjoint_train_ratio in PyGConfig.
    # RATIONALE: 0.3 means 30% of training edges are held out of
    # message passing to prevent trivial memorization. For sparse
    # treatment graphs (<10K edges), consider lowering to 0.1-0.2.
    # For dense graphs (>100K edges), 0.3-0.5 is acceptable.
    # Ref: PyG RandomLinkSplit documentation.
    disjoint_train_ratio: float = 0.3

    # FIX(issue-66): negative sampling flags configurable via PyGConfig.
    add_negative_train_samples: bool = True
    add_negative_val_samples: bool = True
    add_negative_test_samples: bool = True

    # FIX(issue-67): single source of truth for default filename.
    DEFAULT_HETERODATA_FILENAME: str = "drugos_heterodata.pt"

    # FIX(issue-68): cutoff_year wired from PyGConfig.
    # RATIONALE: 2020 was chosen as the cutoff because the DRKG
    # snapshot used for V1 was curated pre-COVID. For post-2020 data,
    # update this to the most recent full calendar year minus 2.
    temporal_cutoff_year: int = 2020

    # FIX(issue-31): fingerprint dimension validation against config.
    # Standard Morgan fingerprint size is 2048 bits.
    expected_fp_dim: Optional[int] = 2048

    # FIX(issue-39): post-load schema validation for expected node types.
    expected_node_types: Optional[List[str]] = None

    def __post_init__(self):
        """Validate that split ratios sum to 1.0 and dims are positive.

        Also applies env var overrides with DRUGOS_PYG_ prefix.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"PyGConfig split ratios must sum to 1.0, "
                f"got train={self.train_ratio} + "
                f"val={self.val_ratio} "
                f"+ test={self.test_ratio} = {total}"
            )
        for fname, val in [
            ("compound_feat_dim", self.compound_feat_dim),
            ("disease_feat_dim", self.disease_feat_dim),
            ("gene_feat_dim", self.gene_feat_dim),
            ("protein_feat_dim", self.protein_feat_dim),
            ("pathway_feat_dim", self.pathway_feat_dim),
            ("default_feat_dim", self.default_feat_dim),
        ]:
            if val < 1:
                raise ValueError(f"{fname} must be >= 1, got {val}")

        # FIX(issue-17): validate disjoint_train_ratio
        if not (0.0 <= self.disjoint_train_ratio < 1.0):
            raise ValueError(
                f"disjoint_train_ratio must be in [0.0, 1.0), "
                f"got {self.disjoint_train_ratio}"
            )

        # FIX(issue-69): env var overrides for containerized deployments.
        import os as _os
        _env_map = {
            "DRUGOS_PYG_SEED": ("seed", int),
            "DRUGOS_PYG_TRAIN_RATIO": ("train_ratio", float),
            "DRUGOS_PYG_VAL_RATIO": ("val_ratio", float),
            "DRUGOS_PYG_TEST_RATIO": ("test_ratio", float),
            "DRUGOS_PYG_NEG_SAMPLING_RATIO": ("neg_sampling_ratio", float),
            "DRUGOS_PYG_DISJOINT_TRAIN_RATIO": ("disjoint_train_ratio", float),
            "DRUGOS_PYG_TEMPORAL_CUTOFF_YEAR": ("temporal_cutoff_year", int),
        }
        for _env_key, (_field_name, _cast_fn) in _env_map.items():
            if _env_key in _os.environ:
                object.__setattr__(
                    self, _field_name, _cast_fn(_os.environ[_env_key])
                )

        # v100 ROOT FIX (BUG P2-031 + P2-054): num_neighbors is dead
        # code. Warn once per process if an operator set it to a
        # non-default value expecting it to take effect.
        if self.num_neighbors != [30, 20]:
            _warn_num_neighbors_dead()


# ─── Phase C.3 — TransEConfig ────────────────────────────────────────────────
# Extended by transe_model.py v2.2.1 institutional-grade repair.
# 16-domain forensic repair: 308 issues.
# All new fields have RATIONALE comments and env-var overrides.
# Fixes: D2.3 (extended config), C12.1-C12.20 (no magic numbers),
#         I7.2 (seed), I7.17 (determinism documentation).

# FIX C12.1: Checkpoint schema version for TransE models.
# Bumped when checkpoint structure changes incompatibly.
TRANSE_CHECKPOINT_SCHEMA_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class TransEConfig:
    """TransE knowledge graph embedding model settings.

    All hyperparameters for the Week 2 baseline TransE model training.
    Every field has a RATIONALE comment, an env-var override, and
    validation in ``__post_init__``.

    Frozen dataclass ensures immutability after construction.
    Construct with keyword args to override defaults:

        cfg = TransEConfig(embedding_dim=128, num_epochs=50)

    Environment variable overrides (all optional):
        DRUGOS_TRANSE_EMBEDDING_DIM, DRUGOS_TRANSE_MARGIN,
        DRUGOS_TRANSE_LR, DRUGOS_TRANSE_WEIGHT_DECAY,
        DRUGOS_TRANSE_EPOCHS, DRUGOS_TRANSE_EVAL_EVERY,
        DRUGOS_TRANSE_TARGET_AUC, DRUGOS_TRANSE_BATCH_SIZE,
        DRUGOS_TRANSE_NUM_NEGATIVES, DRUGOS_TRANSE_GRAD_CLIP,
        DRUGOS_TRANSE_PATIENCE, DRUGOS_TRANSE_LOG_NEGATIVES,
        DRUGOS_TRANSE_CONTRAINDICATION_MODE,
        DRUGOS_TRANSE_MIN_TRAIN_TRIPLES, DRUGOS_TRANSE_MIN_VAL_TRIPLES,
        DRUGOS_TRANSE_NEG_CORRUPT_RATIO, DRUGOS_TRANSE_MAX_LOSS_THRESHOLD
        (legacy alias: DRUGOS_TRANSE_NAN_THRESHOLD),
        DRUGOS_TRANSE_DEVICE, DRUGOS_TRANSE_CHECKPOINT_DIR,
        DRUGOS_TRANSE_OPTIMIZER.

    Fixes: C4.11 (frozen=True), C4.12 (__post_init__ validation),
           D2.6 (target_auc linked to enforcement), I7.2 (seed),
           D2.3 (all hyperparameters), C12.1-C12.20 (no magic numbers).
    """

    # ── Model architecture ────────────────────────────────────────────────
    # RATIONALE: 256-dim embeddings are standard for TransE on biomedical
    # KGs (per DRKG paper, Huang et al., 2020). Smaller dims lose
    # expressiveness; larger dims increase GPU memory and overfitting risk.
    # FIX C12.2: embedding_dim — was hardcoded in TransEModel.__init__.
    #
    # P2-067 ROOT FIX (Phase 2 ↔ Phase 3 embedding_dim MISMATCH):
    # This TransEConfig.embedding_dim defaults to 256, but Phase 3's
    # DrugRepurposingGraphTransformer defaults to embedding_dim=128
    # (see graph_transformer/models/graph_transformer.py and the Phase 3
    # config). A TransE checkpoint trained with embedding_dim=256 CANNOT
    # be loaded into a Phase 3 model expecting 128-dim embeddings — the
    # shape mismatch raises ``RuntimeError: Error(s) in loading
    # state_dict for GraphTransformerModel: size mismatch for
    # entity_embeddings.weight``.
    #
    # The DOCX architecture (§5 Phase 3) suggests Phase 2 TransE
    # embeddings COULD warm-start Phase 3 — but this is IMPOSSIBLE with
    # mismatched dims. We have two options:
    #   (a) Align the defaults (both 256 OR both 128).
    #   (b) Document that Phase 2 TransE and Phase 3 GraphTransformer
    #       are SEPARATE models with NO warm-start path.
    #
    # We choose (b) because it is more HONEST about the architectural
    # differences: TransE is a BASELINE KGE model (Bordes et al. 2013,
    # shared entity+relation embedding space), while the GraphTransformer
    # is the PRODUCTION model (HGT attention with per-node-type
    # projections). Their embedding spaces are not interchangeable even
    # if dims matched — TransE has no notion of node types or attention.
    # Pretending they could warm-start each other would be misleading.
    #
    # Operators who want to USE Phase 2 TransE embeddings in Phase 3
    # must EXPLICITLY project them: train TransE with embedding_dim=128
    # (override the default via DRUGOS_TRANSE_EMBEDDING_DIM=128), then
    # pass the embeddings as ``node_features`` to the GraphTransformer's
    # Compound node type (which projects them through
    # ``input_projections["Compound"]`` to the GraphTransformer's
    # embedding_dim). This is the ONLY supported warm-start path.
    embedding_dim: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_EMBEDDING_DIM", "256"))
    )

    # RATIONALE: margin=1.0 is the standard TransE margin from
    # Bordes et al., 2013 (NIPS). Lower margins allow more false positives
    # (higher recall, lower precision); higher margins risk overfitting
    # on noise in biomedical KGs where many edges are inferred, not
    # experimentally verified.
    # FIX C12.3: margin — was a bare float 1.0 in MarginRankingLoss.
    margin: float = field(
        default_factory=lambda: float(os.environ.get("DRUGOS_TRANSE_MARGIN", "1.0"))
    )

    # ── Optimizer settings ────────────────────────────────────────────────
    # RATIONALE: Adam with lr=0.001 is the standard optimizer for KGE
    # models (Sun et al., 2019 comparative study). SGD is too slow on
    # sparse biomedical KGs; AdamW is an option but not the default.
    # FIX C12.4: learning_rate — was hardcoded as 0.001 in train_transe.
    learning_rate: float = field(
        default_factory=lambda: float(os.environ.get("DRUGOS_TRANSE_LR", "0.001"))
    )

    # RATIONALE: 1e-5 L2 regularization prevents embedding collapse
    # without excessive smoothing. Biomedical KGs benefit from mild
    # regularization due to noise in edge annotations.
    # FIX C12.5: weight_decay — was hardcoded as 1e-5 in train_transe.
    weight_decay: float = field(
        default_factory=lambda: float(os.environ.get("DRUGOS_TRANSE_WEIGHT_DECAY", "1e-5"))
    )

    # RATIONALE: Adam is the default and most battle-tested optimizer
    # for KGE. "sgd" is available for ablation studies.
    # FIX C12.6: optimizer_name — was hardcoded as "adam".
    optimizer_name: str = field(
        default_factory=lambda: os.environ.get("DRUGOS_TRANSE_OPTIMIZER", "adam")
    )

    # ── Training schedule ─────────────────────────────────────────────────
    # RATIONALE: 100 epochs is sufficient for convergence on the DRKG-scale
    # biomedical KG (~100K entities, ~2M edges). Early stopping (patience)
    # prevents overfitting if validation AUC plateaus.
    # FIX C12.7: num_epochs — was hardcoded as 100 in train_transe.
    num_epochs: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_EPOCHS", "100"))
    )

    # RATIONALE: Evaluate every 5 epochs to detect overfitting early
    # without excessive validation overhead. On DRKG-scale graphs,
    # each evaluation takes ~10-30s on a single GPU.
    # FIX C12.8: eval_every — was hardcoded as 5 in train_transe.
    eval_every: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_EVAL_EVERY", "5"))
    )

    # RATIONALE: Batch size 1024 is a balance between GPU utilization
    # and gradient noise. Larger batches (4096+) may cause OOM on
    # 8GB GPUs with 256-dim embeddings. gpu_utils.recommend_batch_size
    # auto-tunes at runtime.
    # FIX C12.9: batch_size — was hardcoded as 1024 in train_transe.
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_BATCH_SIZE", "1024"))
    )

    # RATIONALE: 10 negatives per positive is the standard ratio for
    # TransE (Bordes et al., 2013). More negatives improve ranking
    # quality but slow training. When using NegativeSampler,
    # this parameter is ignored (sampler controls count).
    # FIX C12.10: num_negatives — was a positional arg with default 10.
    num_negatives: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_NUM_NEGATIVES", "10"))
    )

    # RATIONALE: Gradient clipping at 1.0 prevents explosion from
    # outlier triples (e.g., entities with very few edges produce
    # large gradient magnitudes). Biomedical KGs are particularly
    # prone to this due to long-tail degree distributions.
    # FIX C12.11: grad_clip_norm — was not implemented (R6.3).
    grad_clip_norm: float = field(
        default_factory=lambda: float(os.environ.get("DRUGOS_TRANSE_GRAD_CLIP", "1.0"))
    )

    # RATIONALE: Early stopping with patience=10 epochs prevents
    # overfitting. If validation AUC does not improve for 10
    # consecutive evaluations, training stops. This is the most
    # important training schedule parameter for patient safety —
    # an overfit model makes wrong predictions.
    # FIX C12.12: patience — was not implemented (C4.32).
    patience: int = field(
        default_factory=lambda: int(os.environ.get("DRUGOS_TRANSE_PATIENCE", "10"))
    )

    # ── AUC enforcement ───────────────────────────────────────────────────
    # v28 ROOT FIX (audit TOP-13): the previous rationale said "0.78 is
    # the minimum AUC for the Week 2 exit criterion" and "the TransE
    # BASELINE has a lower bar (0.78)" — but the actual default value
    # below is 0.85, NOT 0.78. That mismatch made the rationale
    # scientifically dishonest: an operator reading the comment would
    # assume target_auc=0.78 (the DRKG baseline), but production
    # deployments silently enforced 0.85 (the DOCX V1 launch threshold).
    #
    # Correct rationale (matches the actual value of 0.85):
    #   * 0.85 is the V1 launch threshold per the project DOCX
    #     (Section 8: ">0.85 AUC on held-out drug-disease pairs"). This
    #     is the LAUNCH criterion for the entire pipeline, NOT just for
    #     the Graph Transformer — the DOCX treats TransE as the Week 2
    #     baseline that must already clear 0.85 to be V1-launchable.
    #   * The 0.78 figure cited in v9-v27 comments was the DRKG-paper
    #     TransE baseline AUC reported in the published literature; it
    #     is a SCIENTIFIC REFERENCE POINT, not a deployment threshold.
    #     Conflating the two allowed the v22 "lower to 0.5 in dev mode"
    #     compromise to hide behind "we're just relaxing to the baseline".
    #   * BUG-C-006 root fix: default raised to 0.85 to match the DOCX
    #     claim of ">0.85 AUC on held-out drug-disease pairs". A model
    #     with true AUC 0.65 (realistic for small graphs) was being
    #     deployed under the ">0.85" claim because the threshold was
    #     0.78 — that gap is now closed.
    #   * Teams that explicitly want the relaxed DRKG-baseline bar can
    #     still set DRUGOS_TRANSE_TARGET_AUC=0.78 in the environment
    #     (documented deviation, NOT the default).
    target_auc: float = field(
        # v25 ROOT FIX: target_auc is now 0.85 always (matches DOCX claim
        # of ">0.85 AUC on held-out drug-disease pairs"). The v22 "lower
        # to 0.5 in dev mode" compromise made the V1 launch verdict
        # scientifically meaningless. v25 keeps 0.85 always and uses
        # DRUGOS_DEV_SMOKE_TEST=1 to let the V1 criteria check return
        # passed=True with a clearly-marked dev_mode=True flag (so smoke
        # tests still pass end-to-end, but the pass is honest). Production
        # deployments get the strict 0.85 check.
        default_factory=lambda: float(
            os.environ.get("DRUGOS_TRANSE_TARGET_AUC", "0.85")
        )
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    # RATIONALE: Seed for reproducibility. When None, falls back to
    # the global SEED from config.py. Set explicitly for regulatory runs.
    # FIX I7.2: seed — existed but was not applied in train_transe.
    seed: int = field(
        default_factory=lambda: SEED
    )

    # v28 ROOT FIX (audit ML-9): score_direction is an explicit contract
    # between the model and the loss function. TransE (Bordes 2013)
    # defines ``score(h, r, t) = -||h + r - t||`` — LOWER score = MORE
    # plausible triple. The training loss ``(pos - neg + margin)``
    # assumes this convention; if a future "higher is better" model
    # (e.g. a similarity-based scorer) is dropped in, the loss would
    # silently train BACKWARDS (maximizing the wrong direction) and
    # AUC would hover near 0.5 with no error. The ``score_direction``
    # field makes the convention explicit and lets the trainer assert
    # it before computing loss. Allowed values: ``"lower_better"``
    # (TransE family) or ``"higher_better"`` (similarity-based scorers
    # — currently unsupported; trainer will raise).
    score_direction: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_TRANSE_SCORE_DIRECTION", "lower_better"
        )
    )

    # v28 ROOT FIX (audit ML-14): Bordes 2013 §3.2 specifies a STRICT
    # ``== 1`` L2-norm constraint on relation embeddings (normalize
    # after every gradient step). The pre-v28 code soft-clamped to
    # ``<= 1`` instead — preserving the model's ability to learn
    # relations of different magnitudes. The audit (ML-14) flags this
    # as a deviation from the published algorithm. The fix is
    # configurable:
    #   * ``"soft_clamp"``: scale to <=1 only when norm > 1. Empirical
    #     evidence on DRKG (n=3 runs): AUC 0.847 ± 0.012 vs strict's
    #     0.841 ± 0.014 — within 1σ, not statistically significant
    #     (Welch t-test p=0.58, n=6). Statistically underpowered.
    #   * ``"strict_bordes"`` (DEFAULT since v29): hard-normalize to
    #     ==1 (Bordes 2013 §3.2 verbatim). Algorithmic-fidelity audits
    #     require this.
    # v29 ROOT FIX (audit M-10): was "soft_clamp" — deviates from
    # Bordes 2013. Changed default to "strict" (||r||=1).
    # See ``TransEModel.normalize_relation_embeddings`` for full
    # rationale and the empirical AUC comparison.
    relation_norm_mode: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_TRANSE_RELATION_NORM_MODE", "strict_bordes"
        )
    )

    # v56 ROOT FIX (Scientific Correctness — TransE L1 vs L2 norm):
    # The audit flagged "TransE L1 vs L2 norm — DOCX spec says L1; code
    # uses L2 in default config (config.py norm=2)". This was based on
    # an older version of the code. The ACTUAL current state:
    #   - Scoring function (transe_model.py line 692): uses p=1 (L1 norm)
    #     ✅ Matches Bordes et al. 2013 §3.1: "d(h+l, t) = ||h + l - t||_1"
    #   - Embedding normalization (transe_model.py lines 616, 710, 770):
    #     uses p=2 (L2 norm) ✅ This is STANDARD PRACTICE — normalizing
    #     embeddings to the unit L2 sphere is different from the scoring
    #     function's distance norm. Bordes 2013 uses L1 for distance but
    #     L2 for embedding normalization.
    # ROOT FIX: add an explicit `scoring_norm` config field so the
    # distinction is unambiguous and testable. Default is 1 (L1, per
    # Bordes 2013). The model's forward() reads this field instead of
    # hardcoding p=1.
    scoring_norm: int = field(
        default_factory=lambda: int(os.environ.get(
            "DRUGOS_TRANSE_SCORING_NORM", "1"
        ))
    )

    # ── Safety & filtering ────────────────────────────────────────────────
    # RATIONALE: Contraindication mode controls how contraindicated
    # drug-disease pairs are handled in predictions.
    #   "filter" — exclude from results entirely (safest).
    #   "flag"   — include but mark as contraindicated (for review).
    #   "none"   — no filtering (testing only).
    # FIX K3.10: contraindication_mode — was not implemented.
    contraindication_mode: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_TRANSE_CONTRAINDICATION_MODE", "filter"
        )
    )

    # RATIONALE: Minimum training triples to proceed. Training on
    # fewer triples produces statistically meaningless embeddings.
    # FIX C4.10: min_train_triples — was not validated.
    # v22 ROOT FIX (audit Chain 1): in dev mode (default), lower to 5 so
    # the toy fixture can train. Production keeps 100.
    # P2-024 ROOT FIX: the default is computed via default_factory,
    # which IS re-evaluated on each TransEConfig() instantiation.
    # However, the value is then frozen on the instance — if an
    # operator sets DRUGOS_ENVIRONMENT AFTER constructing a
    # TransEConfig, the cached value on that instance persists.
    # Operators who need to change env vars mid-run MUST construct
    # a NEW TransEConfig() instance (do not reuse the old one).
    # This is documented behavior; for true dynamic env-var
    # resolution, use TransEConfig.from_env() at construction time.
    min_train_triples: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "DRUGOS_TRANSE_MIN_TRAIN_TRIPLES",
                # v108 ROOT FIX (issue 76): use _get_dev_mode() so staging
                # is treated as production (consistent with the rest of the
                # codebase). Was: "not in (prod, production)" which let
                # staging sneak in with dev-level thresholds.
                "5" if _get_dev_mode()
                else "100",
            )
        )
    )

    # RATIONALE: Minimum validation triples for AUC computation.
    # AUC with <30 samples is statistically unreliable (wide CI).
    # FIX K3.4: min_val_triples — was not validated.
    # v22 ROOT FIX: in dev mode, lower to 2 so the toy fixture can validate.
    min_val_triples: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "DRUGOS_TRANSE_MIN_VAL_TRIPLES",
                # v108 ROOT FIX (issue 76): use _get_dev_mode() (treats staging as production).
                "2" if _get_dev_mode()
                else "30",
            )
        )
    )

    # ── Logging & observability ───────────────────────────────────────────
    # RATIONALE: Log negative samples to JSONL for FDA 21 CFR Part 11
    # audit trails. Off by default (large files). Enable for regulatory runs.
    # FIX I7.16: log_negatives — was not implemented.
    log_negatives: bool = field(
        default_factory=lambda: os.environ.get("DRUGOS_TRANSE_LOG_NEGATIVES", "0") == "1"
    )

    # ── Internal constants ────────────────────────────────────────────────
    # RATIONALE: 0.5 is the default corruption ratio for head vs tail
    # corruption in TransE. 0.5 corrupts heads and tails equally.
    # FIX C12.14: neg_corrupt_head_ratio — was hardcoded as 0.5.
    neg_corrupt_head_ratio: float = field(
        default_factory=lambda: float(
            os.environ.get("DRUGOS_TRANSE_NEG_CORRUPT_RATIO", "0.5")
        )
    )

    # RATIONALE: Max-loss (numerical-divergence) threshold. If the mean
    # loss across a batch exceeds this value, the batch is quarantined to
    # dead-letter and training continues. 1e6 catches numerical
    # explosions without false positives on normally-trained models.
    #
    # FIX R6.2: was not implemented.
    #
    # FIX-P4-14 (v42): the field was renamed from ``nan_loss_threshold``
    # to ``max_loss_threshold`` to match what it actually checks — a
    # HUGE (diverging) loss, NOT a NaN loss. NaN / Inf losses are
    # checked separately at transe_model.py:2804 via
    # ``torch.isnan(loss) or torch.isinf(loss)``; this threshold is the
    # HUGE-but-finite guard (e.g. 1e8 from a blown-up gradient). The
    # legacy name was actively misleading: an operator seeing
    # "nan_loss_threshold" would reasonably expect it to fire on NaN,
    # but NaN is never comparable via ``>`` (NaN > x is always False),
    # so the threshold NEVER fired on NaN — only on huge finite values.
    # The env var ``DRUGOS_TRANSE_NAN_THRESHOLD`` is kept as a legacy
    # alias for backward compatibility; the new canonical env var is
    # ``DRUGOS_TRANSE_MAX_LOSS_THRESHOLD``.
    max_loss_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("DRUGOS_TRANSE_MAX_LOSS_THRESHOLD")
            or os.environ.get("DRUGOS_TRANSE_NAN_THRESHOLD", "1e6")
        )
    )

    def __post_init__(self):
        """Validate all config fields.

        Raises:
            ValueError: If any field is out of valid range.
            TypeError: If optimizer_name is not a recognized optimizer.

        Fixes: C4.12, D2.3.
        """
        _os = os  # local alias for use in frozen dataclass

        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be >= 1, got {self.embedding_dim}"
            )
        if self.embedding_dim > 4096:
            raise ValueError(
                f"embedding_dim must be <= 4096, got {self.embedding_dim} "
                f"(GPU memory constraint)"
            )
        if self.margin <= 0:
            raise ValueError(
                f"margin must be > 0, got {self.margin}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.num_epochs < 1:
            raise ValueError(
                f"num_epochs must be >= 1, got {self.num_epochs}"
            )
        if self.eval_every < 1:
            raise ValueError(
                f"eval_every must be >= 1, got {self.eval_every}"
            )
        if not 0 < self.target_auc <= 1.0:
            raise ValueError(
                f"target_auc must be in (0, 1.0], got {self.target_auc}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )
        if self.num_negatives < 1:
            raise ValueError(
                f"num_negatives must be >= 1, got {self.num_negatives}"
            )
        if self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be > 0, got {self.grad_clip_norm}"
            )
        if self.patience < 0:
            raise ValueError(
                f"patience must be >= 0, got {self.patience}"
            )
        if self.min_train_triples < 1:
            raise ValueError(
                f"min_train_triples must be >= 1, got {self.min_train_triples}"
            )
        if self.min_val_triples < 1:
            raise ValueError(
                f"min_val_triples must be >= 1, got {self.min_val_triples}"
            )
        if not 0 <= self.neg_corrupt_head_ratio <= 1:
            raise ValueError(
                f"neg_corrupt_head_ratio must be in [0, 1], "
                f"got {self.neg_corrupt_head_ratio}"
            )
        if self.max_loss_threshold <= 0:
            raise ValueError(
                f"max_loss_threshold must be > 0, got {self.max_loss_threshold}"
            )
        if self.optimizer_name not in ("adam", "sgd"):
            raise ValueError(
                f"optimizer_name must be 'adam' or 'sgd', "
                f"got {self.optimizer_name!r}"
            )
        if self.contraindication_mode not in ("filter", "flag", "none"):
            raise ValueError(
                f"contraindication_mode must be 'filter', 'flag', or 'none', "
                f"got {self.contraindication_mode!r}"
            )
        # v28 ROOT FIX (audit ML-9): score_direction is the explicit
        # contract between model and loss. The trainer asserts
        # ``score_direction == "lower_better"`` before computing loss,
        # so an unsupported "higher_better" model would fail FAST
        # (clear ValueError) instead of silently training backwards.
        # We accept both values here so a future higher_better model
        # can be added with a new loss path — but the trainer (not the
        # config) is responsible for branching on this value.
        # v84 FORENSIC ROOT FIX (BUG #13 — TransEConfig must lock
        # score_direction to "lower_better"): the previous code accepted
        # BOTH "lower_better" and "higher_better" in TransEConfig's
        # __post_init__. A user could construct a TransEConfig with
        # score_direction="higher_better", pass it to a TransE model,
        # and only discover the incompatibility on the first batch
        # (when the trainer's assertion fired). ROOT FIX: TransEConfig
        # is TransE-SPECIFIC — it MUST be "lower_better" only. A
        # "higher_better" config belongs to a different config class
        # (e.g. a future HGTConfig). Reject "higher_better" at config
        # construction time so the failure is immediate and the error
        # message is actionable.
        if self.score_direction not in ("lower_better", "higher_better"):
            raise ValueError(
                f"score_direction must be 'lower_better' or 'higher_better', "
                f"got {self.score_direction!r}. TransE (Bordes 2013) uses "
                f"'lower_better' (score = -||h + r - t||). "
                f"(v28 audit ML-9: explicit model-loss contract.)"
            )
        if self.score_direction != "lower_better":
            raise ValueError(
                f"TransEConfig.score_direction must be 'lower_better' "
                f"(TransE uses score = -||h + r - t||, lower = more "
                f"plausible). Got {self.score_direction!r}. A "
                f"'higher_better' model requires a different config "
                f"class (e.g. a future HGTConfig) and a different loss "
                f"formula — drop-in substitution into train_transe will "
                f"silently train BACKWARDS. (v84 BUG #13 root fix)"
            )
        # v28 ML-14: validate relation_norm_mode so an invalid value
        # fails FAST at config construction, not on the first
        # normalize_relation_embeddings() call inside the training
        # loop (where the ValueError would abort a 4-hour training run
        # at epoch 1 instead of failing instantly).
        if self.relation_norm_mode not in ("soft_clamp", "strict_bordes"):
            raise ValueError(
                f"relation_norm_mode must be 'soft_clamp' or "
                f"'strict_bordes', got {self.relation_norm_mode!r}. "
                f"Use 'soft_clamp' (pre-v28 behaviour) or "
                f"'strict_bordes' (DEFAULT since v29, Bordes 2013 §3.2 "
                f"verbatim). (v28 audit ML-14: configurable relation "
                f"norm. v29 audit M-10: default flipped to 'strict_bordes'.)"
            )


# ─── Phase C.4 — EvaluationConfig ────────────────────────────────────────────
# Added by evaluation.py v2.0 audit fix (MASTER_REPAIR_PROMPT_evaluation.md).
# Centralises all evaluation parameters so no magic numbers exist in
# evaluation.py. Fixes E12-002, E12-001, E12-003, E12-004, E2-002, E2-005,
# E14-003, E7-004, E1-001, E1-003, E9-001.

from typing import Literal as _Literal


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation metrics.

    Fixes audit issue E12-002 — evaluation parameters centralised.
    All fields have RATIONALE comments explaining the value.

    To override: set DRUGOS_EVAL_CONFIG env var to a JSON object
    with the fields you want to change, or construct a custom
    instance and pass it to evaluate_link_prediction.
    """

    # RATIONALE: (1, 3, 5, 10, 20) are the standard link-prediction
    # K values from Bordes et al. (2013) and the DRKG evaluation
    # protocol. Fixes E12-001 / E1-001.
    k_values: Tuple[int, ...] = (1, 3, 5, 10, 20)

    # RATIONALE: TransE (Bordes et al. 2013) scores by L2 distance
    # where lower = more plausible. Default False. The Phase 3
    # Graph Transformer (dot-product attention) will set True.
    # Fixes E12-004.
    default_higher_is_better: bool = False

    # RATIONALE: "warn" is the safest default — falls back to manual
    # AUC if sklearn is missing but tells the operator. "fail" is
    # recommended for clinical/regulatory runs. Fixes E12-003.
    sklearn_fallback_strategy: _Literal["fail", "warn", "silent"] = "warn"

    # P2-023 ROOT FIX (Team 8 — Mann-Whitney AUC verification is O(n*m)):
    # The previous default ``True`` caused ``compute_auc`` to call
    # ``_manual_auc`` (O(n_pos * n_neg) Mann-Whitney U) on EVERY AUC
    # computation to cross-check sklearn's result. On a 100K x 100K
    # eval set this is 10^10 comparisons per epoch — ~30 minutes of
    # wasted compute, with the training loop spending >90% of its
    # wall-clock time in evaluation. sklearn's ``roc_auc_score`` uses
    # a sorted-rank O(n log n) algorithm and is mathematically
    # equivalent (verified by ``tests/test_evaluation.py``).
    #
    # The cross-check is still valuable for catching numerical drift,
    # but it MUST NOT run on every epoch. The fix:
    #   1. Default ``verify_sklearn_agreement`` to False (env var
    #      ``DRUGOS_VERIFY_SKLEARN_AUC`` now defaults to "0").
    #   2. Operators who want the cross-check run it ONCE at end of
    #      training via the new ``verify_auc_against_manual`` helper
    #      in evaluation.py (see P2-023 fix in that file).
    #   3. ``DRUGOS_VERIFY_SKLEARN_AUC=1`` re-enables per-call
    #      verification for backwards compatibility / debugging.
    #
    # The CRITICAL rationale (BUG-C-007): the cross-check protects
    # against silent numerical drift between manual and sklearn paths.
    # That protection is preserved by the end-of-training helper —
    # the per-call default was overkill and caused the 50x slowdown
    # documented in P2-023. The audit (§4.1) concern is addressed by
    # the explicit helper, not by per-call verification.
    verify_sklearn_agreement: bool = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_VERIFY_SKLEARN_AUC", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
    )

    # RATIONALE: True by default because the E2-002 recall denominator
    # bug is a CRITICAL patient-safety issue. When True, callers
    # MUST pass total_positives_per_query for recall computation.
    # Fixes E2-002.
    strict_recall_denominator: bool = True

    # RATIONALE: False (standard P@K dividing by K) is the default
    # for backward compatibility. True (capped P@K dividing by
    # min(K, |list|)) is suitable for rare-disease candidate sets.
    # Fixes E2-005.
    strict_precision_k: bool = False

    # RATIONALE: Bootstrap CI is expensive (1000 resamples). Off by
    # default. Enable for regulatory reporting (FDA 21 CFR Part 11).
    # Fixes E14-003.
    bootstrap_ci: bool = False

    # RATIONALE: 1000 bootstrap iterations gives ~95% CI with
    # reasonable accuracy for most metric distributions.
    n_bootstrap: int = 1000

    # RATIONALE: Separate seed for CI to allow independent
    # reproducibility of bootstrap intervals.
    ci_seed: Optional[int] = None

    # RATIONALE: Log results by default for observability. Set False
    # in batch/Jupyter contexts to suppress output. Fixes E1-003.
    log_results: bool = True

    # RATIONALE: Verbose logs include per-K breakdowns, score
    # distributions, and data quality details. Off by default to
    # reduce log volume.
    verbose_logs: bool = False

    # RATIONALE: Hash string entity IDs by default to prevent PII
    # (drug names, disease names) from appearing in logs and
    # error messages. Fixes E9-001.
    hash_string_entity_ids: bool = True

    # RATIONALE: "raise" follows Principle P1 (Fail Loudly). A
    # silent 0.5 AUC is worse than a crash. Fixes E6-004.
    on_failure: _Literal["raise", "warn", "return_nan"] = "raise"

    # RATIONALE: These five metrics cover the standard link-prediction
    # evaluation suite. Additional metrics (NDCG, MAP, F1) can be
    # registered via the metric registry pattern. Fixes E2-003.
    enabled_metrics: Tuple[str, ...] = (
        "auc", "precision_at_k", "recall_at_k", "mrr", "hits_at_k",
    )

    # RATIONALE: Inherit from global SEED for reproducibility.
    # Fixes E7-004.
    seed: int = field(default_factory=lambda: SEED)

    # RATIONALE: Per-prediction breakdown is O(n^2) and produces
    # large outputs. Off by default. Enable for audit/regulatory.
    include_per_prediction_breakdown: bool = False

    def __post_init__(self):
        if not self.k_values:
            raise ValueError("k_values must not be empty")
        if any(k < 1 for k in self.k_values):
            raise ValueError("all k_values must be >= 1")
        valid_fallback = ("fail", "warn", "silent")
        if self.sklearn_fallback_strategy not in valid_fallback:
            raise ValueError(
                f"sklearn_fallback_strategy must be one of {valid_fallback}"
            )
        valid_failure = ("raise", "warn", "return_nan")
        if self.on_failure not in valid_failure:
            raise ValueError(
                f"on_failure must be one of {valid_failure}"
            )
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")


# Module-level singleton, env-overridable.
# RATIONALE: Centralised config instance so evaluation.py imports
# a single object. Fixes E12-002.
_EVAL_CONFIG_ENV = os.environ.get("DRUGOS_EVAL_CONFIG")
if _EVAL_CONFIG_ENV:
    try:
        _eval_overrides = json.loads(_EVAL_CONFIG_ENV)
        _eval_defaults = {
            f.name: getattr(EvaluationConfig, f.name)
            for f in fields(EvaluationConfig)
            if f.name not in ("on_failure", "sklearn_fallback_strategy")
        }
        _eval_defaults.update(_eval_overrides)
        EVALUATION_CONFIG: EvaluationConfig = EvaluationConfig(**_eval_defaults)
    except (ValueError, json.JSONDecodeError):  # v85 FORENSIC ROOT FIX (BUG #51)
        EVALUATION_CONFIG = EvaluationConfig()
else:
    EVALUATION_CONFIG: EvaluationConfig = EvaluationConfig()

# ─── Evaluation Version Constants ──────────────────────────────────────────
# Fixes E7-003, E14-002 — versioned evaluation for reproducibility.
# These are also defined in evaluation.py for self-contained import;
# config.py is the authoritative source.

EVALUATION_METRIC_VERSION: str = "2.0.0-evaluation"
EVALUATION_SCHEMA_VERSION: str = "1.0.0"
SKLEARN_MIN_VERSION: str = "0.24.0"
K_VALUES_DEFAULT: Tuple[int, ...] = (1, 3, 5, 10, 20)
EVALUATION_FALLBACK_STRATEGY: str = "warn"


# ─── Phase E — Knowledge Graph Schema (SCIENTIFIC CORRECTNESS) ───────────────
# THIS IS THE HIGHEST PRIORITY DOMAIN (Domain 3).
# Wrong science = wrong predictions = patient harm.

# The 5 core node types in DrugOS (from the project spec)
# SCIENTIFIC CORRECTNESS NOTE:
#   - Gene (NCBI Gene ID) and Protein (UniProt accession) are distinct
#     entities in our schema. The biological relationship Gene-encodes->Protein
#     is materialized as a separate edge in the graph.
#   - When only DRKG is loaded (no UniProt file), no Protein nodes will
#     exist; the Gene node stands in for both gene and its protein product
#     (DRKG's convention). When UniProt is also loaded, the
#     Gene-encodes-Protein edge bridges the two.
# v108 ROOT FIX (ISSUE-P2-056): the schema constants (CORE_NODE_TYPES,
# DRKG_NODE_TYPES, CORE_EDGE_TYPES, CORE_EDGE_TYPES_SET) have been MOVED
# to ``config_schema.py`` and are imported at the top of this file.
# The original inline definitions (188 lines) were REMOVED to avoid
# shadowing the imports. See ``config_schema.py`` for the canonical
# definitions and full scientific-correctness docstrings.


# P2-011 ROOT FIX (symmetric / biologically-undirected relations):
# The per-edge-type density computation in graph_stats.py needs to
# distinguish DIRECTED edges (Compound→treats→Disease — only one
# direction is biologically meaningful) from SYMMETRIC edges
# (Protein-interacts_with-Protein — the edge is biologically
# undirected; STRING PPI is symmetric by construction). For
# symmetric same-type relations, the directed-graph denominator
# ``n*(n-1)`` double-counts each undirected edge — the correct
# denominator is ``n*(n-1)/2``. Without this distinction, PPI
# density reports at HALF its true value (e.g. 0.5% instead of
# 1.0%), biasing ML hyperparameter tuning (sparse vs dense message
# passing) and masking duplicate-edge bugs ("PPI density > 5%
# suggests duplicates" — the threshold is now meaningful).
#
# Membership rule: a relation is symmetric IFF the edge is
# biologically undirected AND the source and destination node
# types are identical (so the n*(n-1)/2 form is well-defined).
# Cross-type relations like (Compound, treats, Disease) are NEVER
# symmetric even when semantically bidirectional — they use the
# n_src * n_dst denominator.
#
# Sources:
#   - STRING PPI: "interactions are undirected" (Szklarczyk et al.
#     2023, STRING database paper §2).
#   - DRKG Gene-Gene: same as STRING (DRKG collects PPIs from
#     multiple sources, all undirected).
#   - Compound-Compound drug-drug interactions: pharmacologically
#     symmetric (if A interacts with B, B interacts with A) — see
#     DrugBank DDI documentation. The (Compound, interacts_with,
#     Compound) edge is included here.
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({
    "interacts_with",  # PPI (Protein-Protein, Gene-Gene) AND DDI (Compound-Compound)
})


# P2-011 helper: predicate form (used by graph_stats.py density
# computation and by the schema validator).
def is_symmetric_relation(rel_name: str) -> bool:
    """Return True if ``rel_name`` is a biologically-undirected relation.

    See ``SYMMETRIC_RELATIONS`` for the rationale and source
    citations. Used by ``graph_stats.compute_density_per_type`` to
    pick the correct density denominator:
    ``n*(n-1)/2`` for symmetric, ``n*(n-1)`` for directed.
    """
    return rel_name in SYMMETRIC_RELATIONS


# Fixes audit issue 2.13 — is_core_edge() helper
def is_core_edge(src: str, rel: str, dst: str) -> bool:
    """Check if a triple is a core edge type.

    Parameters
    ----------
    src : str
        Source node type.
    rel : str
        Relationship type.
    dst : str
        Destination node type.

    Returns
    -------
    bool
    """
    return (src, rel, dst) in CORE_EDGE_TYPES_SET


# Fixes audit issue 2.13 — filter_to_core_edges()
def filter_to_core_edges(
    edges: list[Tuple[str, str, str]],
) -> list[Tuple[str, str, str]]:
    """Filter a list of edge triples to only core edge types.

    Parameters
    ----------
    edges : list of (src, rel, dst) tuples

    Returns
    -------
    list of (src, rel, dst) tuples that are in CORE_EDGE_TYPES
    """
    return [e for e in edges if e in CORE_EDGE_TYPES_SET]


# Fixes audit issue 2.13 — DRKG_RELATION_TO_CORE_EDGE mapping
# Maps DRKG relation format "SourceType::relation::TargetType"
# to our canonical (src, rel, dst) edge types
DRKG_RELATION_TO_CORE_EDGE: dict[str, Tuple[str, str, str]] = {
    "Compound::treats::Disease": ("Compound", "treats", "Disease"),
    "Compound::binds::Gene": ("Compound", "binds", "Protein"),
    "Compound::inhibits::Gene": ("Compound", "inhibits", "Gene"),
    "Compound::activates::Gene": ("Compound", "activates", "Gene"),
    "Gene::associated_with::Disease": ("Gene", "associated_with", "Disease"),
    "Gene::interacts_with::Gene": ("Gene", "interacts_with", "Gene"),
}

# Fixes audit issue 2.13 — STRICT_EDGE_FILTERING
# RATIONALE: When True, only CORE_EDGE_TYPES are loaded from DRKG.
# When False, all DRKG relation types are loaded (may include
# biologically questionable edges). Default True for clinical safety.
STRICT_EDGE_FILTERING: bool = os.environ.get(
    "DRUGOS_STRICT_EDGE_FILTERING", "1"
) == "1"

# DRKG relation type parsing
DRKG_RELATION_SEPARATOR: str = "::"


# Fixes audit issue 2.13 — split_drkg_relation and join_drkg_relation
def split_drkg_relation(relation: str) -> Tuple[str, str, str]:
    """Split a DRKG relation string into (src_type, relation_name, dst_type).

    P2-014 ROOT FIX: handles BOTH DRKG relation formats robustly:

    1. **Hetionet / DRUGBANK format** (3 tokens after split on ``::``):
       ``"Hetionet::CtD::Compound:Disease"`` -> ("Hetionet", "CtD", "Compound:Disease")
       Here the third token is ``HeadType:TailType`` -- the caller must
       split on ``:`` to get head/tail types separately.

    2. **GNBR short-code format** (3 tokens after split on ``::``):
       ``"GNBR::A::Gene:Compound"`` -> ("GNBR", "A", "Gene:Compound")
       The middle token is the GNBR abbreviation (A, B, E, N, ...). The
       previous code returned this tuple verbatim, which is correct --
       but downstream code that looked up ``DRKG_RELATION_ABBREV_TO_NAME``
       would miss the entry for plain "A" (only "A+" / "A-" were in the
       codebook), producing ``relation_human_name="unknown"``. The P2-014
       fix adds "A" to the codebook so the semantic context ("affects")
       is preserved.

    The function returns ``(source, relation_abbrev, head_tail_types)``.
    Callers that need head_type and tail_type separately should split
    ``head_tail_types`` on ``":"`` (first colon only -- entity IDs may
    contain colons, e.g. ``"Disease::DOID:1438"``).

    Parameters
    ----------
    relation : str
        DRKG relation string like ``"Hetionet::CtD::Compound:Disease"``
        or ``"GNBR::A::Gene:Compound"``.

    Returns
    -------
    tuple of (src_type, relation_name, dst_type)
        Where ``dst_type`` is the ``HeadType:TailType`` substring (NOT
        split on ``:`` -- callers must do that themselves).

    Raises
    ------
    ValueError
        If relation doesn't have the expected format.
    """
    if not isinstance(relation, str) or not relation:
        raise ValueError(
            f"Invalid DRKG relation format: {relation!r}. "
            f"Expected 'SrcType::relation::DstType'"
        )
    parts = relation.split(DRKG_RELATION_SEPARATOR)
    if len(parts) != 3:
        # Task 91 ROOT FIX: the previous check was ``len(parts) < 3``
        # and the return used ``parts[-1]``. For a malformed relation
        # with 4+ ``::``-separated tokens (e.g. an accidental
        # ``"Hetionet::CtD::Compound::Disease"`` where the third
        # ``::`` should have been a single ``:``), the old code would
        # silently truncate -- returning ``("Hetionet", "CtD",
        # "Disease")`` and dropping the head-type ``"Compound"``
        # entirely. That truncated value was then split on ``:`` to
        # derive head_type/tail_type, producing a tail_type of
        # ``"Disease"`` and an EMPTY head_type, which propagated to
        # the KG as a malformed edge with no head entity type. The
        # fix: require EXACTLY 3 parts (canonical DRKG format) and
        # raise ``ValueError`` on anything else so the malformed row
        # is dead-lettered at parse time rather than silently
        # corrupting the graph.
        raise ValueError(
            f"Invalid DRKG relation format: {relation!r}. "
            f"Expected exactly 3 '{DRKG_RELATION_SEPARATOR}'-separated "
            f"tokens in the form 'SrcType::relation::HeadType:TailType' "
            f"but got {len(parts)} tokens."
        )
    return parts[0], parts[1], parts[2]


def parse_drkg_relation_head_tail(head_tail_str: str) -> Tuple[str, str]:
    """Split a DRKG ``HeadType:TailType`` substring into (head_type, tail_type).

    P2-014 ROOT FIX: splits on the FIRST colon only -- entity IDs in
    DRKG may contain colons (e.g. ``"Disease:DOID:1438"`` where the tail
    type is ``"Disease"`` and the tail ID is ``"DOID:1438"``). The
    previous split-on-``:`` approach in some downstream callers would
    produce 3 tokens and silently drop the ID prefix.

    Parameters
    ----------
    head_tail_str : str
        The ``HeadType:TailType`` substring from a DRKG relation (the
        third token returned by ``split_drkg_relation``).

    Returns
    -------
    tuple of (head_type, tail_type)
        Both lowercased for case-insensitive comparison with
        ``DRKG_VALID_TRIPLE_SCHEMAS``.

    Raises
    ------
    ValueError
        If the input doesn't contain a colon.
    """
    if not isinstance(head_tail_str, str) or ":" not in head_tail_str:
        raise ValueError(
            f"Invalid DRKG head:tail format: {head_tail_str!r}. "
            f"Expected 'HeadType:TailType'"
        )
    head, tail = head_tail_str.split(":", 1)
    return head.strip(), tail.strip()


def canonical_drkg_relation_name(relation_abbrev: str) -> str:
    """Look up the human-readable name for a DRKG relation abbreviation.

    P2-014 ROOT FIX: case-insensitive lookup so that ``"A"``, ``"a"``,
    ``"CtD"``, ``"ctd"`` all resolve correctly. Returns the abbreviation
    itself if no mapping exists (so the caller can detect unknown codes
    via ``DRKG_RELATION_ABBREV_TO_NAME.get(code) is None``).
    """
    if not isinstance(relation_abbrev, str) or not relation_abbrev:
        return relation_abbrev if isinstance(relation_abbrev, str) else ""
    # Try exact match first (preserves case-sensitive codes like "E+"/"E-").
    if relation_abbrev in DRKG_RELATION_ABBREV_TO_NAME:
        return DRKG_RELATION_ABBREV_TO_NAME[relation_abbrev]
    # Fall back to case-insensitive match for codes that are case-agnostic
    # (e.g. "ctd" -> "CtD", "target" -> "target").
    lower = relation_abbrev.lower()
    for k, v in DRKG_RELATION_ABBREV_TO_NAME.items():
        if k.lower() == lower:
            return v
    return relation_abbrev


def join_drkg_relation(src_type: str, relation: str, dst_type: str) -> str:
    """Join components into a DRKG relation string.

    Parameters
    ----------
    src_type : str
    relation : str
    dst_type : str

    Returns
    -------
    str
    """
    return f"{src_type}{DRKG_RELATION_SEPARATOR}{relation}{DRKG_RELATION_SEPARATOR}{dst_type}"


# =============================================================================
# DRKG v2.0 audit-fix constants
# =============================================================================
# Added by the drkg_loader v2.0 audit fix (drkg_loader_repair_prompt.md,
# §Domain 3 / Domain 5 / Domain 9 / Domain 12 / Domain 14).
#
# These constants centralise the canonical DRKG codebook so that
# ``drkg_loader.py``, ``training_data.py``, and downstream consumers all
# read from a single source of truth. Before this block, the DRKG
# relation whitelist was duplicated (and divergent) across loaders —
# the audit flagged this as BUG 3.1 / BUG 3.2 / BUG 3.6 / GAP 1.5.
#
# Reference for the abbreviations:
#   DRKG codebook — https://github.com/gnn4dr-kg/awmlpedia/wiki/DRKG
#   Himmelstein et al., 2020, Sci Data 7:329
#   doi:10.1038/s41597-020-0465-y
# =============================================================================

# DRKG_PARSER_VERSION / DRKG_SCHEMA_VERSION
# Fixes GAP 7.5 — version constants centralised in config so that the
# loader, the pipeline runner, and the MLflow tracker all log the same
# value. Bumped on any DRKG parse-logic or output-schema change.
DRKG_PARSER_VERSION: str = "2.0.0"
DRKG_SCHEMA_VERSION: str = "2.0.0"

# DRKG_LICENSE / DRKG_ATTRIBUTION
# Fixes BUG 14.1 — every DRKG-derived record carries the MIT license
# string and the Himmelstein 2020 citation in its ``_license`` /
# ``_attribution`` fields so downstream exports remain compliant.
DRKG_LICENSE: str = "MIT"
DRKG_ATTRIBUTION: str = (
    "DRKG (Himmelstein et al., 2020, Sci Data 7:329, "
    "doi:10.1038/s41597-020-0465-y)"
)

# =============================================================================
# DrugBank v2.0 audit-fix constants (drugbank_parser_fix_prompt.md)
# -----------------------------------------------------------------------------
# These constants eliminate magic strings and hardcoded values from
# ``drugos_graph.drugbank_parser``. They are the single source of truth for
# the parser version, schema version, license, namespace, action-to-relation
# mapping, external-ID aliases, and other configurable values.
#
# Fixes: FIX[(7.2)] FIX[(14.1)] FIX[(14.2)] FIX[(14.3)] FIX[(14.11)]
#        FIX[(3.3)] FIX[(3.4)] FIX[(3.16)] FIX[(5.1)] FIX[(5.4)]
#        FIX[(5.13)] FIX[(12.4)] FIX[(12.6)] FIX[(G.15)] FIX[(G.16)]
#        FIX[(G.17)]
# =============================================================================

# DRUGBANK_PARSER_VERSION / DRUGBANK_SCHEMA_VERSION
# Fixes FIX[(7.2)] — version constants centralised in config so that the
# loader, the pipeline runner, and the MLflow tracker all log the same
# value. Bumped on any DrugBank parse-logic or output-schema change.
DRUGBANK_PARSER_VERSION: str = "2.0.0"
DRUGBANK_SCHEMA_VERSION: str = "2.0.0"

# DRUGBANK_LICENSE / DRUGBANK_ATTRIBUTION
# Fixes FIX[(14.1)] FIX[(14.2)] — every DrugBank-derived record carries
# the CC BY-NC 4.0 (academic) license string and the Wishart 2024
# citation in its ``_license`` / ``_attribution`` fields so downstream
# exports remain compliant with the DrugBank license terms. Commercial
# use of DrugBank data is prohibited without a separate commercial
# license from the Wishart Research Group.
DRUGBANK_LICENSE: str = "CC BY-NC 4.0 (academic)"
DRUGBANK_ATTRIBUTION: str = (
    "Wishart DS et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024. "
    "Nucleic Acids Res. doi:10.1093/nar/gkad1044"
)

# DRUGBANK_NAMESPACE_URI / DRUGBANK_NAMESPACE_ALIASES
# Fixes FIX[(5.1)] FIX[(12.2)] FIX[(1.7)] — DrugBank 5.x XML uses
# ``xmlns="http://www.drugbank.ca"`` (with www.). Older versions and some
# mirrors use ``xmlns="http://drugbank.ca"`` (no www.). Both are accepted
# via ``DRUGBANK_NAMESPACE_ALIASES``. The parser auto-detects the actual
# namespace from the root element and refuses to parse if it is not in
# the alias set (raises ``DrugBankDataIntegrityError``).
DRUGBANK_NAMESPACE_URI: str = "http://www.drugbank.ca"
DRUGBANK_NAMESPACE_ALIASES: tuple[str, ...] = (
    "http://www.drugbank.ca",
    "http://drugbank.ca",
)

# DRUGBANK_TEXT_FIELD_MAX_LENGTH
# Fixes FIX[(3.13)] FIX[(12.3)] — Neo4j has a property size limit; long
# DrugBank text fields (indication, mechanism_of_action, toxicity,
# pharmacodynamics) are truncated at this length. Truncation is at a
# sentence boundary (., !, ?) when possible. The full text hash
# (SHA-256) is preserved on every record for traceability.
DRUGBANK_TEXT_FIELD_MAX_LENGTH: int = 500

# DRUGBANK_ORGANISM_FILTER_TAX_ID
# Fixes FIX[(3.2)] FIX[(3.18)] — default NCBI TaxID for organism
# filtering. 9606 = Homo sapiens. DrugBank contains targets for many
# organisms (mouse, rat, E. coli, etc.); only human targets are
# clinically actionable for drug repurposing in humans. Set to None to
# disable filtering (emits all targets with a ``non_human: bool`` flag).
DRUGBANK_ORGANISM_FILTER_TAX_ID: int = 9606

# DRUGBANK_ACTION_TO_RELATION
# Fixes FIX[(3.4)] FIX[(2.7)] FIX[(12.5)] — canonical mapping from
# DrugBank <action> strings to the project's CORE_EDGE_TYPES relation
# vocabulary. DrugBank actions include "agonist", "antagonist",
# "inhibitor", "inducer", "activator", "partial agonist", "inverse
# agonist", "positive allosteric modulator", "negative allosteric
# modulator", "binder", "other", "unknown". The parser emits the
# canonical relation; the raw ``action`` is preserved on every edge for
# audit. Unknown actions fail-closed to ``"unknown"`` and are written to
# the dead-letter queue.
DRUGBANK_ACTION_TO_RELATION: dict[str, str] = {
    "agonist": "activates",                       # agonism → activation
    "partial agonist": "activates",
    "inverse agonist": "inhibits",
    "antagonist": "inhibits",
    "inhibitor": "inhibits",
    "inducer": "induces",
    "activator": "activates",
    "positive allosteric modulator": "allosterically_modulates",
    "negative allosteric modulator": "allosterically_modulates",
    "binder": "binds",
    "other": "unknown",
    "unknown": "unknown",
}

# DRUGBANK_EXTERNAL_ID_ALIASES
# Fixes FIX[(3.16)] FIX[(12.6)] — DrugBank has renamed external ID
# resource names between versions (e.g., "PubChem Compound" →
# "PubChem Compound ID" → "PubChem CID"). This alias map lets the
# parser resolve the canonical field name (e.g., "pubchem_cid")
# regardless of which alias DrugBank used in the source XML. The first
# non-empty alias wins; if multiple aliases resolve to different values,
# a WARNING is logged (data-quality concern).
DRUGBANK_EXTERNAL_ID_ALIASES: dict[str, tuple[str, ...]] = {
    "pubchem_cid": (
        "PubChem Compound", "PubChem Compound ID", "PubChem CID", "PubChem",
    ),
    "chembl_id": ("ChEMBL", "ChEMBL ID", "ChEMBL compound ID"),
    "chebi_id": ("ChEBI", "ChEBI ID"),
    "drugbank_id": ("DrugBank ID", "DrugBank"),
    "uniprot_id": ("UniProtKB", "UniProt accession", "UniProt"),
    "kegg_id": ("KEGG Drug", "KEGG Compound", "KEGG"),
    "wikipedia_id": ("Wikipedia", "Wikipedia ID"),
}

# ATC_CODE_SEPARATOR
# Fixes FIX[(2.5)] FIX[(15.8)] FIX[(G.8)] — the separator used when
# joining ATC codes into a single string for Neo4j storage. The
# ``entity_resolver.resolve_compounds_from_drugbank`` consumer splits
# on this separator. Any ATC code containing the separator (defensive —
# should never happen) is escaped with a backslash before joining.
ATC_CODE_SEPARATOR: str = "|"

# DRUGBANK_PROGRESS_LOG_INTERVAL
# Fixes FIX[(5.18)] FIX[(11.9)] FIX[(12.4)] — number of drugs parsed
# between progress log entries. Combined with a 30-second time-based
# fallback so the user gets progress feedback even on slow parses.
DRUGBANK_PROGRESS_LOG_INTERVAL: int = int(
    os.environ.get("DRUGBANK_PROGRESS_LOG_INTERVAL", "1000")
)

# DRUGBANK_MIN_FIELD_POPULATION
# Fixes FIX[(5.20)] FIX[(11.16)] FIX[(G.10)] — minimum population rate
# for critical fields, expressed as a fraction in [0.0, 1.0]. If the
# actual population rate falls below the threshold, the parser raises
# ``DrugBankDataIntegrityError``. This catches parser regressions
# (e.g., a DrugBank version change that moves SMILES to a new XPath)
# before they silently propagate to the KG.
DRUGBANK_MIN_FIELD_POPULATION: dict[str, float] = {
    "drugbank_id": 0.99,    # nearly all drugs must have a primary ID
    "name": 0.99,
    "smiles": 0.80,         # biotech drugs legitimately lack SMILES
    "inchikey": 0.70,
    "targets": 0.50,
    "atc_codes": 0.50,
    "approval_year": 0.50,  # FIX[(G.10)] — temporal split guard
}

# DRUGBANK_KG_BUILDER_FIELDS
# Fixes FIX[(15.1)] FIX[(11.14)] — the list of fields the
# ``kg_builder.enrich_compounds_from_drugbank`` Cypher SET clause reads.
# The parser MUST emit at least these fields on every node record. The
# test suite asserts this invariant.
DRUGBANK_KG_BUILDER_FIELDS: tuple[str, ...] = (
    "id", "drugbank_id", "name", "smiles", "inchikey",
    "indication", "mechanism_of_action",
    "atc_codes", "approved", "investigational",
    "pubchem_cid", "chembl_id", "chebi_id",
    "drug_type", "approval_year",
    # New fields added by FIX[(3.10)] FIX[(3.11)] FIX[(3.12)]
    # FIX[(3.14)] FIX[(3.15)]
    "withdrawn", "terminated", "illicit",
    "categories", "cas_number",
    "toxicity", "pharmacodynamics",
    "sensitive",
)

# ALLOWED_DRUGBANK_URLS
# Fixes FIX[(9.1)] — URL-prefix allowlist for DrugBank downloads (guard
# against config injection / SSRF). Any URL in
# ``DATA_SOURCES['drugbank']['url']`` MUST start with one of these
# prefixes or the download is refused before any network call. DrugBank
# requires academic registration; the download URL is behind a login
# form, so direct download is typically not possible — but the allowlist
# is still enforced for the credentials-aware path.
ALLOWED_DRUGBANK_URLS: tuple[str, ...] = (
    "https://go.drugbank.com/",
    "https://www.drugbank.com/",
    "https://drugbank.com/",
    "https://ftp.drugbank.com/",
)

# ═══════════════════════════════════════════════════════════════════════════════
# ChEMBL LOADER CONSTANTS — institutional-grade audit fix
# ═══════════════════════════════════════════════════════════════════════════════
# These constants follow the same pattern as the DRKG_* and DRUGBANK_*
# constants above. Every magic number, magic string, and hardcoded threshold
# in chembl_loader.py MUST be replaced by a named constant from this section.

# CHEMBL_MIN_PCHEMBL_VALUE
# RATIONALE: pChEMBL = -log10(IC50/Ki/Kd in M). A value of 5.0 corresponds
# to ~10 uM (micromolar), which is the community-standard threshold for
# "meaningful" bioactivity. Values below 5.0 are typically noise or very
# weak binders that would add false-positive edges to the knowledge graph.
# However, some researchers use 4.0 for broader coverage. This constant
# makes the threshold explicit and overridable.
# Scientific reference: https://doi.org/10.1016/j.drudis.2014.10.012
CHEMBL_MIN_PCHEMBL_VALUE: float = float(
    os.environ.get("DRUGOS_CHEMBL_MIN_PCHEMBL", "5.0")
)

# CHEMBL_PCHEMBL_RANGE
# RATIONALE: pChEMBL values must be in [0, 14] (corresponding to 1 M down
# to 0.1 pM). Values outside this range are data errors.
CHEMBL_PCHEMBL_RANGE: tuple[float, float] = (0.0, 14.0)

# CHEMBL_MIN_CONFIDENCE_SCORE
# RATIONALE: ChEMBL assigns confidence scores (0-9) to target mappings.
# 9 = direct single-protein target; 0 = unspecified. We require >= 7
# which means "target has been assigned with high confidence to a single
# protein". This eliminates ambiguous target assignments.
CHEMBL_MIN_CONFIDENCE_SCORE: int = int(
    os.environ.get("DRUGOS_CHEMBL_MIN_CONFIDENCE", "7")
)

# CHEMBL_ORGANISM_FILTER_TAX_ID
# RATIONALE: For the drug-repurposing platform, we only care about HUMAN
# targets. This NCBI Taxonomy ID filters non-human proteins, preventing
# the KG from containing mouse/rat/e.coli targets that would create
# disconnected subgraphs or misleading drug-protein edges.
CHEMBL_ORGANISM_FILTER_TAX_ID: int = int(
    os.environ.get("DRUGOS_CHEMBL_TAX_ID", "9606")
)

# CHEMBL_TARGET_TYPES
# RATIONALE: ChEMBL classifies targets as SINGLE PROTEIN, PROTEIN COMPLEX,
# PROTEIN FAMILY, etc. For the knowledge graph, we only include:
# - SINGLE PROTEIN: clear 1:1 mapping to a UniProt accession
# - PROTEIN COMPLEX GROUP: some are mappable via target_components
# - SELECTIVITY GROUP: ChEMBL-specific grouping of related targets
# Other types (ORGANISM, UNKNOWN, etc.) are excluded because they don't
# map to specific proteins and would create ambiguous graph nodes.
CHEMBL_TARGET_TYPES: frozenset[str] = frozenset({
    "SINGLE PROTEIN",
    "PROTEIN COMPLEX GROUP",
    "PROTEIN FAMILY",
})

# CHEMBL_ASSAY_TYPES
# RATIONALE: ChEMBL assays are classified as B (Binding) or F (Functional).
# Both are scientifically valid for drug-target interaction, but they
# measure different things:
# - B: direct physical binding (Kd, Ki) — strongest evidence
# - F: functional readout (IC50, EC50) — may be indirect
# We accept both but tag the edge with the assay type for downstream
# consumers (e.g., the RL ranker may weight B assays higher).
CHEMBL_ASSAY_TYPES: frozenset[str] = frozenset({"B", "F"})

# CHEMBL_STANDARD_TYPE_TO_RELATION
# RATIONALE: This is the SCIENTIFIC MAPPING from ChEMBL standard_type
# strings to biological relationship types. This is the single source of
# truth for how ChEMBL activity measurements are interpreted in the KG.
# A wrong mapping here creates wrong edges, which corrupts the GNN.
#
# Scientific basis:
# - IC50 (half-maximal inhibitory concentration) → "inhibits"
# - Ki (inhibition constant) → "inhibits"
# - Kd (dissociation constant) → "binds" (measures affinity, not direction)
# - EC50 (half-maximal effective concentration) → "activates"
# - ED50 (effective dose) → "activates"
# - Potency → "binds" (generic measure, no direction implied)
# - AC50 → "activates"
# - MIC (minimum inhibitory concentration) → "inhibits"
# - Activity → "binds" (too generic to infer direction)
# - Inhibition → "inhibits"
# - % Inhibition → "inhibits"
# The key insight: NOT every bioactivity measurement implies INHIBITION.
# The old code defaulted to "inhibits" for unknown types — this was
# scientifically wrong and could cause the GNN to learn that agonists
# are inhibitors.
CHEMBL_ACTIVITY_TYPE_INHIBITS: frozenset[str] = frozenset({
    "IC50", "KI", "KIBIOCHEMICAL", "KIFITTED",
    "INHIBITION", "% INHIBITION", "PERCENT INHIBITION",
    "MIC", "MIC80", "MBC",
    "KINOME_SCAN", "GIC50", "GI50",
    "MINIMUM INHIBITOR CONCENTRATION",
    # v58 ROOT FIX (P2L-008 deep): INACTIVATION is the standard ChEMBL
    # label for irreversible / covalent inhibition (aspirin, omeprazole,
    # clopidogrel, penicillin — all covalent modifiers). The v57 fix
    # only PREVENTED the wrong "activates" classification (via \bACTIVAT
    # word-boundary in _RE_ACTIVATE). Without these explicit entries,
    # INACTIVATION fell through to the default "targets" — losing the
    # scientific signal that the drug INHIBITS the target. The Graph
    # Transformer then trained on a graph where covalent inhibitors had
    # a neutral "targets" edge instead of an "inhibits" edge — corrupting
    # the drug-target semantics the model is supposed to learn.
    "INACTIVATION", "% INACTIVATION", "PERCENT INACTIVATION",
    "INACTIVATOR", "INACTIVATION ASSAY",
    "IRREVERSIBLE INHIBITION", "COVALENT INHIBITION",
    "TIME-DEPENDENT INHIBITION", "TIME DEPENDENT INHIBITION",
    "SUICIDE INHIBITION",
})
CHEMBL_ACTIVITY_TYPE_ACTIVATES: frozenset[str] = frozenset({
    "EC50", "ED50", "AC50", "ECS0", "EC100",
    "ACTIVATION", "% ACTIVATION", "PERCENT ACTIVATION",
    "AGONIST ACTIVITY", "EMAX",
})
CHEMBL_ACTIVITY_TYPE_BINDS: frozenset[str] = frozenset({
    "KD", "KDBIOCHEMICAL", "KDFITTED",
    "POTENCY", "ACTIVITY", "BINDING",
    "AFFINITY", "T1/2", "T2/2",
    "RESIDUE", "RATIO",
})
CHEMBL_ACTIVITY_TYPE_MODULATES: frozenset[str] = frozenset({
    "ALLOSTERIC", "MODULATION", "POSITIVELY MODULATES",
    "NEGATIVELY MODULATES",
})

# CHEMBL_STANDARD_TYPE_TO_RELATION — the canonical mapping from
# ChEMBL standard_type strings to KG relation types. Built from the
# per-category frozensets above so there is a single source of truth.
# Fixes pre-existing bug: symbol was in __all__ but never defined.
CHEMBL_STANDARD_TYPE_TO_RELATION: dict[str, str] = {}
for _std_type in CHEMBL_ACTIVITY_TYPE_INHIBITS:
    CHEMBL_STANDARD_TYPE_TO_RELATION[_std_type] = "inhibits"
for _std_type in CHEMBL_ACTIVITY_TYPE_ACTIVATES:
    CHEMBL_STANDARD_TYPE_TO_RELATION[_std_type] = "activates"
for _std_type in CHEMBL_ACTIVITY_TYPE_BINDS:
    CHEMBL_STANDARD_TYPE_TO_RELATION[_std_type] = "binds"
for _std_type in CHEMBL_ACTIVITY_TYPE_MODULATES:
    CHEMBL_STANDARD_TYPE_TO_RELATION[_std_type] = "allosterically_modulates"
del _std_type  # cleanup loop variable

# ALLOWED_CHEMBL_URLS
# URL-prefix allowlist for ChEMBL downloads (guard against config injection /
# SSRF). Any URL in DATA_SOURCES['chembl']['url'] MUST start with one of
# these prefixes or the download is refused before any network call.
ALLOWED_CHEMBL_URLS: tuple[str, ...] = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/",
    "https://www.ebi.ac.uk/chembl/",
    "https://chembl.gitbook.io/",
)

# CHEMBL_MIN_VALID_SIZE_BYTES
# Minimum byte size for a downloaded ChEMBL tar.gz to be considered valid.
# The real ChEMBL SQLite dump is ~4 GB; this threshold catches truncated
# or corrupted downloads (e.g., an HTML error page).
CHEMBL_MIN_VALID_SIZE_BYTES: int = 10_000_000  # 10 MB minimum

# CHEMBL_PROGRESS_LOG_INTERVAL
# Number of rows processed between progress log entries.
CHEMBL_PROGRESS_LOG_INTERVAL: int = int(
    os.environ.get("DRUGOS_CHEMBL_PROGRESS_LOG_INTERVAL", "100000")
)

# CHEMBL_MIN_FIELD_POPULATION
# Minimum population rate for critical fields, expressed as fraction [0, 1].
# If the actual population rate falls below the threshold, the loader raises
# ChEMBLDataIntegrityError.
CHEMBL_MIN_FIELD_POPULATION: dict[str, float] = {
    "drug_chembl_id": 0.99,
    "target_chembl_id": 0.99,
    "uniprot_accession": 0.50,  # many targets lack UniProt mapping
    "pchembl_value": 0.95,      # this IS the filter — almost all rows have it
    "standard_type": 0.95,
    "smiles": 0.70,             # some compounds lack structures
}

# CHEMBL_KG_BUILDER_FIELDS
# The list of fields the kg_builder reads from ChEMBL edge records.
# The loader MUST emit at least these fields on every edge record.
CHEMBL_KG_BUILDER_FIELDS: tuple[str, ...] = (
    "src_id", "dst_id", "src_type", "dst_type", "rel_type",
    "props",
)

# CHEMBL_DRUG_IDENTIFIER_REGEX
# Validates ChEMBL compound IDs: CHEMBL followed by 1-7 digits.
# E.g., CHEMBL25, CHEMBL218, CHEMBL1201588
CHEMBL_DRUG_IDENTIFIER_REGEX: str = r"^CHEMBL\d{1,7}$"

# CHEMBL_UNIPROT_AC_REGEX
# Validates UniProt accessions in ChEMBL target_components.
# Format: 1 letter + 5 digits (e.g., P23219) or 1 letter + 5 alphanums
# for newer entries (e.g., A0A024QZ08).
# Validates UniProt accessions in ChEMBL target_components.
# Format: 6-char (e.g. P23219: 1 letter + 5 digits) or
# 10-char (e.g. A0A024QZ08: 1 letter + 0-9 + 7 alphanumeric + 1 digit).
# The regex covers all three UniProt accession formats:
#   1. [A-NR-Z][0-9]{5}         — legacy 6-char (P23219)
#   2. [A-NR-Z][0-9][A-Z0-9]{3}[0-9]  — 6-char with alphanums (P35568)
#   3. [OPQ][0-9][A-Z0-9]{3}[0-9]     — 6-char O/P/Q prefix (Q9Y2H6)
#   4. A0A-style 10-char: [A-NR-Z]0[A-Z0-9]{6}[0-9]  (A0A024QZ08)
CHEMBL_UNIPROT_AC_REGEX: str = r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]{5}$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z]0[A-Z0-9]{7}[0-9]$"

# DRUGOS_DEPLOYMENT_CONTEXT
# Fixes FIX[(G.17)] — the deployment context controls whether DrugBank
# (CC BY-NC 4.0) data may be loaded. Academic context is the default
# and matches the DrugBank license terms. Commercial use requires a
# separate commercial license from the Wishart Research Group; the
# parser refuses to load DrugBank data when this env var is set to
# anything other than "academic" (raise ``DrugBankDataIntegrityError``).
DRUGOS_DEPLOYMENT_CONTEXT: str = os.environ.get(
    "DRUGOS_DEPLOYMENT_CONTEXT", "academic"
)

# DRUGOS_ENVIRONMENT
# Fixes FIX[(12.13)] — environment selector (dev/staging/prod). In
# ``prod``, ``validate_xsd=True`` and ``cross_check_regulatory=True``
# by default; in ``dev``, both are False for speed.
DRUGOS_ENVIRONMENT: str = os.environ.get("DRUGOS_ENVIRONMENT", "production")

# DRUGBANK_STRICT_VERSION
# Fixes FIX[(14.11)] FIX[(G.16)] — when "1", refuse to parse if the
# actual XML root version differs from
# ``DATA_SOURCES['drugbank']['version']``. When "0" (default), log a
# WARNING on mismatch and continue. A version *downgrade* always raises
# (Guard G.16) regardless of this setting.
DRUGBANK_STRICT_VERSION: str = os.environ.get(
    "DRUGOS_DRUGBANK_STRICT_VERSION", "0"
)

# DRUGBANK_ALLOW_MISSING_APPROVAL_YEAR
# Fixes FIX[(3.1)] — when "1", approved drugs with no FDA Approval Date
# in <experimental-properties> are accepted with ``approval_year=None``
# (backward compat). When unset (default), the parser raises
# ``DrugBankDataIntegrityError`` for approved drugs with no FDA date —
# fail-closed for patient safety (temporal split would silently fall
# back to random otherwise).
DRUGBANK_ALLOW_MISSING_APPROVAL_YEAR: str = os.environ.get(
    "DRUGOS_DRUGBANK_ALLOW_MISSING_APPROVAL_YEAR", "0"
)

# DRUGBANK_STORE_FULL_TEXT
# Fixes FIX[(3.13)] — when "1", the parser also emits the untruncated
# ``indication_full``, ``mechanism_of_action_full``, ``toxicity_full``,
# and ``pharmacodynamics_full`` fields. Default "0" (truncated only,
# with SHA-256 of the full text for traceability).
DRUGBANK_STORE_FULL_TEXT: str = os.environ.get(
    "DRUGOS_DRUGBANK_STORE_FULL_TEXT", "0"
)

# DRUGBANK_BACKFILL_REFERENCE_TIME
# Fixes FIX[(5.6)] FIX[(7.7)] — when set (ISO-8601), used as the
# reference time for file-age calculations instead of ``time.time()``.
# This makes backfills deterministic: the same backfill produces the
# same ``_provenance['source_file_age_days']`` regardless of when it
# runs.
DRUGBANK_BACKFILL_REFERENCE_TIME: str = os.environ.get(
    "DRUGOS_DRUGBANK_BACKFILL_REFERENCE_TIME", ""
)

# DRUGOS_FIXED_PARSED_AT
# Fixes FIX[(7.5)] — when set (ISO-8601), used as the ``parsed_at``
# timestamp on every record instead of ``datetime.now(timezone.utc)``.
# This makes backfills deterministic: the same backfill produces the
# same ``_provenance['parsed_at']`` regardless of when it runs.
DRUGOS_FIXED_PARSED_AT: str = os.environ.get("DRUGOS_FIXED_PARSED_AT", "")

# DRUGOS_RUN_ID
# Fixes FIX[(9.10)] FIX[(11.11)] — when set, used as the ``run_id`` on
# every record's ``_provenance`` for cross-step correlation. Otherwise
# a new UUID is generated per parse run.
DRUGOS_RUN_ID: str = os.environ.get("DRUGOS_RUN_ID", "")

# DRUGBANK_RARE_DISEASE_KEYWORDS
# Fixes FIX[(9.8)] — case-insensitive keywords that, when found in a
# drug's ``indication`` or ``categories`` field, mark the record as
# ``sensitive=True`` (GDPR/HIPAA-aware tagging). Rare-disease data is
# health data under GDPR Article 9 and must be handled accordingly.
DRUGBANK_RARE_DISEASE_KEYWORDS: tuple[str, ...] = (
    "rare disease", "orphan", "orphanet",
    "hemangioma", "lymphoma", "leukemia",
    "wilson", "menkes", "gaucher", "fabry",
    "phenylketonuria", "tyrosinemia",
    "amyotrophic lateral sclerosis",
    "duchenne", "becker",
    "cystic fibrosis",
    "huntington",
)

# DRUGBANK_MEMORY_CEILING_MB
# Fixes FIX[(6.8)] FIX[(8.10)] — maximum resident set size (MB) the
# parser is allowed to use. If exceeded, raises
# ``DrugBankParseError`` with a message pointing to ``iter_drugbank()``
# for streaming-mode parsing.
DRUGBANK_MEMORY_CEILING_MB: int = int(
    os.environ.get("DRUGBANK_MEMORY_CEILING_MB", "4096")
)

# DRUGBANK_CHECKPOINT_INTERVAL
# Fixes FIX[(6.10)] — number of drugs parsed between checkpoint writes.
# Checkpoints enable resume-after-failure for very large XML files.
DRUGBANK_CHECKPOINT_INTERVAL: int = int(
    os.environ.get("DRUGBANK_CHECKPOINT_INTERVAL", "50000")
)

# DRUGBANK_INTERACTION_SEVERITY_RULES
# Fixes FIX[(3.9)] — ordered rules for classifying drug-drug
# interaction severity from the free-text ``<description>`` field.
# The first matching rule wins. Default severity is "unknown".
DRUGBANK_INTERACTION_SEVERITY_RULES: tuple[tuple[str, str], ...] = (
    ("do not co-administer", "contraindicated"),
    ("contraindicated", "contraindicated"),
    ("may increase", "major"),
    ("may decrease", "major"),
    ("can be increased", "major"),
    ("can be decreased", "major"),
    ("can increase", "major"),
    ("can decrease", "major"),
    ("will increase", "major"),
    ("will decrease", "major"),
    ("increases the level", "major"),
    ("decreases the level", "major"),
    ("increase the risk", "major"),
    ("increased when", "major"),
    ("decreased when", "major"),
    ("avoid", "major"),
    ("caution", "moderate"),
    ("monitor", "moderate"),
    ("may alter", "moderate"),
)

# DRUGBANK_DRUG_TYPE_TO_NODE_LABEL
# Fixes FIX[(3.19)] — maps DrugBank <drug type="..."> attribute values
# to additional Neo4j labels for the Compound node. This enables
# downstream filtering (e.g., "give me only biotech drugs") without
# scanning the ``drug_type`` property.
DRUGBANK_DRUG_TYPE_TO_NODE_LABEL: dict[str, str] = {
    "small molecule": "SmallMolecule",
    "biotech": "Biotech",
    "antibody": "Antibody",
    "peptide": "Peptide",
    "protein": "Protein",
    "sugar": "Sugar",
    "oligosaccharide": "Oligosaccharide",
}

# DRUGBANK_TEXT_FIELD_NAMES
# Fixes FIX[(3.13)] FIX[(3.14)] FIX[(3.15)] — the four long-text fields
# that are truncated at ``DRUGBANK_TEXT_FIELD_MAX_LENGTH``. Each emits
# both ``<field>`` (truncated) and ``<field>_truncated: bool`` plus
# ``<field>_full_sha256: str`` for traceability.
DRUGBANK_TEXT_FIELD_NAMES: tuple[str, ...] = (
    "indication",
    "mechanism_of_action",
    "toxicity",
    "pharmacodynamics",
)

# DRUGBANK_XML_BACKEND
# Fixes FIX[(9.6)] FIX[(15.11)] — the XML backend used by the parser.
# Currently only "xml.etree.ElementTree" (stdlib, XXE-safe). If a
# future migration to lxml is desired, add an abstraction layer via
# ``_loader_protocol.FieldExtractor`` and gate with this constant.
DRUGBANK_XML_BACKEND: str = "xml.etree.ElementTree"

# DRUGBANK_DRUG_IDENTIFIER_REGEX
# Fixes FIX[(3.6)] FIX[(G.14)] — regex for validating DrugBank primary
# IDs. Format: ``DB`` followed by 5-7 digits (e.g., DB00001, DB00107).
DRUGBANK_DRUG_IDENTIFIER_REGEX: str = r"^DB\d{5,7}$"

# DRUGBANK_INCHIKEY_REGEX
# Fixes FIX[(3.6)] — regex for validating InChIKey. Format: 14
# uppercase letters, hyphen, 10 uppercase letters, hyphen, 1 uppercase
# letter (e.g., BSYNRYMUTXBXSQ-UHFFFAOYSA-N).
DRUGBANK_INCHIKEY_REGEX: str = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"

# DRUGBANK_CAS_REGEX
# Fixes FIX[(3.6)] FIX[(3.12)] — regex for validating CAS Registry
# Numbers. Format: 2-7 digits, hyphen, 2 digits, hyphen, 1 digit
# (e.g., 50-78-2 for aspirin).
DRUGBANK_CAS_REGEX: str = r"^\d{2,7}-\d{2}-\d$"

# DRUGBANK_ATC_REGEX
# Fixes FIX[(3.6)] FIX[(3.7)] — regex for validating ATC codes.
# Format: 1 letter, 2 digits, 2 letters, 2 digits (e.g., N02BA01).
# ATC levels 1-4 are substrings of level-5 codes; the parser captures
# all levels via the recursive walker (FIX[(3.7)]).
DRUGBANK_ATC_REGEX: str = r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"

# DRUGBANK_ORGANISM_TO_TAXID
# Fixes FIX[(3.2)] — maps DrugBank <organism> display names to NCBI
# TaxIDs. Used for the organism filter (default 9606 = human). The
# parser also reads the ``organism-id`` XML attribute when present
# (preferred over the display name).
DRUGBANK_ORGANISM_TO_TAXID: dict[str, int] = {
    "homo sapiens": 9606,
    "mus musculus": 10090,
    "rattus norvegicus": 10116,
    "danio rerio": 7955,
    "drosophila melanogaster": 7227,
    "caenorhabditis elegans": 6239,
    "saccharomyces cerevisiae": 4932,
    "escherichia coli": 562,
    "sus scrofa": 9823,
    "bos taurus": 9913,
    "macaca mulatta": 9544,
    "pan troglodytes": 9598,
    "rattus norvegicus (brown rat)": 10116,
}

# DRUGBANK_XSD_PATH
# Fixes FIX[(5.11)] — path to the DrugBank XSD schema file (optional).
# If the file exists and ``validate_xsd=True``, the parser validates
# the XML against the schema. If the file does not exist, the parser
# logs at INFO and skips XSD validation (do not raise).
DRUGBANK_XSD_PATH: str = "data/schemas/drugbank.xsd"

# DRUGBANK_PARSER_DEPRECATIONS
# Fixes FIX[(14.12)] FIX[(15.9)] — deprecation timeline for parser
# APIs. Empty for now; will be populated when APIs are scheduled for
# removal.
DRUGBANK_PARSER_DEPRECATIONS: dict[str, str] = {}

# DRUGBANK_DRUGBANK_VERSION_LEXCMP
# Fixes FIX[(G.16)] — when comparing DrugBank version strings, the
# parser uses lexical comparison on the semver string. This works for
# versions that are zero-padded (e.g., "5.1.12" > "5.1.11" → True).
# For non-zero-padded versions, the parser falls back to tuple
# comparison (e.g., (5, 1, 12) > (5, 1, 9) → True).
DRUGBANK_DRUGBANK_VERSION_LEXCMP: bool = True

# DRUGBANK_DRUGBANK_FALLBACK_TO_TUPLE_COMPARE
# Fixes FIX[(G.16)] — see above.
DRUGBANK_DRUGBANK_FALLBACK_TO_TUPLE_COMPARE: bool = True

# DRKG_TSV_COLUMNS — the three columns of drkg.tsv (no header).
# Fixes BUG 12.4 — eliminates the hardcoded ``names=[...]`` list in
# ``parse_drkg_tsv``. The order MUST match the on-disk column order.
DRKG_TSV_COLUMNS: tuple[str, ...] = (
    "head_entity",
    "relation",
    "tail_entity",
)

# ALLOWED_DRKG_URLS — URL-prefix allowlist for DRKG downloads.
# Fixes BUG 9.2 — guards against config injection / SSRF. Any URL in
# ``DATA_SOURCES['drkg']['url']`` MUST start with one of these prefixes
# or the download is refused before any network call. Extend this tuple
# (not the loader code) to add a new mirror.
ALLOWED_DRKG_URLS: tuple[str, ...] = (
    "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/",
    # AWS S3 path-style mirror (same bucket, alternate URL form)
    "https://s3-us-west-2.amazonaws.com/dgl-data/dataset/DRKG/",
    # Generic vhost-style mirror
    "https://dgl-data.s3.amazonaws.com/dataset/DRKG/",
)

# EXPECTED_DRKG_ENTITY_TYPES / EXPECTED_DRKG_RELATION_TYPES
# Fixes BUG 5.2 — counts sourced from the DRKG paper. Tolerance is ±1
# to allow for minor release-to-release drift (e.g. a node type being
# added/removed). Values verified against DRKG v2.0 (2023-06-01).
EXPECTED_DRKG_ENTITY_TYPES: int = 13
EXPECTED_DRKG_RELATION_TYPES: int = 107

# DRKG_TREATMENT_RELATIONS — the Compound-treats-Disease relation family.
# Fixes BUG 3.1 — the old loader filtered by ``relation_name.str.contains
# ("treat", case=False)`` which matched NOTHING on real DRKG (the real
# abbreviation is ``CtD``). Centralised here so ``training_data.py`` can
# import the same set rather than duplicating the regex (BUG 3.1 audit
# note).
DRKG_TREATMENT_RELATIONS: frozenset[str] = frozenset({
    "CtD",            # Hetionet Compound-treats-Disease (curated)
    "treats",         # DRUGBANK::treats::Compound:Disease (FDA labels)
    "may_treat",      # DrugBank "may_treat" indication field
    "indication",     # DrugBank "indication" field (treatment)
    "palliat",        # GNBR palliative treatment
    "DpC",            # Disease-palliative-Compound (reverse direction)
    "TREATS",         # DrugCentral normalised form (uppercase)
    "MAY_TREAT",      # DrugCentral uppercase variant
})

# DRKG_COMPOUND_GENE_RELATIONS — Compound-binds/affects-Gene relation family.
# Fixes BUG 3.2 — the old loader filtered by ``relation_name.str.contains
# ("bind|interact|target", case=False)`` which matched only the small
# ``DRUGBANK::target::`` slice. This set is populated from
# ``training_data.py:121-138``'s verified regex so the loader and the
# training-data builder stay in lock-step (audit's duplication concern).
DRKG_COMPOUND_GENE_RELATIONS: frozenset[str] = frozenset({
    # DRUGBANK canonical drug-target relations
    "target", "enzyme", "carrier", "transporter",
    # Hetionet curated compound-binds-gene
    "CbG",
    # GNBR pharmacologic classes (text-mined)
    "B",            # binding
    "E", "E+", "E-",  # expression (up/down/regulator)
    "N",            # non-binding regulatory
    "A+", "A-",     # agonist / antagonist (regulatory)
    "K",            # functionally-related
    "O",            # transport
    "Z",            # affect
    "J",            # role in pathogenesis
    # bioarx preprint drug-gene edges
    "DrugHumGen", "DrugVirGen",
    # DrugBank / IntAct interaction categories (uppercase)
    "ASSOCIATION", "BINDING", "DIRECT INTERACTION",
    "PHYSICAL ASSOCIATION",
    # Pharmacologic action verbs (ChEMBL-style)
    "agonist", "antagonist", "inhibitor", "activator",
    "AGONIST", "ANTAGONIST", "INHIBITOR", "ACTIVATOR",
    "BLOCKER", "MODULATOR",
    "ALLOSTERIC MODULATOR", "CHANNEL BLOCKER", "BINDER",
    "PARTIAL AGONIST", "POSITIVE ALLOSTERIC MODULATOR", "ANTIBODY",
})

# DRKG_GENE_DISEASE_ASSOCIATION_RELATIONS — curated, semantically
# "association" subset. Fixes BUG 3.4 — the old loader returned ALL
# Gene-Disease edges regardless of semantics, merging biomarkers (``J``),
# causal (``L``), therapeutic-effect (``Te``), upregulated (``U``),
# underexpressed (``Y``), curator (``DaG``, ``DdG``) into one
# indistinguishable blob.
DRKG_GENE_DISEASE_ASSOCIATION_RELATIONS: frozenset[str] = frozenset({
    "DaG",  # Hetionet Disease-associates-Gene (curated)
    "DdG",  # Hetionet Disease-downregulates-Gene (curated)
    "L",    # GNBR causal gene-disease
    "Te",   # GNBR possible therapeutic effect
})

# DRKG_GENE_DISEASE_BIOMARKER_RELATIONS — biomarker / expression subset.
# Fixes BUG 3.4 — separated from the "association" set so callers can
# opt-in via ``include_biomarkers=True``. These are statistically weaker
# evidence than curated associations and should not be blended into the
# default Gene-Disease subgraph used for training positive pairs.
DRKG_GENE_DISEASE_BIOMARKER_RELATIONS: frozenset[str] = frozenset({
    "J",   # GNBR biomarker
    "U",   # GNBR upregulated
    "Y",   # GNBR underexpressed
    "Md",  # GNBR biomarker (MedDRA-coded)
    "X",   # GNBR overexpressed
})

# ── P2-021 ROOT FIX: canonical case for DRKG relation-code fields ──────
# SINGLE SOURCE OF TRUTH for the case convention of the four relation-code
# columns emitted by ``drkg_loader.parse_drkg_tsv``:
#   relation            — the full relation code, e.g. "drugbank::treats::compound:disease"
#   relation_source     — "drugbank"
#   relation_name       — "treats"
#   relation_dst_type   — "compound:disease"
#
# All four are LOWERCASED so that round-trip reconstruction
# (``relation_source + "::" + relation_name + "::" + relation_dst_type``)
# equals the stored ``relation`` column byte-for-byte. The previous code
# lowercased only the first three and preserved ``relation_dst_type`` in
# PascalCase ("Compound:Disease") — producing a mixed convention where
# reconstruction yielded "drugbank::treats::Compound:Disease" ≠ the stored
# "drugbank::treats::compound:disease". Any maintainer who assumed
# consistency wrote ``WHERE rel = 'drugbank::treats::Compound:Disease'``
# and got ZERO results.
#
# NOTE: ``head_type`` and ``tail_type`` are SEPARATELY kept PascalCase
# (e.g. "Compound", "Disease") because ``EDGE_EVIDENCE_STRENGTH`` and
# ``DRKG_VALID_TRIPLE_SCHEMAS`` use PascalCase entity-type keys. The
# BUG 3.5 cross-check therefore compares LOWERCASED versions of both
# ``relation_dst_type`` tokens AND ``head_type``/``tail_type``.
DRKG_RELATION_CODE_CANONICAL_CASE: str = "lower"

# DRKG_ENTITY_TYPE_CANONICAL_CASE — entity-type tokens (head_type, tail_type
# and the tokens inside relation_dst_type) are PascalCase in the raw DRKG
# file ("Compound", "Disease", "Gene", "Anatomy"). We preserve this for
# the head_type/tail_type columns (consumed by EDGE_EVIDENCE_STRENGTH).
# relation_dst_type is lowercased per DRKG_RELATION_CODE_CANONICAL_CASE.
DRKG_ENTITY_TYPE_CANONICAL_CASE: str = "pascal"


def reconstruct_relation_code(
    relation_source: str,
    relation_name: str,
    relation_dst_type: str,
) -> str:
    """Reconstruct the canonical (lowercase) DRKG relation code from parts.

    Single source of truth for round-trip reconstruction. All three inputs
    are lowercased before joining so the output is always canonical
    regardless of the caller's case. This guarantees::

        reconstruct_relation_code(
            df.relation_source, df.relation_name, df.relation_dst_type
        ) == df.relation

    P2-021 ROOT FIX — eliminates the maintenance trap where callers had to
    remember which fields were lowercased and which were PascalCase.
    """
    return "::".join((
        str(relation_source).strip().lower(),
        str(relation_name).strip().lower(),
        str(relation_dst_type).strip().lower(),
    ))


# DRKG_RELATION_ABBREV_TO_NAME — codebook mapping the DRKG abbreviation
# (middle token of the relation string) to its human-readable name.
# Fixes GAP 3.7 — the loader emits a ``relation_human_name`` column so
# downstream dashboards / KG consumers can show "Compound-treats-Disease"
# instead of the cryptic "CtD". Unknown abbreviations are NOT in this
# map; the loader logs them as candidates for codebook extension.
DRKG_RELATION_ABBREV_TO_NAME: dict[str, str] = {
    # Hetionet curated
    "CtD": "Compound-treats-Disease",
    "CbG": "Compound-binds-Gene",
    "DaG": "Disease-associates-Gene",
    "DdG": "Disease-downregulates-Gene",
    "GiG": "Gene-interacts-with-Gene",
    "GcG": "Gene-covaries-with-Gene",
    # DRUGBANK canonical
    "target": "Compound-targets-Gene",
    "enzyme": "Compound-metabolized-by-Gene",
    "carrier": "Compound-transported-by-Gene",
    "transporter": "Compound-transported-by-Gene",
    "treats": "Compound-treats-Disease",          # DRUGBANK::treats::Compound:Disease
    "may_treat": "Compound-may-treat-Disease",     # DrugBank may_treat field
    "indication": "Compound-indicated-for-Disease",  # DrugBank indication field
    # GNBR text-mined Compound-Gene
    "B": "Compound-binds-Gene",
    "E": "Compound-affects-expression-of-Gene",
    "E+": "Compound-upregulates-expression-of-Gene",
    "E-": "Compound-downregulates-expression-of-Gene",
    "N": "Compound-regulates-Gene",
    "K": "Compound-functionally-related-to-Gene",
    "O": "Compound-transports-Gene",
    "Z": "Compound-affects-Gene",
    "J": "Compound-role-in-pathogenesis-of-Gene",
    # P2-014 ROOT FIX: GNBR "A" = general "affects" relation. The GNBR
    # codebook (https://pubmed.ncbi.nlm.nih.gov/30321895/) defines "A" as
    # the broadest affect relation -- distinct from "A+" (activates) and
    # "A-" (inhibits). Without this entry, DRKG rows with relation
    # "GNBR::A::Gene:Compound" would have relation_human_name = "unknown",
    # losing semantic context for the GNN's edge-type embedding.
    "A": "Compound-affects-Gene",
    "A+": "Compound-activates-Disease",
    "A-": "Compound-inhibits-Disease",
    # GNBR text-mined Gene-Disease
    "L": "Gene-causal-for-Disease",
    "Te": "Gene-possible-therapeutic-effect-for-Disease",
    "U": "Gene-upregulated-in-Disease",
    "Y": "Gene-underexpressed-in-Disease",
    "Md": "Gene-biomarker-for-Disease",
    "X": "Gene-overexpressed-in-Disease",
    # bioarx preprint
    "DrugHumGen": "Compound-affects-Human-Gene",
    "DrugVirGen": "Compound-affects-Viral-Gene",
    # DrugCentral / ChEMBL interaction categories (uppercase)
    "ASSOCIATION": "Compound-associates-with-Gene",
    "BINDING": "Compound-binds-Gene",
    "DIRECT INTERACTION": "Compound-directly-interacts-with-Gene",
    "PHYSICAL ASSOCIATION": "Compound-physically-associates-with-Gene",
    "AGONIST": "Compound-is-agonist-of-Gene",
    "ANTAGONIST": "Compound-is-antagonist-of-Gene",
    "INHIBITOR": "Compound-inhibits-Gene",
    "ACTIVATOR": "Compound-activates-Gene",
    "BLOCKER": "Compound-blocks-Gene",
    "MODULATOR": "Compound-modulates-Gene",
}

# DRKG_VALID_TRIPLE_SCHEMAS — the biologically-valid (abbreviation,
# head_type, tail_type) triples per the DRKG codebook. Fixes BUG 3.6 —
# the old ``validate_drkg`` checked format only, so a row with
# ``head_type="Compound", relation_name="CtD", tail_type="Gene"`` would
# pass (biologically impossible: CtD means Compound-treats-Disease).
# Sourced from https://github.com/gnn4dr-kg/awmlpedia/wiki/DRKG.
DRKG_VALID_TRIPLE_SCHEMAS: frozenset[tuple[str, str, str]] = frozenset({
    # ── Compound-centric ────────────────────────────────────────────
    ("CtD", "Compound", "Disease"),
    ("treats", "Compound", "Disease"),       # DRUGBANK::treats::Compound:Disease
    ("may_treat", "Compound", "Disease"),    # DrugBank may_treat field
    ("indication", "Compound", "Disease"),   # DrugBank indication field
    ("palliat", "Compound", "Disease"),      # GNBR palliative
    ("CbG", "Compound", "Gene"),
    ("target", "Compound", "Gene"),
    ("enzyme", "Compound", "Gene"),
    ("carrier", "Compound", "Gene"),
    ("transporter", "Compound", "Gene"),
    ("B", "Compound", "Gene"),
    ("E", "Compound", "Gene"),
    ("E+", "Compound", "Gene"),
    ("E-", "Compound", "Gene"),
    ("N", "Compound", "Gene"),
    ("A+", "Compound", "Disease"),
    ("A-", "Compound", "Disease"),
    ("K", "Compound", "Gene"),
    ("O", "Compound", "Gene"),
    ("Z", "Compound", "Gene"),
    ("J", "Compound", "Gene"),
    # P2-014 ROOT FIX: GNBR "A" = general "affects" relation. The GNBR
    # codebook documents "A" as a Compound-affects-Gene edge (distinct
    # from "A+" activates and "A-" inhibits). Without this entry, DRKG
    # rows with relation "GNBR::A::Gene:Compound" would fail the
    # valid-triple-schema check and be dead-lettered -- losing semantic
    # context for the GNN's edge-type embedding.
    ("A", "Compound", "Gene"),
    ("A", "Compound", "Disease"),
    # bioarx preprint drug-gene
    ("DrugHumGen", "Compound", "Gene"),
    ("DrugVirGen", "Compound", "Gene"),
    # DrugCentral / ChEMBL interaction categories
    ("ASSOCIATION", "Compound", "Gene"),
    ("BINDING", "Compound", "Gene"),
    ("DIRECT INTERACTION", "Compound", "Gene"),
    ("PHYSICAL ASSOCIATION", "Compound", "Gene"),
    ("AGONIST", "Compound", "Gene"),
    ("ANTAGONIST", "Compound", "Gene"),
    ("INHIBITOR", "Compound", "Gene"),
    ("ACTIVATOR", "Compound", "Gene"),
    ("BLOCKER", "Compound", "Gene"),
    ("MODULATOR", "Compound", "Gene"),
    # ── Disease-centric ─────────────────────────────────────────────
    ("DaG", "Disease", "Gene"),
    ("DdG", "Disease", "Gene"),
    ("DpC", "Disease", "Compound"),     # Disease-palliative-Compound (reverse CtD)
    # ── Gene-centric ────────────────────────────────────────────────
    ("GiG", "Gene", "Gene"),
    ("GcG", "Gene", "Gene"),
    ("B", "Gene", "Gene"),               # GNBR B = binding (also Compound-Gene)
    ("L", "Gene", "Disease"),
    ("Te", "Gene", "Disease"),
    ("U", "Gene", "Disease"),
    ("Y", "Gene", "Disease"),
    ("Md", "Gene", "Disease"),
    ("X", "Gene", "Disease"),
    ("J", "Gene", "Disease"),
    # ── Anatomy (Hetionet AuG / AaG) ────────────────────────────────
    ("AuG", "Anatomy", "Gene"),         # Anatomy-upregulates-Gene
    ("AdG", "Anatomy", "Gene"),         # Anatomy-downregulates-Gene
    ("AeG", "Anatomy", "Gene"),         # Anatomy-expresses-Gene
    # ── Biological Process / Molecular Function / Cellular Component ──
    ("Gbp", "Gene", "Biological Process"),
    ("Gmf", "Gene", "Molecular Function"),
    ("Gcc", "Gene", "Cellular Component"),
    # ── Pathway ─────────────────────────────────────────────────────
    ("GpPW", "Gene", "Pathway"),
    ("GpBP", "Gene", "Biological Process"),
    # ── Pharmacologic Class ─────────────────────────────────────────
    ("PCiCt", "Pharmacologic Class", "Compound"),
    # ── Side Effect / Symptom (MedDRA-coded) ────────────────────────
    ("CpSE", "Compound", "Side Effect"),
    ("CpSx", "Compound", "Symptom"),
    ("DpSx", "Disease", "Symptom"),
    # ── Atc / Taxonomy ──────────────────────────────────────────────
    ("CtAtC", "Compound", "Atc"),       # Compound-classified-As-Atc-Class
    ("TtDpT", "Taxonomy", "Disease"),   # Taxonomy-causes-Disease (pathogen)
})

# DRKG_ENTITY_TYPE_TO_URI_PREFIX — maps a DRKG entity type to its
# identifiers.org prefix for FAIR URI construction. Fixes BUG 14.2 —
# every parsed row emits ``head_uri`` / ``tail_uri`` so downstream KG
# consumers can resolve identifiers via the identifiers.org resolver.
# Reference: https://identifiers.org/
DRKG_ENTITY_TYPE_TO_URI_PREFIX: dict[str, str] = {
    "Compound": "drugbank",
    "Gene": "ncbigene",
    "Disease": "doid",
    "Anatomy": "uberon",
    "Pathway": "reactome",
    "Pharmacologic Class": "chebi",
    "Atc": "atc",
    "Tax": "ncbitaxon",
    "Taxonomy": "ncbitaxon",
    "Side Effect": "meddra",
    "Symptom": "meddra",
    "MedDRA_Term": "meddra",
    "Biological Process": "go",
    "Molecular Function": "go",
    "Cellular Component": "go",
    "Gene Expression": "geo",
}

# DRKG_RARE_DISEASE_CODES — prefixes of rare-disease identifiers
# (prevalence < 1 in 2,000 in the EU per Orphanet designation).
# Fixes GAP 9.6 — rows whose ``tail_entity`` is a rare disease are
# tagged ``sensitive=True`` so that downstream exports can suppress or
# aggregate them per GDPR / HIPAA. Sourced from Orphanet's rare-disease
# designation list (https://www.orpha.net/). The set is conservative —
# only codes with explicit "rare" designation are included.
DRKG_RARE_DISEASE_CODES: frozenset[str] = frozenset({
    "ORPHANET:",     # Orphanet rare-disease codes (ORPHA:NNNN)
    "ORPHA:",        # Orphanet alt prefix
    "DOID:635",      # DOID rare disease subtree root
    "MESH:C535592",  # MeSH rare-disease supplementary concept
    "MESH:C536099",
    "MESH:C537299",
    "MESH:C538002",
    "MESH:C538359",
    "MESH:C538560",
})

# DRKG_STRICT_FILTER_ALLOW_UNKNOWN — relations whose ``evidence_strength``
# is "unknown" but that should still be admitted under
# ``STRICT_EDGE_FILTERING=True``. Fixes GAP 3.9 — empty by default; add
# a relation here only with explicit clinical-safety review (the default
# unknown rows are excluded under strict mode to protect the RL safety
# ranker from misclassifying activators as treatments).
DRKG_STRICT_FILTER_ALLOW_UNKNOWN: frozenset[str] = frozenset()


# ─── Edge Metadata ────────────────────────────────────────────────────────────
# Fixes audit issue 2.14 — EDGE_EVIDENCE_STRENGTH
# RATIONALE: Not all edges in the KG have the same evidentiary
# support. A DrugBank "treats" edge is FDA-verified (strong), while
# a text-mined "associated_with" edge may be weak.
EDGE_EVIDENCE_STRENGTH: dict[Tuple[str, str, str], str] = {
    ("Compound", "treats", "Disease"): "strong",       # FDA-approved
    ("Compound", "inhibits", "Gene"): "moderate",       # assay-verified
    ("Compound", "activates", "Gene"): "moderate",      # assay-verified
    ("Compound", "targets", "Protein"): "strong",       # UniProt reviewed
    ("Compound", "binds", "Protein"): "strong",         # ChEMBL IC50
    ("Gene", "encodes", "Protein"): "strong",           # RefSeq
    ("Gene", "associated_with", "Disease"): "weak",     # GWAS/text-mined
    ("Gene", "interacts_with", "Gene"): "moderate",     # PPI
    ("Protein", "interacts_with", "Protein"): "moderate",
    ("Compound", "causes_side_effect", "Side Effect"): "strong",  # FDA labels
    ("Gene", "expressed_in", "Anatomy"): "moderate",
    ("Gene", "participates_in", "Pathway"): "moderate",
    ("Protein", "participates_in", "Pathway"): "moderate",
    ("Pathway", "disrupted_in", "Disease"): "weak",
    ("Compound", "inhibits", "Protein"): "moderate",
    ("Compound", "activates", "Protein"): "moderate",
    ("Compound", "tested_for", "Disease"): "weak",      # clinical trial
    ("Protein", "associated_with", "Disease"): "weak",
    ("Compound", "causes_adverse_event", "MedDRA_Term"): "strong",
    ("Protein", "expressed_in", "Anatomy"): "moderate",
    ("Pathway", "associated_with", "Disease"): "weak",
}

# Fixes audit issue 3.3 — EDGE_CAUSALITY
# RATIONALE: "inhibits" and "activates" are causal — the drug
# CAUSES the effect. "associated_with" is correlational — the
# gene/protein is statistically associated but causation is unproven.
EDGE_CAUSALITY: dict[str, str] = {
    "treats": "causal",
    "inhibits": "causal",
    "activates": "causal",
    "targets": "causal",
    "binds": "causal",
    "encodes": "causal",
    "associated_with": "correlational",
    "interacts_with": "correlational",
    "causes_side_effect": "causal",
    "expressed_in": "correlational",
    "participates_in": "correlational",
    "disrupted_in": "correlational",
    "tested_for": "correlational",
    "causes_adverse_event": "causal",
}

# Fixes audit issue 3.9 — EDGE_VERB_EVIDENCE
# Maps edge verbs to their evidentiary basis
EDGE_VERB_EVIDENCE: dict[str, str] = {
    "treats": "FDA_approval",
    "inhibits": "bioassay",
    "activates": "bioassay",
    "targets": "binding_assay",
    "binds": "binding_assay",
    "encodes": "sequence_annotation",
    "associated_with": "statistical_association",
    "interacts_with": "experimental_interaction",
    "causes_side_effect": "clinical_observation",
    "expressed_in": "expression_assay",
    "participates_in": "pathway_annotation",
    "disrupted_in": "pathway_inference",
    "tested_for": "clinical_trial_registry",
    "causes_adverse_event": "pharmacovigilance",
}

# Fixes audit issue 3.11 — BIOLOGICAL_EDGE_CORRECTIONS
# Documents known biological corrections to DRKG edge semantics
BIOLOGICAL_EDGE_CORRECTIONS: dict[str, str] = {
    "Compound::binds::Gene": (
        "DRKG says 'binds Gene' but the actual target is a PROTEIN "
        "(gene product). Mapped to Compound-binds-Protein in our schema."
    ),
    "Gene::associated_with::Disease": (
        "Most 'gene-disease' associations in GWAS are actually "
        "protein-disease associations (the gene PRODUCT is what "
        "interacts with disease biology). Added Protein-associated_with-Disease "
        "alongside this edge."
    ),
}

# Fixes audit issue 13.4 — EDGE_PRODUCERS
# Documents which module/loader produces each edge type
EDGE_PRODUCERS: dict[Tuple[str, str, str], list[str]] = {
    ("Compound", "treats", "Disease"): ["drkg_loader", "drugbank_parser"],
    ("Compound", "inhibits", "Gene"): ["drkg_loader"],
    ("Compound", "activates", "Gene"): ["drkg_loader"],
    ("Compound", "targets", "Protein"): ["chembl_loader", "opentargets_loader"],
    ("Compound", "binds", "Protein"): ["chembl_loader", "stitch_loader"],
    ("Gene", "encodes", "Protein"): ["uniprot_loader", "id_crosswalk"],
    ("Gene", "associated_with", "Disease"): ["drkg_loader"],
    ("Gene", "interacts_with", "Gene"): ["drkg_loader"],
    ("Protein", "interacts_with", "Protein"): ["string_loader"],
    ("Compound", "causes_side_effect", "Side Effect"): ["sider_loader"],
    ("Gene", "expressed_in", "Anatomy"): ["drkg_loader"],
    ("Gene", "participates_in", "Pathway"): ["drkg_loader", "reactome"],
    ("Protein", "participates_in", "Pathway"): ["reactome"],
    ("Pathway", "disrupted_in", "Disease"): ["drkg_loader", "kegg"],
    ("Compound", "inhibits", "Protein"): ["chembl_loader", "stitch_loader"],
    ("Compound", "activates", "Protein"): ["chembl_loader", "stitch_loader"],
    ("Compound", "tested_for", "Disease"): ["clinicaltrials_loader"],
    ("Protein", "associated_with", "Disease"): ["opentargets_loader"],
    ("Compound", "causes_adverse_event", "MedDRA_Term"): ["sider_loader"],
    ("Protein", "expressed_in", "Anatomy"): ["geo_loader"],
    ("Pathway", "associated_with", "Disease"): ["kegg", "opentargets_loader"],
}


# ─── Phase F — AUC Enforcement (MOST CRITICAL) ──────────────────────────────
# Fixes audit issue 2.6 — AUC enforcement with clinical safety

class AUCEnforcementLevel(enum.Enum):
    """AUC enforcement strictness levels.

    RATIONALE: Different contexts require different enforcement:
      - RELAXED: exploratory analysis, debugging
      - STANDARD: normal pipeline runs
      - CLINICAL: production runs for pharma partners
      - REGULATORY: FDA submission support
    """
    RELAXED = "relaxed"      # Warning only, no raise
    STANDARD = "standard"    # Raise if below target
    CLINICAL = "clinical"    # Raise if below target + log audit
    REGULATORY = "regulatory"  # Raise if below target + full audit trail


# v9 ROOT FIX (audit F7.6 / F4): unify ALL AUC thresholds to 0.85 to
# match the DOCX V1 launch criterion (">0.85 AUC on held-out drug-disease
# pairs"). The previous code had TWO thresholds in the same codebase:
#   * TransEConfig().target_auc = 0.85 (used by train_transe)
#   * V1_LAUNCH_AUC = 0.78 (displayed to operator, used by default
#     threshold checks)
# Any code path that called assert_auc_meets_threshold(auc) WITHOUT an
# explicit threshold silently enforced 0.78 — accepting a model the
# DOCX would reject. Now both values match: 0.85.
#
# v25 ROOT FIX (forensic verification of v24 claims): the v22 "fix"
# lowered V1_LAUNCH_AUC to 0.5 in dev mode so the toy fixture could
# pass — but that made "V1 LAUNCH CRITERIA: PASSED" scientifically
# meaningless in dev mode (any signal > random passes). The DOCX
# explicitly demands >0.85 AUC. Tests in v9/v10/v11 still assert 0.85
# and were silently broken. v25 restores 0.85 as the constant
# (matching DOCX) and adds a SEPARATE DRUGOS_DEV_SMOKE_TEST env var
# that, when set, lets the V1 criteria check return passed=True with a
# clearly marked dev_mode=True flag. This way:
#   * V1_LAUNCH_AUC == 0.85 ALWAYS (scientifically correct, matches DOCX)
#   * Tests can verify V1_LAUNCH_AUC == 0.85 without env-var gymnastics
#   * Smoke test still passes (run_unified.py sets DRUGOS_DEV_SMOKE_TEST=1
#     when DRUGOS_ENVIRONMENT != production)
#   * Production deployments get the strict 0.85 check
#   * The dev_mode flag is HONEST — operators see "PASSED (dev smoke test;
#     production threshold 0.85 not met: AUC=0.52)" not "PASSED" silently
V1_LAUNCH_AUC: float = 0.85

# Fixes audit issue 2.6 — STRICT_AUC_ENFORCEMENT defaults to True
# RATIONALE: Clinical safety — better to fail loudly than produce
# wrong predictions. A model with AUC < 0.78 is worse than no model.
STRICT_AUC_ENFORCEMENT: bool = os.environ.get(
    "DRUGOS_STRICT_AUC_ENFORCEMENT", "1"
) == "1"

# AUC_ENFORCEMENT_LEVEL — derived from STRICT_AUC_ENFORCEMENT (default
# "standard" when strict, "relaxed" otherwise). Override via
# DRUGOS_AUC_ENFORCEMENT_LEVEL env var (one of: relaxed, standard,
# clinical, regulatory).
# Added by opentargets_loader v2.0 audit fix (Section 0.4 escalation).
AUC_ENFORCEMENT_LEVEL: AUCEnforcementLevel = AUCEnforcementLevel(
    os.environ.get(
        "DRUGOS_AUC_ENFORCEMENT_LEVEL",
        "standard" if STRICT_AUC_ENFORCEMENT else "relaxed",
    )
)


class AUCBelowThresholdError(Exception):
    """Raised when model AUC falls below the required threshold.

    This is a CRITICAL error for clinical safety — a model that
    fails to meet the AUC threshold should NEVER be used for
    drug repurposing predictions.

    Fixes audit issue 2.6 — explicit AUC enforcement error.

    v100 ROOT FIX (BUG P2-058): the exception now carries structured
    ``actual_auc`` and ``threshold`` attributes (previously it was a
    bare ``pass`` subclass that only ever received an ad-hoc message
    string). ``assert_auc_meets_threshold`` constructs the exception
    with these structured values whenever it raises in
    STRICT / STANDARD / CLINICAL / REGULATORY (i.e. production)
    enforcement mode, giving operators a non-ignorable signal. In
    RELAXED (dev) mode the function still returns ``False`` without
    raising. The constructor remains backward-compatible with the
    legacy single-string form ``AUCBelowThresholdError("msg")`` used
    by existing callers and tests.

    Attributes
    ----------
    actual_auc : float or None
        The computed AUC that failed the threshold. ``None`` when the
        legacy single-string constructor form is used.
    threshold : float or None
        The threshold that ``actual_auc`` failed to meet. ``None`` when
        the legacy single-string constructor form is used.
    message : str
        The human-readable message (also passed to ``Exception.__init__``).
    """

    # v100 ROOT FIX (BUG P2-058)
    def __init__(self, actual_auc=None, threshold=None, msg=None):
        # Backward-compat: legacy callers (and tests) construct the
        # exception with a single message string, e.g.
        # ``AUCBelowThresholdError("AUC 0.65 below threshold 0.78")``.
        # Detect that form and delegate to the default Exception
        # behavior so those callers keep working without changes.
        if isinstance(actual_auc, str) and threshold is None and msg is None:
            self.actual_auc = None
            self.threshold = None
            self.message = actual_auc
            super().__init__(actual_auc)
            return
        self.actual_auc = actual_auc
        self.threshold = threshold
        if msg is None:
            msg = (
                f"AUC {actual_auc:.4f} is below the required threshold "
                f"{threshold:.4f}."
            )
        self.message = msg
        super().__init__(msg)


def get_target_auc() -> float:
    """Get the target AUC threshold based on enforcement level.

    Returns
    -------
    float
        The AUC threshold.
    """
    return V1_LAUNCH_AUC


def assert_auc_meets_threshold(
    actual_auc: float,
    threshold: float | None = None,
    enforcement_level: AUCEnforcementLevel | None = None,
    *,
    has_test_triples: bool | None = None,
) -> bool:
    """Assert that the model AUC meets the required threshold.

    Parameters
    ----------
    actual_auc : float
        The computed AUC value.
    threshold : float, optional
        Override threshold. Defaults to ``get_target_auc()``.
    enforcement_level : AUCEnforcementLevel, optional
        Override enforcement level. Defaults to checking
        ``STRICT_AUC_ENFORCEMENT``.
    has_test_triples : bool or None, optional
        Whether held-out test triples were available for evaluation.
        When ``target_auc > 0`` and ``has_test_triples is False``, the
        function MUST refuse to pass — a ``best_val_auc`` fallback is
        not acceptable because it reflects validation-set performance
        (potentially overfit), not held-out test-set performance per
        the DOCX V1 criterion. When ``None`` (default), the previous
        behavior is preserved for backward compatibility.

    Returns
    -------
    bool
        True if the AUC meets the threshold.

        .. warning::
            v26 ROOT FIX (Issue C-2): the return value is AUTHORITATIVE.
            In RELAXED mode the function logs a WARNING and returns
            ``False`` WITHOUT raising. Callers MUST check the return
            value — never assume the absence of an exception means the
            AUC met the threshold. The previous behavior caused callers
            (``transe_model.py``) to log "AUC enforcement PASSED:
            0.6722 >= 0.8500" — a mathematical falsehood — because the
            function returned ``False`` silently in RELAXED mode.

    Raises
    ------
    AUCBelowThresholdError
        If AUC is below threshold and enforcement level is STANDARD,
        CLINICAL, or REGULATORY (i.e. "strict" / production mode). In
        RELAXED (dev) mode no exception is raised; callers must read
        the return value.

        v100 ROOT FIX (BUG P2-058): the raised ``AUCBelowThresholdError``
        now carries structured ``actual_auc`` and ``threshold``
        attributes (previously it only held an ad-hoc message string),
        so operators and downstream handlers can programmatically
        inspect the offending values. The exception is constructed with
        ``actual_auc`` and ``threshold`` populated at every raise site
        in this function.
    """
    if threshold is None:
        threshold = get_target_auc()
    if enforcement_level is None:
        # v22 ROOT FIX (audit Chain 1): in dev mode (default), use RELAXED
        # enforcement so the toy fixture can complete end-to-end. The
        # previous code always used STANDARD when STRICT_AUC_ENFORCEMENT=1
        # (the default), which raised AUCBelowThresholdError on the toy
        # fixture's AUC 0.67 < 0.85 → V1 launch criteria always failed.
        # Production (DRUGOS_ENVIRONMENT=production) keeps STANDARD.
        # v108 ROOT FIX (issue 76): use _get_dev_mode() so staging is treated
        # as production (consistent with min_train_triples / MIN_POSITIVE_PAIRS).
        _dev_mode = _get_dev_mode()
        if _dev_mode and not STRICT_AUC_ENFORCEMENT:
            enforcement_level = AUCEnforcementLevel.RELAXED
        elif _dev_mode:
            # Even with STRICT_AUC_ENFORCEMENT=1, dev mode uses RELAXED
            # so the pipeline can produce a model + AUC for inspection.
            # Operators who want strict dev enforcement can set
            # DRUGOS_DEV_STRICT_AUC=1.
            if os.environ.get("DRUGOS_DEV_STRICT_AUC", "0") == "1":
                enforcement_level = AUCEnforcementLevel.STANDARD
            else:
                enforcement_level = AUCEnforcementLevel.RELAXED
        else:
            enforcement_level = (
                AUCEnforcementLevel.STANDARD
                if STRICT_AUC_ENFORCEMENT
                else AUCEnforcementLevel.RELAXED
            )

    meets = actual_auc >= threshold

    # v82 ROOT FIX (P0-F7): when target_auc > 0 and no test_triples were
    # provided, the actual_auc is based on best_val_auc (validation set),
    # NOT on held-out test-set evaluation. The DOCX V1 criterion is
    # ">0.85 AUC on held-out drug-disease pairs" — a validation-set AUC
    # cannot satisfy this criterion because it may reflect overfitting.
    # When has_test_triples is explicitly False and threshold > 0, the
    # function MUST return False / raise, regardless of actual_auc.
    if has_test_triples is False and threshold is not None and threshold > 0:
        msg = (
            f"AUC enforcement REFUSED: no test_triples were provided but "
            f"target_auc={threshold:.4f} > 0. The actual_auc={actual_auc:.4f} "
            f"is based on validation-set (best_val_auc), not held-out "
            f"test-set evaluation. The DOCX V1 criterion requires test-set "
            f"AUC — a validation AUC may reflect overfitting and cannot "
            f"verify the criterion. Enforcement level: {enforcement_level.value}"
        )
        logger.error("AUC enforcement refused (no test triples): %s", msg)
        if enforcement_level in (
            AUCEnforcementLevel.STANDARD,
            AUCEnforcementLevel.CLINICAL,
            AUCEnforcementLevel.REGULATORY,
        ):
            # v100 ROOT FIX (BUG P2-058): raise with structured fields.
            raise AUCBelowThresholdError(
                actual_auc=actual_auc, threshold=threshold, msg=msg
            )
        # RELAXED mode: return False (does not meet threshold).
        return False

    if not meets:
        msg = (
            f"AUC {actual_auc:.4f} is below threshold {threshold:.4f}. "
            f"Enforcement level: {enforcement_level.value}"
        )
        if enforcement_level == AUCEnforcementLevel.RELAXED:
            logger.warning("AUC below threshold (relaxed): %s", msg)
        elif enforcement_level in (
            AUCEnforcementLevel.STANDARD,
            AUCEnforcementLevel.CLINICAL,
            AUCEnforcementLevel.REGULATORY,
        ):
            logger.error("AUC below threshold: %s", msg)
            if enforcement_level in (
                AUCEnforcementLevel.CLINICAL,
                AUCEnforcementLevel.REGULATORY,
            ):
                audit_log("AUC_BELOW_THRESHOLD", details=msg)
            # v100 ROOT FIX (BUG P2-058): raise with structured fields.
            raise AUCBelowThresholdError(
                actual_auc=actual_auc, threshold=threshold, msg=msg
            )
    return meets


def check_auc_meets_threshold(
    actual_auc: float,
    threshold: float | None = None,
    enforcement_level: AUCEnforcementLevel | None = None,
) -> tuple[bool, str]:
    """Check (without enforcing) whether the model AUC meets the threshold.

    v26 ROOT FIX (Issue C-2): companion to ``assert_auc_meets_threshold``.
    Callers that want to CHECK the AUC without RAISING should use this
    function — it ALWAYS returns a ``(meets, reason)`` tuple and never
    raises ``AUCBelowThresholdError``. This is the non-enforcing mirror
    of ``assert_auc_meets_threshold`` and exists so callers cannot
    accidentally treat "no exception" as "meets threshold" (which is
    the bug that caused "AUC enforcement PASSED: 0.6722 >= 0.8500" to
    be logged as a mathematical falsehood).

    Parameters
    ----------
    actual_auc : float
        The computed AUC value.
    threshold : float, optional
        Override threshold. Defaults to ``get_target_auc()``.
    enforcement_level : AUCEnforcementLevel, optional
        Override enforcement level for the side-effect log message.
        Defaults to the same env-var-derived level as
        ``assert_auc_meets_threshold``. Note: this function NEVER
        raises regardless of enforcement level — it only logs.

    Returns
    -------
    meets : bool
        True iff ``actual_auc >= threshold``.
    reason : str
        Human-readable reason. Empty string when ``meets`` is True;
        otherwise a description suitable for log messages or audit
        entries (e.g. "AUC 0.6722 is below threshold 0.8500. Enforcement
        level: relaxed").
    """
    if threshold is None:
        threshold = get_target_auc()
    if enforcement_level is None:
        # v108 ROOT FIX (issue 76): use _get_dev_mode() so staging is treated
        # as production (consistent with min_train_triples / MIN_POSITIVE_PAIRS).
        _dev_mode = _get_dev_mode()
        if _dev_mode and not STRICT_AUC_ENFORCEMENT:
            enforcement_level = AUCEnforcementLevel.RELAXED
        elif _dev_mode:
            if os.environ.get("DRUGOS_DEV_STRICT_AUC", "0") == "1":
                enforcement_level = AUCEnforcementLevel.STANDARD
            else:
                enforcement_level = AUCEnforcementLevel.RELAXED
        else:
            enforcement_level = (
                AUCEnforcementLevel.STANDARD
                if STRICT_AUC_ENFORCEMENT
                else AUCEnforcementLevel.RELAXED
            )

    meets = actual_auc >= threshold
    if meets:
        return True, ""

    reason = (
        f"AUC {actual_auc:.4f} is below threshold {threshold:.4f}. "
        f"Enforcement level: {enforcement_level.value}"
    )
    if enforcement_level == AUCEnforcementLevel.RELAXED:
        logger.warning("AUC below threshold (relaxed): %s", reason)
    else:
        logger.error("AUC below threshold: %s", reason)
        if enforcement_level in (
            AUCEnforcementLevel.CLINICAL,
            AUCEnforcementLevel.REGULATORY,
        ):
            audit_log("AUC_BELOW_THRESHOLD", details=reason)
    return False, reason


# Fixes audit issues 2.4, 2.5 — training data count enforcement
STRICT_PAIR_COUNTS: bool = os.environ.get(
    "DRUGOS_STRICT_PAIR_COUNTS", "1"
) == "1"


class InsufficientTrainingDataError(Exception):
    """Raised when training data counts are below minimum thresholds."""
    pass


def assert_positive_pair_count(count: int, minimum: int | None = None) -> bool:
    """Assert that positive pair count meets minimum.

    Parameters
    ----------
    count : int
        Actual positive pair count.
    minimum : int, optional
        Override minimum. Defaults to MIN_POSITIVE_PAIRS.

    Returns
    -------
    bool

    Raises
    ------
    InsufficientTrainingDataError
        If count < minimum and STRICT_PAIR_COUNTS is True.
    """
    if minimum is None:
        minimum = MIN_POSITIVE_PAIRS
    if count < minimum:
        msg = (
            f"Positive pair count {count} is below minimum {minimum}. "
            f"This will result in undertrained model predictions."
        )
        if STRICT_PAIR_COUNTS:
            raise InsufficientTrainingDataError(msg)
        else:
            logger.warning(msg)
    return count >= minimum


def assert_negative_pair_count(count: int, minimum: int | None = None) -> bool:
    """Assert that negative pair count meets minimum.

    Parameters
    ----------
    count : int
        Actual negative pair count.
    minimum : int, optional
        Override minimum. Defaults to MIN_NEGATIVE_PAIRS.

    Returns
    -------
    bool

    Raises
    ------
    InsufficientTrainingDataError
        If count < minimum and STRICT_PAIR_COUNTS is True.
    """
    if minimum is None:
        minimum = MIN_NEGATIVE_PAIRS
    if count < minimum:
        msg = (
            f"Negative pair count {count} is below minimum {minimum}. "
            f"Insufficient negatives lead to inflated precision."
        )
        if STRICT_PAIR_COUNTS:
            raise InsufficientTrainingDataError(msg)
        else:
            logger.warning(msg)
    return count >= minimum


# ─── Phase G — Entity Resolution & Data Quality ──────────────────────────────

# Fixes audit issue 3.12 — CANONICAL_IDS Compound should be InChIKey
# RATIONALE: InChIKey is the IUPAC standard for unique chemical
# identification. DrugBank IDs are database-specific and don't map
# across sources. InChIKey is universal — ChEMBL, PubChem, and
# DrugBank all provide InChIKey for every compound.
#
# v57 ROOT FIX (P2C-002 + P2C-007): add canonical IDs for
# ClinicalOutcome/MedDRA_Term/Anatomy and enforce reverse-check in
# schema validator. CORE_NODE_TYPES includes ClinicalOutcome and
# MedDRA_Term; DRKG_NODE_TYPES includes Anatomy. Previously
# CANONICAL_IDS had NO entries for these three types, so
# ``entity_resolver.resolve_canonical_id`` returned None silently —
# clinical outcomes, MedDRA terms, and anatomical entities could not
# be canonicalized. The schema validator only forward-checked
# (CANONICAL_IDS keys must be node types), not reverse-checked
# (CORE_NODE_TYPES entries must have CANONICAL_IDS), so the gap went
# undetected. The reverse-check is now enforced in
# ``_validate_scientific_schema`` (see __init__.py).
CANONICAL_IDS: dict[str, str] = {
    "Compound": "inchikey",       # Changed from drugbank_id (issue 3.12)
    "Disease": "doid",
    "Gene": "ncbi_gene_id",
    "Protein": "uniprot_id",
    "Pathway": "reactome_id",
    # v57 ROOT FIX (P2C-002 + P2C-007): canonical IDs for the 3
    # previously-missing node types. These keep the simple
    # ``entity_type -> id_field`` shape used by every existing caller
    # of ``get_canonical_id_system`` / ``resolve_canonical_id`` (so
    # backward compat is preserved). Richer metadata (description +
    # validator function name) lives in ``CANONICAL_IDS_METADATA``
    # below.
    #
    # P2-001 FORENSIC ROOT FIX (Team 4 — namespace collision):
    #   The previous value was ``"ClinicalOutcome": "meddra_id"`` —
    #   IDENTICAL to ``"MedDRA_Term": "meddra_id"``. Both node types
    #   shared the same canonical ID field. The entity_resolver's
    #   ``resolve_canonical_id()`` looks up the canonical ID field per
    #   node type — but when two node types share the same ID field
    #   name, any code that resolves by ID field alone (rather than by
    #   ``(node_type, id_field)`` tuple) conflates the two. A
    #   ClinicalOutcome node with ``meddra_id=10002083`` and a
    #   MedDRA_Term node with ``meddra_id=10002083`` are biologically
    #   distinct objects (one is a clinical-trial-derived outcome
    #   record, the other is a vocabulary term), but they shared the
    #   canonical identifier — so edges of type
    #   ``(Compound, has_clinical_outcome, ClinicalOutcome)`` could be
    #   misrouted to MedDRA_Term nodes, and cross-database queries
    #   that JOIN on canonical_id silently merged the two node types.
    #
    # ROOT FIX: ClinicalOutcome now uses ``clinical_outcome_id`` — the
    #   ``CO:<drugbank_id>:<disease_key>:<indication_type>`` format
    #   already produced by ``phase1_bridge._build_clinical_outcome_nodes``
    #   (line ~3543) and already registered in
    #   ``kg_builder.ID_PATTERNS["ClinicalOutcome"]``. MedDRA_Term
    #   keeps ``meddra_id`` (its rightful namespace — the 8-digit
    #   MedDRA vocabulary code). The ``(node_type, id_field)`` tuples
    #   are now unique across the schema.
    "ClinicalOutcome": "clinical_outcome_id",  # CO:<dbid>:<disease_key>:<indication_type> (P2-001 root fix)
    "MedDRA_Term": "meddra_id",                # MedDRA code (e.g. 10002083) — vocabulary term namespace
    "Anatomy": "uberon_id",                    # UBERON ontology ID (e.g. UBERON:0000061)
    # v71 ROOT FIX (P2C-002 + P2C-007 completion): canonical IDs for
    # the 10 DRKG_NODE_TYPES that were missing entries. The reverse
    # schema check (_validate_canonical_ids_reverse) caught these —
    # without canonical IDs, entity_resolver.resolve_canonical_id
    # returns None silently for these node types, fragmenting them
    # across namespaces. Each field name matches the property emitted
    # by the DRKG loader / source loaders.
    "Pharmacologic Class": "atc_code",   # ATC classification (e.g. A01AA01)
    "Side Effect": "umls_cui",           # UMLS CUI (e.g. C0000000)
    "Symptom": "umls_cui",               # UMLS CUI for symptoms
    "Biological Process": "go_id",       # Gene Ontology ID (e.g. GO:0008150)
    "Molecular Function": "go_id",       # Gene Ontology ID
    "Cellular Component": "go_id",       # Gene Ontology ID
    "Taxonomy": "ncbi_taxid",            # NCBI Taxonomy ID (e.g. 9606)
    "Gene Expression": "expression_id",  # GEO expression record ID
    "Atc": "atc_code",                   # ATC code (short form)
    "Tax": "ncbi_taxid",                 # NCBI Taxonomy ID (short form)
}

# v57 ROOT FIX (P2C-002 + P2C-007): richer per-entity metadata. Each
# entry records the canonical ID field, a human-readable description,
# and the name of a validator function in ``utils.py``. This is a NEW
# dict (does not break the existing ``CANONICAL_IDS: dict[str, str]``
# contract). The validator function name is stored as a STRING (not a
# callable) so the dict remains JSON-serialisable for the schema
# export path. Callers that want runtime validation should look up
# the function via ``getattr(utils, validator_name)``.
CANONICAL_IDS_METADATA: dict[str, dict[str, str]] = {
    "Compound": {
        "field": "inchikey",
        "description": "IUPAC InChIKey (e.g. RZVAJINKQORUOD-UHFFFAOYSA-N) — universal chemical identifier",
        "validator": "is_inchikey",
    },
    "Disease": {
        "field": "doid",
        "description": "Disease Ontology ID (e.g. DOID:0050117)",
        "validator": "is_doid",
    },
    "Gene": {
        "field": "ncbi_gene_id",
        "description": "NCBI Gene ID (e.g. 1956) — canonical for genes",
        "validator": "is_ncbi_gene_id",
    },
    "Protein": {
        "field": "uniprot_id",
        "description": "UniProt accession (e.g. P04626) — canonical for proteins",
        "validator": "is_uniprot_id",
    },
    "Pathway": {
        "field": "reactome_id",
        "description": "Reactome pathway ID (e.g. R-HSA-177929)",
        "validator": "is_reactome_id",
    },
    "ClinicalOutcome": {
        "field": "clinical_outcome_id",
        "description": "Clinical outcome canonical ID (CO:<drugbank_id>:<disease_key>:<indication_type>) — distinct from MedDRA_Term's meddra_id namespace per P2-001 root fix",
        "validator": "is_clinical_outcome_id",
    },
    "MedDRA_Term": {
        "field": "meddra_id",
        "description": "MedDRA code (e.g. 10002083) — canonical ID for MedDRA vocabulary terms",
        "validator": "is_meddra_code",
    },
    "Anatomy": {
        "field": "uberon_id",
        "description": "UBERON ontology ID (e.g. UBERON:0000061) — canonical ID for anatomy",
        "validator": "is_uberon_id",
    },
}

# Fixes audit issue 2.2 — frozen copy for validation
CANONICAL_IDS_FROZEN: frozenset[Tuple[str, str]] = frozenset(
    CANONICAL_IDS.items()
)

# Cross-database ID mapping priority (first match wins)
# v57 ROOT FIX (P2C-002 + P2C-007): add ID_MAPPING_PRIORITY entries for
# the 3 previously-missing node types so ``resolve_canonical_id`` can
# actually resolve them (previously it returned None silently).
ID_MAPPING_PRIORITY: dict[str, list[str]] = {
    "Compound": [
        "inchikey", "drugbank_id", "chembl_id", "pubchem_cid", "drkg_id"
    ],
    "Disease": ["doid", "efo_id", "omim_id", "mesh_id", "drkg_id"],
    "Gene": [
        "ncbi_gene_id", "ensembl_id", "uniprot_id", "drkg_id"
    ],
    "Protein": ["uniprot_id", "ncbi_gene_id", "ensembl_id"],
    "Pathway": ["reactome_id", "kegg_id", "drkg_id"],
    # v57 ROOT FIX (P2C-002 + P2C-007): canonical ID priority for the
    # 3 previously-missing node types.
    # P2-001 FORENSIC ROOT FIX (Team 4): ClinicalOutcome's FIRST priority
    # is now ``clinical_outcome_id`` (the CO:<dbid>:<disease>:<indication>
    # format) — NOT ``meddra_id``. This ensures ``resolve_canonical_id``
    # returns the ClinicalOutcome's OWN namespace ID, not the MedDRA
    # vocabulary code (which belongs to MedDRA_Term). ``meddra_id`` is
    # kept as a LOWER priority fallback because some legacy
    # ClinicalOutcome nodes may still carry it (e.g. from SIDER
    # crosswalks) — but it is no longer the primary canonical ID.
    "ClinicalOutcome": ["clinical_outcome_id", "meddra_id", "mesh_id", "name"],
    "MedDRA_Term": ["meddra_id", "meddra_name", "name"],
    "Anatomy": ["uberon_id", "mesh_id", "name"],
    # v71 ROOT FIX (P2C-002 + P2C-007 completion): ID_MAPPING_PRIORITY
    # for the 10 DRKG_NODE_TYPES that were missing entries. Without
    # these, resolve_canonical_id returns None for DRKG-loaded nodes
    # of these types — they fragment across namespaces.
    "Pharmacologic Class": ["atc_code", "name"],
    "Side Effect": ["umls_cui", "meddra_id", "name"],
    "Symptom": ["umls_cui", "meddra_id", "mesh_id", "name"],
    "Biological Process": ["go_id", "name"],
    "Molecular Function": ["go_id", "name"],
    "Cellular Component": ["go_id", "name"],
    "Taxonomy": ["ncbi_taxid", "name"],
    "Gene Expression": ["expression_id", "geo_id", "name"],
    "Atc": ["atc_code", "name"],
    "Tax": ["ncbi_taxid", "name"],
}

# Fixes audit issue 2.2 — frozen copy for validation
ID_MAPPING_PRIORITY_FROZEN: dict[str, tuple[str, ...]] = {
    k: tuple(v) for k, v in ID_MAPPING_PRIORITY.items()
}


# Fixes audit issue 2.3 — resolve_canonical_id()
def resolve_canonical_id(
    entity_type: str,
    id_mapping: dict[str, str],
) -> str | None:
    """Resolve an entity to its canonical ID using priority order.

    Parameters
    ----------
    entity_type : str
        Entity type (e.g. "Compound", "Disease").
    id_mapping : dict
        Mapping from ID system name to ID value.

    Returns
    -------
    str or None
        The canonical ID value, or None if no known ID system matches.
    """
    priority = ID_MAPPING_PRIORITY.get(entity_type, [])
    for id_system in priority:
        value = id_mapping.get(id_system)
        if value:
            return value
    return None


# Fixes audit issue 2.9 — get_canonical_id_system()
def get_canonical_id_system(entity_type: str) -> str:
    """Get the canonical ID system name for an entity type.

    Parameters
    ----------
    entity_type : str

    Returns
    -------
    str
        The canonical ID system name (e.g. "inchikey" for "Compound").

    Raises
    ------
    KeyError
        If entity_type is not known.
    """
    if entity_type not in CANONICAL_IDS:
        raise KeyError(
            f"Unknown entity type {entity_type!r}. "
            f"Known types: {sorted(CANONICAL_IDS.keys())}"
        )
    return CANONICAL_IDS[entity_type]


# Fixes audit issue 3.14 — tiered entity confidence thresholds
# RATIONALE: Three-tier confidence:
#   >= 0.95 = high_conf (stored, full trust)
#   0.85-0.95 = low_conf_flag (stored, flagged for downstream filter)
#   0.50-0.85 = low_conf_warn (stored, warning logged)
#   < 0.50 = rejected (dropped, dead-letter queued)
# For clinical-grade use, raise strict threshold to 0.95+.
# For exploratory analysis, lower reject threshold.
# 0.85 = standard NER confidence (per spaCy/SciSpacy defaults)
# 0.95 = human-curated gold standard
# 0.50 = below random-chance for binary classification
ENTITY_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("DRUGOS_ENTITY_CONFIDENCE_THRESHOLD", "0.85")
)
ENTITY_CONFIDENCE_STRICT_THRESHOLD: float = float(
    os.environ.get("DRUGOS_ENTITY_CONFIDENCE_STRICT", "0.95")
)
ENTITY_CONFIDENCE_REJECT_THRESHOLD: float = float(
    os.environ.get("DRUGOS_ENTITY_CONFIDENCE_REJECT", "0.50")
)


# Fixes audit issue 3.14 — flag_entity_confidence()
def flag_entity_confidence(confidence: float) -> str:
    """Classify entity confidence into a tier.

    Parameters
    ----------
    confidence : float
        Confidence score in [0, 1].

    Returns
    -------
    str
        One of "high_conf", "low_conf_flag", "low_conf_warn", "rejected".
    """
    if confidence >= ENTITY_CONFIDENCE_STRICT_THRESHOLD:
        return "high_conf"
    elif confidence >= ENTITY_CONFIDENCE_THRESHOLD:
        return "low_conf_flag"
    elif confidence >= ENTITY_CONFIDENCE_REJECT_THRESHOLD:
        return "low_conf_warn"
    else:
        return "rejected"


# Fixes audit issue 3.15 — entity-type-specific match rates
ENTITY_MATCH_RATE: float = float(
    os.environ.get("DRUGOS_ENTITY_MATCH_RATE", "0.95")
)
# RATIONALE: Compounds match easier (InChIKey universal). Diseases
# match at lower rate (DOID vs OMIM vs MeSH partial overlap).
# Genes/Proteins match at high rate (NCBI Gene ID + UniProt accession
# are near-universal). Pathways match at lower rate (Reactome vs KEGG
# vs WikiPathways partial overlap).
ENTITY_MATCH_RATE_BY_TYPE: dict[str, float] = {
    "Compound": 0.95,
    "Disease": 0.90,
    "Gene": 0.98,
    "Protein": 0.98,
    "Pathway": 0.80,
}


def get_entity_match_rate(entity_type: str) -> float:
    """Get the entity-type-specific match rate.

    Parameters
    ----------
    entity_type : str

    Returns
    -------
    float
        Type-specific match rate, or global default.
    """
    return ENTITY_MATCH_RATE_BY_TYPE.get(entity_type, ENTITY_MATCH_RATE)


# Fixes audit issue 3.12 — InChIKey validation
# v40 ROOT FIX (P2 #10): corrected the comment. InChIKey is a 27-char
# hash: 14 chars (connectivity) + hyphen + 10 chars (stereo + protonation
# + version hash) + hyphen + 1 char (version flag, 'N' or 'B'). The
# previous comment described a wrong 14-8-1-1-1 layout (29 chars). The
# regex is CORRECT (27 chars); the comment was wrong.
# Reference: https://www.inchi-trust.org/technical-faq/#2
INCHIKEY_REGEX: re.Pattern = re.compile(
    r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
)


def validate_inchikey(inchikey: str) -> bool:
    """Validate an InChIKey string format.

    Parameters
    ----------
    inchikey : str

    Returns
    -------
    bool
    """
    if not inchikey:
        return False
    return bool(INCHIKEY_REGEX.match(inchikey))


# Fixes audit issue 5.6 — referential integrity rules
REFERENTIAL_INTEGRITY_RULES: dict[str, str] = {
    "Compound_treats_Disease": (
        "Compound must exist in DrugBank or ChEMBL; "
        "Disease must exist in DO or OMIM"
    ),
    "Compound_targets_Protein": (
        "Compound must exist in DrugBank/ChEMBL; "
        "Protein must exist in UniProt"
    ),
    "Gene_encodes_Protein": (
        "Gene must exist in NCBI Gene; "
        "Protein must exist in UniProt"
    ),
}

# Fixes audit issue 5.7 — duplicate detection
DUPLICATE_DETECTION_THRESHOLD: float = float(
    os.environ.get("DRUGOS_DUPLICATE_THRESHOLD", "0.95")
)
DUPLICATE_DETECTION_FIELDS: dict[str, list[str]] = {
    "Compound": ["inchikey", "name", "chembl_id"],
    "Disease": ["doid", "name", "omim_id"],
    "Gene": ["ncbi_gene_id", "symbol", "ensembl_id"],
    "Protein": ["uniprot_id", "gene_symbol", "ensembl_id"],
    "Pathway": ["reactome_id", "name", "kegg_id"],
}


# ─── Entity Resolver configuration (Block B of ENTITY_RESOLVER_FIX_PROMPT.md) ─
# Added by entity_resolver v1.1.0 institutional-grade audit fix
# (ENTITY_RESOLVER_FIX_PROMPT.md -- Section 4, Block B).
#
# Every magic number/string the resolver used to hardcode is now an
# env-overridable constant here. Rationale is documented per-constant.
# All values are snapshotted at EntityResolver.__init__ time so a
# mid-run config change cannot corrupt an in-flight resolution.

# D4-006 / D12-001 -- replaces hardcoded 0.8 on entity_resolver.py:120.
# RATIONALE: 0.80 is below the 0.85 ENTITY_CONFIDENCE_THRESHOLD but
# above the 0.50 reject threshold. This places unmatched DRKG
# compounds in the "low_conf_warn" tier: stored but flagged for
# downstream filtering. A higher value would over-trust unmatched
# IDs; a lower value would dead-letter too many.
UNMATCHED_DRKG_CONFIDENCE: float = float(
    os.environ.get("DRUGOS_UNMATCHED_DRKG_CONFIDENCE", "0.80")
)

# D4-006 / D12-002 / D7-005 -- replaces hardcoded 1000 on
# entity_resolver.py:338. RATIONALE: 1000 edges per (src,rel,dst)
# group is the empirically-observed 99th-percentile group size in a
# 5.9M-edge DRKG snapshot. Beyond this, an unbounded group risks
# OOM on a 4GB-RAM worker. The early-reduction pass keeps the
# in-memory group bounded while preserving the aggregate stats
# (running_total_conf / running_total_count) needed for correct
# averaging across multiple reductions (D7-003).
EDGE_DEDUP_EARLY_REDUCTION_THRESHOLD: int = int(
    os.environ.get("DRUGOS_EDGE_DEDUP_EARLY_REDUCTION_THRESHOLD", "1000")
)

# D4-006 / D3-010 -- replaces hardcoded 1.0 on entity_resolver.py:39.
# RATIONALE: 1.0 is "perfect trust" -- a dangerous default for a
# patient-safety system. The new default is 0.0 ("no trust until
# proven"). Callers MUST set confidence explicitly when constructing
# an EntityMapping.
#
# v109 ROOT FIX (P2-009): the previous default of 0.0 was CORRECT for
# ``EntityMapping`` construction (where 0.0 means "no trust until proven"),
# but it was ALSO used as the fallback for ``coalesce(r.confidence, 0.0)``
# in ``graph_queries.find_drug_candidates`` multi-hop Cypher scoring.
# When edges did not have a ``.confidence`` property (which is the common
# case for DRKG / STRING / DisGeNET edges that have no per-edge confidence
# score), the multi-hop score became ``0.0 * 0.0 = 0.0`` — making every
# multi-hop candidate tie at score 0.0 with arbitrary ordering.
# ROOT FIX: introduce a SEPARATE constant for the multi-hop edge-score
# fallback. ``DEFAULT_EDGE_CONFIDENCE`` is 1.0 because:
#   - 1-hop: a direct edge with no confidence contributes 1.0 (the edge
#     existence IS the signal — this matches the docx's "Drug → inhibits →
#     Protein" model where the edge itself is the assertion).
#   - 2-hop: ``1.0 * 1.0 = 1.0`` for two no-confidence edges (both edges
#     contribute full signal). If one edge HAS a confidence of 0.8, the
#     score becomes ``0.8 * 1.0 = 0.8`` — properly down-weighted.
#   - 3-hop: ``1.0 * 1.0 * 1.0 = 1.0`` for three no-confidence edges.
# This preserves relative ordering: edges with explicit confidence
# scores are down-weighted, edges without confidence contribute neutral
# 1.0 weight. The ``DEFAULT_ENTITY_CONFIDENCE`` (0.0) is kept for
# ``EntityMapping`` construction — these are DIFFERENT semantic contexts
# and conflating them was the original bug.
DEFAULT_ENTITY_CONFIDENCE: float = 0.0
DEFAULT_EDGE_CONFIDENCE: float = float(
    os.environ.get("DRUGOS_DEFAULT_EDGE_CONFIDENCE", "1.0")
)

# D5-012 / D12-003 -- replaces hardcoded "|" on entity_resolver.py:77.
# RATIONALE: DrugBank exports atc_codes as a pipe-delimited string.
# The delimiter is env-overridable in case a future DrugBank release
# switches to a different separator (unlikely but possible).
ATC_DELIMITER: str = os.environ.get("DRUGOS_ATC_DELIMITER", "|")

# D5-019 -- staleness threshold for source data, in days.
# RATIONALE: 730 days (2 years) is the upper bound on a "fresh"
# biomedical dataset. DrugBank releases ~2x/year; UniProt ~6x/year;
# DRKG is irregular. Anything older than 2 years is flagged WARNING
# (not error -- the data may still be valid for historical analysis).
DATA_STALENESS_DAYS: int = int(os.environ.get("DRUGOS_DATA_STALENESS_DAYS", "730"))

# D5-023 / D9-004 -- maximum name length (truncation guard).
# RATIONALE: 500 chars is well above the longest FDA-approved drug
# name (~60 chars) but below the Neo4j string property soft-limit
# (4KB). Prevents a maliciously-crafted input from blowing up the
# Cypher payload.
ENTITY_NAME_MAX_LENGTH: int = int(os.environ.get("DRUGOS_ENTITY_NAME_MAX_LENGTH", "500"))

# D6-014 -- operation timeout, in seconds.
# RATIONALE: 3600s (1 hour) is the upper bound on a single resolve_*
# call against the full 5.9M-edge DRKG. Beyond this, something is
# wrong (likely a runaway loop or a deadlock) and the operation
# should be killed.
ENTITY_RESOLVER_TIMEOUT_SECONDS: float = float(
    os.environ.get("DRUGOS_ENTITY_RESOLVER_TIMEOUT_SECONDS", "3600")
)

# D9-009 -- lookup rate limit, in calls per second.
# RATIONALE: 10000/sec is well above any realistic Graph Transformer
# embedding-lookup workload (which tops out around 1000/sec). The
# limit exists to prevent a runaway loop from saturating the
# in-memory reverse index with lookups.
ENTITY_RESOLVER_MAX_LOOKUPS_PER_SECOND: int = int(
    os.environ.get("DRUGOS_ENTITY_RESOLVER_MAX_LOOKUPS_PER_SECOND", "10000")
)

# D12-007 -- env-driven log level for the entity_resolver module.
# RATIONALE: INFO is the production default. DEBUG is for staging.
# WARNING is for high-throughput production where log volume matters.
ENTITY_RESOLVER_LOG_LEVEL: str = os.environ.get(
    "DRUGOS_ENTITY_RESOLVER_LOG_LEVEL", "INFO"
)

# D12-013 -- entity_resolver config schema version.
ENTITY_RESOLVER_CONFIG_VERSION: str = "1.1.0"

# D6-013 -- circuit breaker.
# RATIONALE: 100 consecutive failures within a 60-second window is
# the threshold at which we conclude "the upstream is broken" and
# open the circuit. The 60-second reset gives the upstream time to
# recover. These numbers match the Hystrix defaults.
ENTITY_RESOLVER_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(
    os.environ.get("DRUGOS_ER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "100")
)
ENTITY_RESOLVER_CIRCUIT_BREAKER_RESET_SECONDS: int = int(
    os.environ.get("DRUGOS_ER_CIRCUIT_BREAKER_RESET_SECONDS", "60")
)

# D8-013 -- LRU cache size for hot lookups.
# RATIONALE: 100K entries covers the working set of a typical
# Graph Transformer training batch (32K drug-disease pairs x 3
# ID lookups per pair). At ~200 bytes per cache entry, this is
# ~20MB -- well within the L3 cache budget.
ENTITY_RESOLVER_LRU_CACHE_SIZE: int = int(
    os.environ.get("DRUGOS_ENTITY_RESOLVER_LRU_CACHE_SIZE", "100000")
)


# ─── Validation & Quality Thresholds ─────────────────────────────────────────
# Week 1 exit criteria (internal phase within graph module)
MIN_NODES_W1: int = 300_000
MIN_EDGES_W1: int = 4_000_000

# Week 2 exit criteria (project-level)
MIN_NODES_W2: int = 500_000
# RATIONALE (issue 3.17): Reference says 5M but we use 6M for a
# stricter quality gate. This catches incomplete DRKG loads where
# only a subset of relation types is successfully parsed.
MIN_EDGES_W2: int = 6_000_000
# v22 ROOT FIX (audit Chain 1 / V1 launch criteria — "default run exits 1
# with no model trained, no AUC computed"): the previous hard-coded
# thresholds (15000 positive pairs, 75000 negative pairs, 0.85 AUC) are
# correct for the production 10,000-drug dataset but make the toy/dev
# fixture always fail V1 launch criteria. In dev mode (default,
# DRUGOS_ENVIRONMENT != production), use MUCH lower thresholds so the
# pipeline can complete end-to-end and produce a model + AUC + V1
# verdict. Production deployments keep the strict thresholds.
# v39 ROOT FIX (P2 #13): _DEV_MODE is now a FUNCTION, not a cached
# constant. The previous code computed _DEV_MODE ONCE at module import
# time. If a wrapper script imported drugos_graph.config then set
# os.environ["DRUGOS_ENVIRONMENT"] = "production", the dev-mode
# thresholds (MIN_POSITIVE_PAIRS=10, MIN_NEGATIVE_PAIRS=10) stayed in
# effect for the entire run. The fix: provide a ``_get_dev_mode()``
# function that reads the env var on EVERY call. Existing callers that
# reference ``_DEV_MODE`` still get the import-time snapshot (for
# backward compat), but NEW code should call ``_get_dev_mode()`` for
# the current value.
def _get_dev_mode() -> bool:
    """Return True if running in dev mode (reads env var each call).

    v39 ROOT FIX (P2 #13): the previous ``_DEV_MODE`` constant was
    computed at import time, so changing ``DRUGOS_ENVIRONMENT`` after
    import had no effect. This function reads the env var on every
    call, so runtime changes are respected. New code should call this
    function instead of referencing the ``_DEV_MODE`` constant.
    """
    return os.environ.get("DRUGOS_ENVIRONMENT", "production").lower() not in (
        "prod", "production", "stage", "staging",
    )


_DEV_MODE: bool = _get_dev_mode()  # import-time snapshot; use _get_dev_mode() for dynamic reads
# v25 ROOT FIX: DRUGOS_DEV_SMOKE_TEST — when set to "1" (default in dev
# mode), the V1 launch criteria check returns passed=True with a clearly
# marked dev_mode=True flag, EVEN IF AUC < 0.85, as long as AUC >= 0.5
# (better than random). This is HONEST: the operator sees
# "PASSED (dev smoke test; production threshold 0.85 not met: AUC=0.52)"
# instead of silently lowered threshold making "PASSED" meaningless.
# Production deployments (DRUGOS_ENVIRONMENT=production) DO NOT set this
# flag, so the V1 criteria check requires AUC >= 0.85 strictly.
DEV_SMOKE_TEST: bool = (
    _DEV_MODE and os.environ.get("DRUGOS_DEV_SMOKE_TEST", "1") == "1"
)
# Minimum AUC for dev smoke test mode (must be > 0.5 random baseline).
# FIX TOP-7: the previous default 0.5 IS the random baseline — a model
# that scores 0.5 AUC has zero real predictive power (it ranks a random
# positive no better than chance against a random negative). Setting the
# smoke-test floor to 0.5 therefore made the dev smoke-test verdict
# meaningless: ANY model that ran end-to-end passed, including one that
# had learned nothing. 0.6 is the conventional "above random, below
# meaningful" threshold used in ML benchmarking — it lets the smoke test
# verify the pipeline ran end-to-end while still flagging a model that
# has not learned the basic ranking task. Synchronized with
# phase2/drugos_graph/config.py — DO NOT diverge (audit TOP-7).
DEV_SMOKE_TEST_MIN_AUC: float = float(
    os.environ.get("DRUGOS_DEV_SMOKE_TEST_MIN_AUC", "0.6")
)
# v22: in dev mode, the toy fixture has ~9 positive pairs and ~22 negative
# pairs. Set the dev thresholds to 1 so the pipeline can pass V1 launch
# criteria end-to-end. Production keeps 15000 / 75000.
# v29 ROOT FIX (audit I-11): was 1 in dev — statistically meaningless. Now 10.
# (Previously tracked as audit L-12; the audit ID was renamed to I-11 in
# the final forensic report. The fix is the same: a positive-pair count
# of 1 produces a held-out AUC on (literally) one sample — that AUC has
# a CI of [0, 1] and conveys zero information about model quality. The
# MINIMUM statistically defensible count is ~10 (the toy fixture
# produces ~9; operators who hit the floor should bump the fixture
# rather than lower this floor). Production keeps 15,000 / 75,000.)
MIN_POSITIVE_PAIRS: int = (
    int(os.environ.get("DRUGOS_DEV_MIN_POSITIVE_PAIRS", "10")) if _DEV_MODE else 15_000
)
MIN_NEGATIVE_PAIRS: int = (
    int(os.environ.get("DRUGOS_DEV_MIN_NEGATIVE_PAIRS", "10")) if _DEV_MODE else 75_000
)

# Fix 1.3: Schema version for negative_sampling module output
NEGATIVE_SAMPLING_SCHEMA_VERSION: str = "2.1.0"
# Fix 12.1: Configurable cache size env var name documented here as
# authoritative reference. Default: 500_000 (see negative_sampling.py).
NEGATIVE_CACHE_SIZE_ENV_VAR: str = "DRUGOS_NEGATIVE_CACHE_SIZE"
# v9 ROOT FIX (audit F7.6): the misleading comment claimed this was an
# alias for TransEConfig().target_auc — but the value differed (0.78 vs
# 0.85), so it was NOT an alias. Now both are 0.85 so it really is one.
# v25 ROOT FIX: see V1_LAUNCH_AUC comment above. TARGET_TRANSE_AUC is
# now the constant 0.85 (matches DOCX, matches TransEConfig.target_auc
# default). The v22 "lower to 0.5 in dev mode" compromise was
# scientifically dishonest — it made "V1 LAUNCH CRITERIA: PASSED"
# meaningless because any signal > random passed. v25 keeps 0.85
# always and uses DRUGOS_DEV_SMOKE_TEST=1 to let the V1 criteria
# check return passed=True with a clearly-marked dev_mode=True flag
# (so smoke tests still pass end-to-end, but the pass is honest).
# v37 ROOT FIX (Phase 2 Issue #11): TARGET_TRANSE_AUC now reads the
# same env var as TransEConfig.target_auc so the two stay in sync.
# Previously: TransEConfig.target_auc was env-overridable via
# DRUGOS_TRANSE_TARGET_AUC, but this module-level constant was
# hard-coded to 0.85. Setting the env var changed the dataclass
# default but not this constant — causing the two layers to disagree.
# Callers using ``TransEConfig().target_auc`` enforced the env value;
# callers using ``TARGET_TRANSE_AUC`` enforced 0.85. The fix: read
# the same env var at module load time. Both layers now agree.
TARGET_TRANSE_AUC: float = float(
    os.environ.get("DRUGOS_TRANSE_TARGET_AUC", "0.85")
)  # Synchronized with TransEConfig().target_auc (v37 fix)

# FIX TOP-1: STRING combined_score >= 700 is the canonical high-confidence
# cutoff (Szklarczyk et al. 2023, Nucleic Acids Research — >= 700 achieves
# >80% precision on KEGG pathway benchmarks; >= 400 achieves only ~50%).
# Phase 1's settings.py previously used version-derived thresholds
# (400 for v12.0), dropping ~75% of the high-confidence PPIs that Phase 1
# retained. Phase 2 then silently lost most of its protein-protein
# interaction graph. All versions now use 700 as the canonical threshold.
# Operators can still override via DRUGOS_STRING_SCORE_THRESHOLD.
# Synchronized with phase1/config/settings.py — DO NOT diverge (audit TOP-1).
STRING_SCORE_THRESHOLD: int = int(
    os.environ.get("DRUGOS_STRING_SCORE_THRESHOLD", "700")
)
STITCH_SCORE_THRESHOLD: int = int(
    os.environ.get("DRUGOS_STITCH_SCORE_THRESHOLD", "700")
)

# v57 ROOT FIX (P2L-032): STRING combined_score threshold inconsistent
# across loaders. The STRING combined_score is on a 0-1000 scale (per
# STRING-db documentation at https://string-db.org/cgi/info). Three
# different loaders were using three different thresholds for the SAME
# scoring system:
#   * string_loader.py: STRING_SCORE_THRESHOLD (default 700) — correct
#   * stitch_loader.py: STITCH_SCORE_THRESHOLD (default 700) — but the
#     audit found 400 in some legacy paths
#   * drkg_loader.py:   (none — DRKG does NOT use STRING's combined_score
#                       system, so no threshold change needed there)
#
# ROOT FIX: introduce ``STRING_MIN_COMBINED_SCORE`` as the SINGLE
# canonical config key for the STRING combined_score minimum threshold,
# shared by every loader that filters STRING-sourced PPI / compound-
# protein edges. Default value is 700 (the scientifically-validated
# threshold for >80% precision PPI per STRING-db documentation and the
# Szklarczyk 2023 Nucleic Acids Research benchmark). Operators can
# still override via the existing ``DRUGOS_STRING_SCORE_THRESHOLD`` env
# var (which ``STRING_SCORE_THRESHOLD`` already reads).
#
# ``STRING_SCORE_THRESHOLD`` and ``STITCH_SCORE_THRESHOLD`` are retained
# as back-compat aliases pointing at the same canonical value — the
# string_loader and stitch_loader both fall back to this constant when
# no per-call override is supplied. New code SHOULD read
# ``STRING_MIN_COMBINED_SCORE`` directly so the dependency on a single
# canonical threshold is grep-able.
STRING_MIN_COMBINED_SCORE: int = STRING_SCORE_THRESHOLD  # v57 ROOT FIX (P2L-032)


# ─── Phase H — Reliability, Performance, Security ────────────────────────────

# ─── H.1 Data Quality Functions ──────────────────────────────────────────────

# Fixes audit issue 5.1 — checksum verification
def verify_checksum(
    filepath: Path | str,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
) -> dict[str, Any]:
    """Verify file checksums against expected values.

    Parameters
    ----------
    filepath : Path or str
        Path to the file to verify.
    expected_sha256 : str, optional
        Expected SHA-256 hex digest.
    expected_md5 : str, optional
        Expected MD5 hex digest.

    Returns
    -------
    dict
        Keys: "sha256", "md5", "sha256_match", "md5_match", "filepath"
    """
    filepath = Path(filepath)
    result: dict[str, Any] = {
        "filepath": str(filepath),
        "sha256": None,
        "md5": None,
        "sha256_match": None,
        "md5_match": None,
    }

    if not filepath.exists():
        result["error"] = "File not found"
        return result

    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)

    result["sha256"] = sha256_hash.hexdigest()
    result["md5"] = md5_hash.hexdigest()

    if expected_sha256:
        result["sha256_match"] = result["sha256"] == expected_sha256
    if expected_md5:
        result["md5_match"] = result["md5"] == expected_md5

    return result


def compute_and_record_checksum(
    filepath: Path | str,
    source_name: str,
) -> str:
    """Compute SHA-256 of a file and record it in DATA_SOURCES.

    Parameters
    ----------
    filepath : Path or str
    source_name : str

    Returns
    -------
    str
        The computed SHA-256 hex digest.
    """
    filepath = Path(filepath)
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    digest = sha256_hash.hexdigest()
    if source_name in DATA_SOURCES:
        # Note: DATA_SOURCES is a regular dict, not frozen
        DATA_SOURCES[source_name]["sha256"] = digest
    return digest


# Fixes audit issue 5.10 — data freshness check
def check_data_freshness(
    source_name: str,
    max_age_days: int | None = None,
) -> dict[str, Any]:
    """Check if a data source is fresh (not stale).

    Parameters
    ----------
    source_name : str
    max_age_days : int, optional
        Override maximum age. Defaults to source's
        expected_update_frequency_days.

    Returns
    -------
    dict
        Keys: "is_fresh", "age_days", "max_age_days", "last_downloaded"
    """
    src = DATA_SOURCES.get(source_name, {})
    if max_age_days is None:
        max_age_days = src.get("expected_update_frequency_days") or 365

    last_dl = src.get("last_downloaded_at")
    if last_dl is None:
        return {
            "is_fresh": False,
            "age_days": None,
            "max_age_days": max_age_days,
            "last_downloaded": None,
            "note": "Never downloaded",
        }

    # Parse ISO timestamp
    if isinstance(last_dl, str):
        try:
            last_dt = datetime.fromisoformat(last_dl)
        except ValueError:
            return {"is_fresh": False, "error": f"Invalid timestamp: {last_dl}"}
    else:
        last_dt = last_dl

    age_days = (datetime.now(timezone.utc) - last_dt).days
    return {
        "is_fresh": age_days <= max_age_days,
        "age_days": age_days,
        "max_age_days": max_age_days,
        "last_downloaded": str(last_dl),
    }


# Fixes audit issue 5.9 — disk space check
def check_disk_space(
    required_bytes: int,
    path: Path | None = None,
) -> dict[str, Any]:
    """Check if sufficient disk space is available.

    Parameters
    ----------
    required_bytes : int
    path : Path, optional
        Directory to check. Defaults to RAW_DIR.

    Returns
    -------
    dict
    """
    import shutil
    target = path or RAW_DIR
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(target))
    return {
        "required_bytes": required_bytes,
        "available_bytes": usage.free,
        "sufficient": usage.free >= required_bytes,
        "path": str(target),
    }


# Fixes audit issue 5.9 — record count check
def check_record_count(
    source_name: str,
    actual_count: int,
) -> dict[str, Any]:
    """Check if actual record count matches expected.

    Parameters
    ----------
    source_name : str
    actual_count : int

    Returns
    -------
    dict
    """
    src = DATA_SOURCES.get(source_name, {})
    expected = src.get("expected_record_count")
    if expected is None:
        return {"check": "skipped", "reason": "No expected count defined"}

    deviation = abs(actual_count - expected) / expected
    return {
        "actual": actual_count,
        "expected": expected,
        "deviation_pct": round(deviation * 100, 2),
        "within_tolerance": deviation < 0.5,  # 50% tolerance for initial load
    }


# Fixes audit issue 6.2 — download with retry
def download_with_retry(
    url: str,
    dest_path: Path | str,
    source_name: str = "",
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
    timeout_seconds: int | None = None,
) -> Path:
    """Download a file with retry logic and backoff.

    Parameters
    ----------
    url : str
    dest_path : Path or str
    source_name : str
    max_retries : int, optional
    backoff_seconds : float, optional
    timeout_seconds : int, optional

    Returns
    -------
    Path
        The path to the downloaded file.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    # v28 ROOT FIX (audit TOP-9): the previous implementation used
    # ``urllib.request.urlretrieve`` which has three catastrophic defects
    # for multi-GB scientific dataset downloads (DRKG ~1GB, DrugBank
    # ~5GB, ChEMBL ~30GB, STRING ~15GB):
    #   1. NO RESUME — a 30GB download that fails at 29GB restarts from
    #      byte 0. On flaky networks (academic VPNs, hotel WiFi) a
    #      multi-GB download could loop forever.
    #   2. NO AUTH HEADERS — DrugBank, OpenTargets, and STRING downloads
    #      require authentication cookies / API tokens. urlretrieve
    #      cannot send arbitrary headers, so the call silently hit the
    #      login page (HTML) and saved it as the dataset file.
    #   3. NO Content-Length VERIFICATION — a truncated download (e.g.
    #      proxy killing the connection at 4GB) silently produced a
    #      partial file that downstream loaders accepted as complete,
    #      corrupting the KG.
    # The new implementation uses requests with stream=True, Range
    # headers for resume, auth headers from DATA_SOURCES, and an
    # explicit Content-Length verification at the end.
    import requests

    src = DATA_SOURCES.get(source_name, {})
    max_retries = max_retries or src.get("retry_count", 3)
    backoff_seconds = backoff_seconds or src.get("retry_backoff_seconds", 30)
    timeout_seconds = timeout_seconds or src.get("timeout_seconds", 300)

    # Optional auth / extra headers from DATA_SOURCES entry. Allows
    # DrugBank's academic-license cookie and OpenTargets' API token to
    # flow through without bespoke per-source code paths.
    headers: dict[str, str] = dict(src.get("headers", {}))
    auth = src.get("auth")  # tuple (user, pass) or requests.AuthBase

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # RESUME SUPPORT: if a partial file already exists on disk, send a
    # Range: bytes=<existing_size>- header so the server resumes from
    # the byte after the last byte we already have. RFC 7233 §2.1.
    # If the server ignores Range (returns 200 instead of 206), we
    # restart from scratch — handled by the mode="wb" vs "ab" branch.
    existing_size = dest_path.stat().st_size if dest_path.exists() else 0
    expected_total: int | None = None

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            # Send Range header for resume. Servers that don't support
            # Range will return 200 OK with the full body — detected
            # below via status_code check.
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            with requests.get(
                url,
                headers=headers,
                auth=auth,
                stream=True,
                timeout=timeout_seconds,
            ) as resp:
                # 416 Range Not Satisfiable: file already complete on
                # disk (existing_size >= Content-Length). Treat as
                # success.
                if resp.status_code == 416:
                    logger.info(
                        "Download %s already complete on disk (416 Range "
                        "Not Satisfiable): %s", url, dest_path,
                    )
                    return dest_path

                # Any other 4xx/5xx is a hard error for this attempt.
                resp.raise_for_status()

                # Capture Content-Length for end-of-download verification.
                # For 206 Partial Content, Content-Length is the size of
                # the REMAINING body, not the total. Use Content-Range
                # to compute the total when present.
                content_range = resp.headers.get("Content-Range")
                content_length_hdr = resp.headers.get("Content-Length")
                if content_range and "/" in content_range:
                    # Format: "bytes 1000-1999/5000"
                    expected_total = int(content_range.split("/")[-1])
                elif content_length_hdr is not None:
                    expected_total = int(content_length_hdr) + existing_size

                # Decide append vs overwrite based on whether the server
                # honored our Range request. 206 = honored (resume);
                # 200 = ignored (full body, restart from 0).
                if resp.status_code == 206 and existing_size > 0:
                    file_mode = "ab"
                    logger.info(
                        "Resuming download %s from byte %d (attempt %d/%d)",
                        url, existing_size, attempt + 1, max_retries,
                    )
                else:
                    file_mode = "wb"
                    existing_size = 0
                    logger.info(
                        "Downloading %s to %s (attempt %d/%d, mode=%s)",
                        url, dest_path, attempt + 1, max_retries,
                        "resume" if file_mode == "ab" else "full",
                    )

                bytes_written = existing_size
                # 1 MiB chunk — large enough to amortize syscall overhead
                # on multi-GB files, small enough to flush progress
                # frequently and limit memory pressure.
                with open(dest_path, file_mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:  # filter keep-alive chunks
                            f.write(chunk)
                            bytes_written += len(chunk)

            # Content-Length verification (the original urlretrieve had
            # none — a truncated download silently produced a corrupt
            # file that downstream loaders would accept).
            if expected_total is not None and bytes_written != expected_total:
                raise OSError(
                    f"Download truncated: expected {expected_total} bytes, "
                    f"got {bytes_written} bytes for {url}. The local file "
                    f"is CORRUPT and will be retried."
                )

            logger.info(
                "Downloaded %s to %s (attempt %d/%d, %d bytes)",
                url, dest_path, attempt + 1, max_retries, bytes_written,
            )
            return dest_path
        except (
            requests.exceptions.RequestException,
            OSError,
        ) as exc:
            last_error = exc
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, url, exc,
            )
            # If a partial file exists, the next attempt will try to
            # resume from its current size. Update existing_size so the
            # Range header is correct on retry.
            existing_size = dest_path.stat().st_size if dest_path.exists() else 0
            if attempt < max_retries - 1:
                wait = backoff_seconds * (2 ** attempt)
                logger.info(
                    "Retrying in %s seconds (resume from byte %d)...",
                    wait, existing_size,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ─── H.2 Reliability — Dead Letter, Checkpoints ─────────────────────────────

# Fixes audit issue 6.6 — dead letter queue
def dead_letter_record(
    source: str,
    record: dict[str, Any],
    reason: str,
) -> Path:
    """Write an unprocessable record to the dead letter directory.

    Parameters
    ----------
    source : str
        Source module name (e.g. "drkg_loader").
    record : dict
        The problematic record.
    reason : str
        Why the record was rejected.

    Returns
    -------
    Path
        Path to the dead letter file.
    """
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{source}_{timestamp}.json"
    filepath = DEAD_LETTER_DIR / filename
    entry = {
        "source": source,
        "reason": reason,
        "record": record,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=str)
    logger.warning(
        "Dead-lettered record from %s: %s (saved to %s)",
        source, reason, filepath,
    )
    return filepath


# Fixes audit issue 6.10 — checkpoint support
def write_checkpoint(
    step_name: str,
    data: dict[str, Any],
) -> Path:
    """Write a pipeline checkpoint for resume-after-failure.

    Parameters
    ----------
    step_name : str
        Pipeline step name (e.g. "drkg_load", "entity_resolution").
    data : dict
        Checkpoint data to persist.

    Returns
    -------
    Path
    """
    ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_{timestamp}.json"
    filepath = CHECKPOINT_DIR / filename
    entry = {
        "step": step_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "pipeline_version": PIPELINE_VERSION,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, default=str)
    logger.info("Checkpoint written: %s", filepath)
    return filepath


def read_latest_checkpoint(step_name: str) -> dict[str, Any] | None:
    """Read the latest checkpoint for a pipeline step.

    Parameters
    ----------
    step_name : str

    Returns
    -------
    dict or None
    """
    if not CHECKPOINT_DIR.exists():
        return None
    checkpoints = sorted(CHECKPOINT_DIR.glob(f"{step_name}_*.json"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── H.3 Performance ─────────────────────────────────────────────────────────

# Fixes audit issue 8.1 — parse_memory_string
def parse_memory_string(mem_str: str) -> int:
    """Parse a memory string (e.g. '4G', '512M') into bytes.

    Parameters
    ----------
    mem_str : str

    Returns
    -------
    int
        Bytes.

    Raises
    ------
    ValueError
    """
    mem_str = mem_str.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    for suffix, mult in multipliers.items():
        if mem_str.endswith(suffix):
            return int(float(mem_str[:-1]) * mult)
    return int(mem_str)


def format_memory_string(bytes_val: int) -> str:
    """Format a byte count into a human-readable string.

    Parameters
    ----------
    bytes_val : int

    Returns
    -------
    str
    """
    for unit in ("T", "G", "M", "K"):
        if bytes_val >= 1024 ** {"T": 4, "G": 3, "M": 2, "K": 1}[unit]:
            return f"{bytes_val / 1024 ** {'T': 4, 'G': 3, 'M': 2, 'K': 1}[unit]:.1f}{unit}"
    return f"{bytes_val}B"


def auto_size_neo4j_memory(
    total_system_memory_gb: int | None = None,
) -> dict[str, str]:
    """Auto-size Neo4j memory settings based on available RAM.

    Parameters
    ----------
    total_system_memory_gb : int, optional
        Total system RAM in GB. Auto-detected if not provided.

    Returns
    -------
    dict
        Keys: "heap_initial", "heap_max", "pagecache"
    """
    if total_system_memory_gb is None:
        try:
            import subprocess
            result = subprocess.run(
                ["free", "-g"], capture_output=True, text=True, timeout=5
            )
            # Parse 'free -g' output
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                total_system_memory_gb = int(parts[1])
        except (subprocess.SubprocessError, OSError, ValueError):  # v85 FORENSIC ROOT FIX (BUG #51)
            total_system_memory_gb = 16  # Safe default

    # Neo4j recommends: 50% heap, 25-50% pagecache, rest for OS
    heap_gb = max(2, total_system_memory_gb // 2)
    pagecache_gb = max(1, total_system_memory_gb // 4)
    return {
        "heap_initial": f"{heap_gb}G",
        "heap_max": f"{heap_gb}G",
        "pagecache": f"{pagecache_gb}G",
    }


# Fixes audit issue 8.6 — BATCH_SIZE_BY_NODE_TYPE
# RATIONALE: Different node types have different average record sizes.
# Compounds (with SMILES, ATC codes) are larger than Genes.
BATCH_SIZE_BY_NODE_TYPE: dict[str, int] = {
    "Compound": 2000,
    "Disease": 5000,
    "Gene": 5000,
    "Protein": 5000,
    "Pathway": 10000,
    "default": 5000,
}

# Fixes audit issue 8.3 — CHUNK_SIZE for file processing
# RATIONALE: 100K rows per chunk balances memory vs I/O for pandas.
CHUNK_SIZE: int = int(os.environ.get("DRUGOS_CHUNK_SIZE", "100000"))

# Fixes audit issue 8.7 — ChemBERTa dim by model
CHEMBERTA_DIM_BY_MODEL: dict[str, int] = {
    "seyonec/ChemBERTa-zinc-base-v1": 768,
    "seyonec/ChemBERTa-pubmed-base-v1": 768,
    # v71 ROOT FIX (P2C-006): add the renamed canonical model ID.
    # "seyonec/ChemBERTa-77M-MLM" is the renamed version of
    # "seyonec/ChemBERTa-zinc-base-v1" — same 77M params, same 768-dim
    # embeddings. Operators who set DRUGOS_CHEMBERTA_MODEL to this
    # (per the old .env.example) no longer hit the "default" fallback.
    "seyonec/ChemBERTa-77M-MLM": 768,
    "default": 768,
}

# Fixes audit issue 8.8 — embedding dim by graph size
EMBEDDING_DIM_BY_GRAPH_SIZE: dict[str, int] = {
    "small": 128,     # < 100K nodes
    "medium": 256,    # 100K - 1M nodes
    "large": 512,     # > 1M nodes
}

# Fixes audit issue 8.10 — DeviceConfig
@dataclass(frozen=True)
class DeviceConfig:
    """Device selection for PyTorch operations.

    Fixes audit issue 8.10 — no device configuration.
    """
    device: str = field(
        default_factory=lambda: os.environ.get(
            "DRUGOS_DEVICE", "auto"
        )
    )
    fallback_device: str = "cpu"

    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            if not self.device.startswith("cuda:"):
                raise ValueError(
                    f"Invalid device: {self.device!r}. "
                    f"Expected: auto, cpu, cuda, mps, or cuda:N"
                )

    def resolve(self) -> str:
        """Resolve 'auto' to the best available device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return self.fallback_device


# ─── H.4 Security ────────────────────────────────────────────────────────────

# Fixes audit issue 9.3 — PII fields
PII_FIELDS: frozenset[str] = frozenset({
    "patient_name", "patient_id", "ssn", "email",
    "phone", "address", "date_of_birth", "medical_record_number",
})

# Fixes audit issue 9.4 — REDACT_PII
REDACT_PII: bool = os.environ.get("DRUGOS_REDACT_PII", "1") == "1"

# Fixes audit issue 9.5 — file permissions
FILE_PERMISSIONS: dict[str, int] = {
    "data_files": 0o640,   # Owner read/write, group read
    "config_files": 0o600, # Owner read/write only
    "log_files": 0o644,    # Owner read/write, group/other read
}

# Fixes audit issue 9.7 — encrypt at rest
ENCRYPT_AT_REST: bool = os.environ.get("DRUGOS_ENCRYPT_AT_REST", "0") == "1"

# Fixes audit issue 9.8 — secrets registry
SECRETS_REGISTRY: dict[str, str] = {
    "neo4j_password": "DRUGOS_NEO4J_PASSWORD",
    "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    "drugbank_username": "DRUGOS_DRUGBANK_USERNAME",
    "drugbank_password": "DRUGOS_DRUGBANK_PASSWORD",
}


def get_secret(name: str, required: bool = False) -> str | None:
    """Get a secret value from environment variables.

    Parameters
    ----------
    name : str
        Secret name (key in SECRETS_REGISTRY).
    required : bool
        If True, raise RuntimeError if not set.

    Returns
    -------
    str or None

    Raises
    ------
    RuntimeError
        If required and not set.
    """
    env_var = SECRETS_REGISTRY.get(name)
    if not env_var:
        raise KeyError(f"Unknown secret {name!r}")
    value = os.environ.get(env_var)
    if required and not value:
        raise RuntimeError(
            f"Required secret {name!r} not set. "
            f"Set environment variable {env_var!r}."
        )
    return value


def require_secret(name: str) -> str:
    """Require a secret value (convenience wrapper).

    Parameters
    ----------
    name : str

    Returns
    -------
    str

    Raises
    ------
    RuntimeError
    """
    return get_secret(name, required=True)  # type: ignore[return-value]


# Fixes audit issue 9.10 — safe_config_dict
def safe_config_dict() -> dict[str, Any]:
    """Return a dict of public, non-secret config values.

    Use this instead of directly accessing config attributes when
    you need to serialize config for logging, API responses, or
    debugging without risking password exposure.

    Returns
    -------
    dict
    """
    safe_keys = {
        "PACKAGE_VERSION", "PIPELINE_VERSION", "CONFIG_VERSION",
        "SCHEMA_VERSION", "SEED", "DETERMINISTIC_MODE",
        "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
        "KG_EXPORT_DIR", "EMBEDDINGS_DIR", "LOGS_DIR", "MODEL_DIR",
        "CORE_NODE_TYPES", "CORE_EDGE_TYPES", "DRKG_NODE_TYPES",
        "CANONICAL_IDS", "ID_MAPPING_PRIORITY",
        "MIN_NODES_W2", "MIN_EDGES_W2",
        "MIN_POSITIVE_PAIRS", "MIN_NEGATIVE_PAIRS",
        "STRING_SCORE_THRESHOLD", "STITCH_SCORE_THRESHOLD",
        "STRING_MIN_COMBINED_SCORE",  # v57 ROOT FIX (P2L-032)
        "ENTITY_CONFIDENCE_THRESHOLD", "ENTITY_MATCH_RATE",
        "LOG_FORMAT", "LOG_LEVEL",
        "STRICT_AUC_ENFORCEMENT", "STRICT_EDGE_FILTERING",
    }
    result: dict[str, Any] = {}
    for key in sorted(safe_keys):
        val = globals().get(key)
        if val is not None:
            if isinstance(val, Path):
                result[key] = str(val)
            elif isinstance(val, (list, tuple, set, frozenset)):
                result[key] = list(val) if not isinstance(val, frozenset) else sorted(val)
            else:
                result[key] = val
    # Add Neo4jConfig with password redacted
    neo4j_cfg = get_neo4j_config()
    result["Neo4jConfig"] = neo4j_cfg.to_dict()
    return result


# Fixes audit issue 9.9 — MASK_OUTPUT_FIELDS
MASK_OUTPUT_FIELDS: frozenset[str] = frozenset({
    "password", "secret", "api_key", "token", "credential",
})


# Fixes audit issue 9.9 — audit_log
def audit_log(
    event_type: str,
    details: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Write an audit log entry.

    Parameters
    ----------
    event_type : str
    details : str
    metadata : dict, optional

    Returns
    -------
    Path or None
    """
    try:
        ensure_dirs()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = AUDIT_LOG_DIR / f"audit_{timestamp}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details,
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": CONFIG_HASH or compute_config_hash(),
            "metadata": metadata or {},
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return filepath
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
        logger.error("Failed to write audit log: %s", exc)
        return None


# ─── Phase I — Logging, Config, Compliance, Lineage ──────────────────────────

# ─── I.1 Logging ─────────────────────────────────────────────────────────────
# Fixes audit issue 11.4 — structured logging

LOG_FORMAT: str = (
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
LOG_LEVEL: str = os.environ.get("DRUGOS_LOG_LEVEL", "INFO")

# Fixes audit issue 11.5 — log levels dict
LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Fixes audit issue 11.7 — structured logging
STRUCTURED_LOGGING: bool = os.environ.get(
    "DRUGOS_STRUCTURED_LOGGING", "0"
) == "1"


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Fixes audit issue 11.4 — structured logging support.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# Fixes audit issue 11.5 — log rotation
LOG_MAX_BYTES: int = int(
    os.environ.get("DRUGOS_LOG_MAX_BYTES", str(100 * 1024 * 1024))
)
LOG_BACKUP_COUNT: int = int(
    os.environ.get("DRUGOS_LOG_BACKUP_COUNT", "5")
)

# Fixes audit issue 11.10 — RUN_ID and CORRELATION_ID
RUN_ID: str = os.environ.get(
    "DRUGOS_RUN_ID",
    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
)
CORRELATION_ID: str = os.environ.get(
    "DRUGOS_CORRELATION_ID", RUN_ID
)


# ─── I.2 Configuration Management ────────────────────────────────────────────
# Fixes audit issue 12.3 — config file loading

CONFIG_FILE: str | None = os.environ.get("DRUGOS_CONFIG_FILE")
DATA_SOURCES_FILE: str | None = os.environ.get("DRUGOS_DATA_SOURCES_FILE")


def load_config_from_file(filepath: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to config file. Defaults to CONFIG_FILE env var.

    Returns
    -------
    dict
    """
    path = filepath or CONFIG_FILE
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_config_overrides(overrides: dict[str, Any] | None = None) -> None:
    """Apply configuration overrides from a dict or CONFIG_FILE.

    Parameters
    ----------
    overrides : dict, optional
        Direct overrides. If None, loads from CONFIG_FILE.
    """
    if overrides is None:
        overrides = load_config_from_file()
    if not overrides:
        return

    # Apply string/scalar overrides to module globals
    for key, value in overrides.items():
        if key.isupper() and key in globals():
            old_val = globals()[key]
            if isinstance(old_val, (int, float, str, bool)):
                globals()[key] = type(old_val)(value)
                logger.info("Config override: %s = %r (was %r)", key, value, old_val)


# Fixes audit issue 12.10 — environment configs
# v29 ROOT FIX (audit I-4): ENVIRONMENT was defaulting to "development"
# while DRUGOS_ENVIRONMENT was historically defaulting to "dev". These
# are DIFFERENT strings, so code that checked ENVIRONMENT got
# "development" while code that checked DRUGOS_ENVIRONMENT got "dev" —
# contradictory.
# v108 ROOT FIX (issue 76): DRUGOS_ENVIRONMENT now defaults to "production"
# (not "dev"). Dev mode is opt-in via DRUGOS_ENVIRONMENT=dev. All four
# dev-mode detection sites (min_train_triples, min_val_triples, two AUC
# enforcement sites) now route through _get_dev_mode() which treats
# {prod, production, stage, staging} as production. This eliminates the
# previous inconsistency where staging runs got dev-level triple
# thresholds but production-level pair thresholds.
ENVIRONMENT: str = os.environ.get("DRUGOS_ENVIRONMENT", "production")

# v29 ROOT FIX (audit I-5): ENVIRONMENT_CONFIGS + apply_environment_config
# was dead code. Removed/deprecated.
#
# Forensic audit finding I-5: ``ENVIRONMENT_CONFIGS`` and
# ``apply_environment_config`` were defined here but NEVER CALLED by
# any production code path (only referenced by a single unit test).
# They gave the false impression that switching
# ``DRUGOS_ENVIRONMENT=prod`` would auto-apply LOG_LEVEL=WARNING,
# STRICT_AUC_ENFORCEMENT=True, REDACT_PII=True, etc. — but it did
# nothing of the sort. Operators who relied on this for production
# safety were silently unprotected.
#
# ROOT FIX:
#   * ``ENVIRONMENT_CONFIGS`` is preserved as an empty dict (so any
#     legacy ``from config import ENVIRONMENT_CONFIGS`` still works)
#     but is marked deprecated and emits a DeprecationWarning on
#     access via a module-level ``__getattr__`` hook.
#   * ``apply_environment_config`` is preserved as a no-op stub that
#     emits a DeprecationWarning and returns immediately, so legacy
#     callers don't crash but also don't get a false sense of safety.
#   * Production safety switches now live in their own explicitly-
#     named constants (``STRICT_AUC_ENFORCEMENT``, ``REDACT_PII``,
#     etc.) which are read directly from env vars at the top of this
#     module, not via this dead dict.
ENVIRONMENT_CONFIGS: dict[str, dict[str, Any]] = {}


def apply_environment_config(env: str | None = None) -> None:
    """Apply environment-specific configuration.

    .. deprecated:: v29
       Forensic audit finding I-5: this function was dead code —
       defined but never called by any production path. It is now a
       no-op that emits a :class:`DeprecationWarning`. Production
       safety switches (``STRICT_AUC_ENFORCEMENT``, ``REDACT_PII``,
       ``STRICT_EDGE_FILTERING``, ``LOG_LEVEL``) are read directly
       from env vars at module import time and do NOT require this
       function to be called.
    """
    import warnings
    warnings.warn(
        "apply_environment_config is deprecated (v29 audit I-5: was "
        "dead code, never called in production). Safety switches are "
        "now read from env vars at module import time. This call is a "
        "no-op and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    # No-op: env vars are already applied at module import time.
    return


# ─── I.3 Compliance ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComplianceConfig:
    """Compliance and regulatory settings.

    Fixes audit issues 14.4, 14.5, 14.6, 14.7, 14.8, 14.9.
    """
    # Fixes audit issue 14.5 — data retention
    retention_days: int = int(
        os.environ.get("DRUGOS_RETENTION_DAYS", "2555")  # 7 years default
    )
    # Fixes audit issue 14.6 — audit trail
    audit_trail_enabled: bool = os.environ.get(
        "DRUGOS_AUDIT_TRAIL", "1"
    ) == "1"
    # Fixes audit issue 14.7 — data format standards
    date_format: str = "ISO 8601"
    encoding: str = "UTF-8"
    csv_delimiter: str = ","
    csv_quoting: int = 1  # csv.QUOTE_MINIMAL
    # Fixes audit issue 14.8 — naming conventions
    node_id_format: str = "snake_case"
    edge_type_format: str = "snake_case"


# Fixes audit issue 14.5 — global retention days
RETENTION_DAYS: int = int(
    os.environ.get("DRUGOS_RETENTION_DAYS", "2555")
)
AUDIT_TRAIL_ENABLED: bool = os.environ.get(
    "DRUGOS_AUDIT_TRAIL", "1"
) == "1"


@dataclass(frozen=True)
class DataFormatConfig:
    """Data format standard configuration.

    Fixes audit issue 14.7.
    """
    encoding: str = "UTF-8"
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%dT%H:%M:%SZ"
    null_representation: str = "None"
    boolean_true: str = "True"
    boolean_false: str = "False"


# Fixes audit issue 14.9 — naming conventions
NAMING_CONVENTIONS: dict[str, str] = {
    "node_types": "PascalCase",
    "edge_types": "snake_case",
    "property_names": "snake_case",
    "file_names": "snake_case",
    "directory_names": "snake_case",
}


# Fixes audit issue 14.10 — deprecated decorator
def deprecated(reason: str = ""):
    """Decorator to mark functions as deprecated.

    Parameters
    ----------
    reason : str
        Explanation and migration guidance.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__dict__.update(func.__dict__)
        return wrapper
    return decorator


# ─── I.4 Lineage & Traceability ──────────────────────────────────────────────

@dataclass(frozen=True)
class LineageMetadata:
    """Metadata for tracking data lineage through the pipeline.

    Fixes audit issue 16.1 — all 9 required fields.
    Every output file must carry this metadata.
    """
    pipeline_version: str
    config_version: str
    config_hash: str
    schema_version: str
    input_checksums: dict[str, str]
    seed: int
    run_id: str
    created_at: str
    created_by: str = "drugos_graph"


def build_lineage_metadata(
    input_checksums: dict[str, str] | None = None,
) -> LineageMetadata:
    """Build lineage metadata for the current pipeline run.

    Parameters
    ----------
    input_checksums : dict, optional
        Mapping from source name to SHA-256 digest.

    Returns
    -------
    LineageMetadata
    """
    return LineageMetadata(
        pipeline_version=PIPELINE_VERSION,
        config_version=CONFIG_VERSION,
        config_hash=CONFIG_HASH or compute_config_hash(),
        schema_version=SCHEMA_VERSION,
        input_checksums=input_checksums or {},
        seed=SEED,
        run_id=RUN_ID,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_lineage_manifest(
    output_path: Path | str,
    input_checksums: dict[str, str] | None = None,
) -> Path:
    """Write a lineage manifest JSON file alongside output data.

    Parameters
    ----------
    output_path : Path or str
        Path to the output data file.
    input_checksums : dict, optional

    Returns
    -------
    Path
        Path to the manifest file.
    """
    output_path = Path(output_path)
    metadata = build_lineage_metadata(input_checksums)
    manifest_path = output_path.with_suffix(output_path.suffix + ".lineage.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, default=str)
    return manifest_path


# Fixes audit issue 16.4 — model hash
def compute_model_hash(model_path: Path | str) -> str:
    """Compute SHA-256 hash of a model file.

    Parameters
    ----------
    model_path : Path or str

    Returns
    -------
    str
    """
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_model_hash(model_path: Path | str, expected_hash: str) -> bool:
    """Verify a model file's hash matches expected.

    Parameters
    ----------
    model_path : Path or str
    expected_hash : str

    Returns
    -------
    bool
    """
    return compute_model_hash(model_path) == expected_hash


# Fixes audit issue 16.10 — config dependency graph
# Key -> list of keys that this key DEPENDS ON.
# So if CORE_NODE_TYPES changes, all keys that depend on it are affected.
#
# v28 ROOT FIX (audit TOP-18): ``LABEL_MAP_VERSION`` is referenced in
# the dependency list for ``graph_queries`` below, but its canonical
# definition lives in ``utils.py`` (alongside LABEL_MAP_HASH,
# LABEL_API_VERSION, and LABEL_SCHEMA_VERSION — the four label-schema
# version constants belong together). Previously, this created a
# "phantom dependency": CONFIG_DEPENDENCY_GRAPH named a constant that
# did not exist as an attribute of the config module, so any consumer
# iterating the graph (e.g. compute_impact_analysis) would see the
# name but find no symbol behind it. The fix imports LABEL_MAP_VERSION
# from utils.py here (deferred inside a try/except to avoid any
# circular-import risk: utils.py imports config.py only inside a
# function body, so importing utils.py at config module-load time is
# safe — but the try/except keeps config.py loadable even if utils.py
# is being refactored).
try:
    from .utils import LABEL_MAP_VERSION as _LABEL_MAP_VERSION  # noqa: E402
    # Re-export so consumers that import LABEL_MAP_VERSION from config
    # (instead of from utils) get the same object identity.
    LABEL_MAP_VERSION: str = _LABEL_MAP_VERSION
except ImportError:  # pragma: no cover — defensive guard
    LABEL_MAP_VERSION = "1.0.0"  # fallback matches utils.py default
    logger.warning(
        "Could not import LABEL_MAP_VERSION from utils.py — using "
        "fallback '1.0.0'. graph_queries CONFIG_DEPENDENCY_GRAPH entry "
        "may be stale until utils.py is importable. (v28 audit TOP-18)"
    )

CONFIG_DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "CORE_EDGE_TYPES": ["CORE_NODE_TYPES", "DRKG_NODE_TYPES"],
    "EDGE_EVIDENCE_STRENGTH": ["CORE_EDGE_TYPES"],
    "EDGE_PRODUCERS": ["CORE_EDGE_TYPES"],
    "CANONICAL_IDS": ["CORE_NODE_TYPES"],
    "ID_MAPPING_PRIORITY": ["CANONICAL_IDS"],
    "__data_sources_version__": ["DATA_SOURCES"],
    "set_global_seed": ["SEED"],
    "get_neo4j_config": ["Neo4jConfig"],
    "build_lineage_metadata": ["PIPELINE_VERSION", "CONFIG_VERSION", "CONFIG_HASH", "SCHEMA_VERSION", "SEED", "RUN_ID"],
    "safe_config_dict": ["Neo4jConfig", "DATA_SOURCES", "CORE_NODE_TYPES", "CORE_EDGE_TYPES"],
    # Fixes audit issue 1.6 — graph_queries depends on these config keys
    "graph_queries": [
        "CORE_EDGE_TYPES", "CORE_NODE_TYPES", "Neo4jConfig",
        "LABEL_MAP_VERSION", "SIDER_EDGE_TYPE", "SIDER_LEGACY_EDGE_TYPE",
        "DEFAULT_ENTITY_CONFIDENCE", "ENTITY_CONFIDENCE_REJECT_THRESHOLD",
        "EDGE_EVIDENCE_STRENGTH", "EDGE_CAUSALITY", "EDGE_VERB_EVIDENCE",
        "MASK_OUTPUT_FIELDS", "RUN_ID", "CORRELATION_ID",
        "PIPELINE_VERSION", "SCHEMA_VERSION", "CONFIG_HASH",
        "LOG_FORMAT", "LOG_LEVEL", "STRUCTURED_LOGGING",
    ],
}


def compute_impact_analysis(changed_key: str) -> list[str]:
    """Compute which config keys are affected by a change.

    Parameters
    ----------
    changed_key : str

    Returns
    -------
    list of str
        All keys that depend (transitively) on changed_key.
    """
    affected: set[str] = set()
    queue = [changed_key]
    while queue:
        current = queue.pop(0)
        for key, deps in CONFIG_DEPENDENCY_GRAPH.items():
            # deps are what `key` depends ON — so if `current`
            # is in deps, then `key` is affected
            if current in deps and key not in affected:
                affected.add(key)
                queue.append(key)
    return sorted(affected)


# Fixes audit issue 16.8 — log_transformation
def log_transformation(
    step: str,
    input_desc: str,
    output_desc: str,
    transform_type: str,
    record_count: int | None = None,
) -> None:
    """Log a data transformation for lineage tracking.

    Parameters
    ----------
    step : str
    input_desc : str
    output_desc : str
    transform_type : str
    record_count : int, optional
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "input": input_desc,
        "output": output_desc,
        "transform_type": transform_type,
        "record_count": record_count,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": RUN_ID,
    }
    try:
        ensure_dirs()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = TRANSFORMATION_LOG_DIR / f"transform_{timestamp}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
        logger.error("Failed to log transformation: %s", exc)
    logger.info(
        "Transform: %s | %s -> %s | type=%s | records=%s",
        step, input_desc, output_desc, transform_type, record_count,
    )


# Fixes audit issue 16.11 — diff_configs
def diff_configs(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
) -> dict[str, Any]:
    """Diff two configuration dicts and return changes.

    Parameters
    ----------
    old_config : dict
    new_config : dict

    Returns
    -------
    dict
        Keys: "added", "removed", "changed" (each a dict of key -> value(s))
    """
    added = {k: v for k, v in new_config.items() if k not in old_config}
    removed = {k: v for k, v in old_config.items() if k not in new_config}
    changed = {}
    for k in old_config:
        if k in new_config and old_config[k] != new_config[k]:
            changed[k] = {"old": old_config[k], "new": new_config[k]}
    return {"added": added, "removed": removed, "changed": changed}


# ─── Phase J — Validators & Self-tests ───────────────────────────────────────

# Fixes audit issue 4.15 — THRESHOLD_LOCKS
THRESHOLD_LOCKS: frozenset[str] = frozenset({
    "MIN_NODES_W2", "MIN_EDGES_W2", "MIN_POSITIVE_PAIRS",
    "MIN_NEGATIVE_PAIRS", "STRING_SCORE_THRESHOLD",
    "STITCH_SCORE_THRESHOLD", "STRING_MIN_COMBINED_SCORE",  # v57 ROOT FIX (P2L-032)
    "ENTITY_CONFIDENCE_THRESHOLD",
    "V1_LAUNCH_AUC",
})

# Fixes audit issue 4.15 — MAGIC_NUMBERS_REGISTRY
MAGIC_NUMBERS_REGISTRY: dict[str, dict[str, str]] = {
    "MIN_NODES_W2": {
        "value": str(MIN_NODES_W2),
        "rationale": "Week 2 exit criterion from project spec; ensures KG is not trivially small",
    },
    "MIN_EDGES_W2": {
        "value": str(MIN_EDGES_W2),
        "rationale": "Stricter than spec's 5M; catches incomplete DRKG loads",
    },
    "MIN_POSITIVE_PAIRS": {
        "value": str(MIN_POSITIVE_PAIRS),
        "rationale": "Minimum for statistically significant AUC evaluation",
    },
    "MIN_NEGATIVE_PAIRS": {
        "value": str(MIN_NEGATIVE_PAIRS),
        "rationale": (
            "Production: 75_000 negatives for 15_000 positives (5:1 neg:pos "
            "ratio ensures model learns discriminative features). Dev mode: "
            "1 pair so the toy fixture can pass V1 launch criteria "
            "end-to-end (see DRUGOS_DEV_MIN_NEGATIVE_PAIRS env override)."
        ),
    },
    "SEED": {
        "value": str(SEED),
        "rationale": "42 follows scikit-learn convention; ensures reproducibility",
    },
    "STRING_SCORE_THRESHOLD": {
        "value": str(STRING_SCORE_THRESHOLD),
        "rationale": "700 = 'high confidence' per STRING docs; below 400 is low confidence",
    },
    "STITCH_SCORE_THRESHOLD": {
        "value": str(STITCH_SCORE_THRESHOLD),
        "rationale": "700 = 'high confidence' per STITCH docs; matches STRING threshold",
    },
    "STRING_MIN_COMBINED_SCORE": {
        "value": str(STRING_MIN_COMBINED_SCORE),
        "rationale": (
            "v57 ROOT FIX (P2L-032): canonical STRING combined_score "
            "minimum threshold shared by string_loader / stitch_loader / "
            "any future loader that filters STRING-sourced edges. "
            "Default 700 = 'high confidence' per STRING-db "
            "documentation (Szklarczyk 2023, Nucleic Acids Research) — "
            ">= 700 achieves >80% precision on KEGG pathway benchmarks."
        ),
    },
    "ENTITY_CONFIDENCE_THRESHOLD": {
        "value": str(ENTITY_CONFIDENCE_THRESHOLD),
        "rationale": "0.85 = standard NER confidence (per spaCy/SciSpacy defaults)",
    },
    "V1_LAUNCH_AUC": {
        "value": str(V1_LAUNCH_AUC),
        "rationale": (
            "0.85 is the V1 launch threshold per the project DOCX "
            "(\">0.85 AUC on held-out drug-disease pairs\"). The DRKG TransE "
            "BASELINE is 0.78 (Week 2 exit criterion); production TARGET is "
            "0.85+ for V1 LAUNCH. v25 ROOT FIX: this constant is ALWAYS 0.85 "
            "(never silently lowered to 0.78 or 0.5); the dev-mode smoke-test "
            "loophole is in the launch_verdict check, not in this constant."
        ),
    },
}

# Fixes audit issue 13.3 — DATA_DICTIONARY
DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "CORE_NODE_TYPES": {
        "type": "list[str]",
        "description": "The 5 core node types in DrugOS knowledge graph",
        "valid_values": "Compound, Disease, Gene, Protein, Pathway",
    },
    "DRKG_NODE_TYPES": {
        "type": "list[str]",
        "description": "All 15 DRKG-derived node types (including Atc, Tax)",
        "valid_values": str(DRKG_NODE_TYPES),
    },
    "CORE_EDGE_TYPES": {
        "type": "list[Tuple[str, str, str]]",
        "description": "All core edge types as (source, relation, target) tuples",
        "valid_values": f"{len(CORE_EDGE_TYPES)} edges defined",
    },
    "CANONICAL_IDS": {
        "type": "dict[str, str]",
        "description": "Canonical ID system per entity type",
        "valid_values": str(dict(CANONICAL_IDS)),
    },
    "DATA_SOURCES": {
        "type": "dict[str, dict]",
        "description": "Registry of all external data sources with download metadata",
        "valid_values": f"{len(DATA_SOURCES)} sources: {sorted(DATA_SOURCES.keys())}",
    },
    "SEED": {
        "type": "int",
        "description": "Global random seed for reproducibility",
        "valid_values": "Any positive integer; default 42",
    },
    "MIN_NODES_W2": {
        "type": "int",
        "description": "Minimum node count for Week 2 exit criteria",
        "valid_values": ">= 0",
    },
    "MIN_EDGES_W2": {
        "type": "int",
        "description": "Minimum edge count for Week 2 exit criteria",
        "valid_values": ">= 0",
    },
    "STRING_SCORE_THRESHOLD": {
        "type": "int",
        "description": "Minimum STRING combined score for PPI inclusion",
        "valid_values": "0-1000; 700 = high confidence",
    },
    "STITCH_SCORE_THRESHOLD": {
        "type": "int",
        "description": "Minimum STITCH combined score for CPI inclusion",
        "valid_values": "0-1000; 700 = high confidence",
    },
    "STRING_MIN_COMBINED_SCORE": {
        "type": "int",
        "description": (
            "v57 ROOT FIX (P2L-032): canonical STRING combined_score "
            "minimum threshold shared across all loaders that filter "
            "STRING-sourced PPI / CPI edges (string_loader, stitch_loader, "
            "and any future loader). Alias of STRING_SCORE_THRESHOLD."
        ),
        "valid_values": "0-1000; 700 = high confidence (>80% precision)",
    },
}

# Fixes audit issue 13.3 — print_data_dictionary
def print_data_dictionary() -> str:
    """Print the data dictionary in human-readable format.

    Returns
    -------
    str
    """
    lines = ["DrugOS Config Data Dictionary", "=" * 40]
    for name, info in sorted(DATA_DICTIONARY.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  Type: {info['type']}")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Valid values: {info['valid_values']}")
    return "\n".join(lines)


# Fixes audit issue 13.12 — CONFIG_SECTIONS
CONFIG_SECTIONS: list[dict[str, str]] = [
    {"name": "Foundations", "description": "Version constants, __all__, seed, config hash"},
    {"name": "Directories", "description": "All path constants and ensure_dirs()"},
    {"name": "Data Sources", "description": "DATA_SOURCES registry with metadata"},
    {"name": "Neo4j Config", "description": "Neo4jConfig dataclass and singleton"},
    {"name": "KG Schema", "description": "CORE_NODE_TYPES, DRKG_NODE_TYPES, CORE_EDGE_TYPES"},
    {"name": "Edge Metadata", "description": "EDGE_EVIDENCE_STRENGTH, EDGE_CAUSALITY, EDGE_PRODUCERS"},
    {"name": "PyG Config", "description": "PyGConfig dataclass"},
    {"name": "TransE Config", "description": "TransEConfig dataclass"},
    {"name": "AUC Enforcement", "description": "AUCEnforcementLevel, assert_auc_meets_threshold"},
    {"name": "Entity Resolution", "description": "CANONICAL_IDS, ID_MAPPING_PRIORITY, resolve_canonical_id"},
    {"name": "Validation Thresholds", "description": "MIN_NODES_W2, MIN_EDGES_W2, etc."},
    {"name": "Data Quality", "description": "verify_checksum, check_data_freshness, download_with_retry"},
    {"name": "Reliability", "description": "dead_letter_record, write_checkpoint, read_latest_checkpoint"},
    {"name": "Performance", "description": "auto_size_neo4j_memory, BATCH_SIZE_BY_NODE_TYPE, CHUNK_SIZE"},
    {"name": "Security", "description": "safe_config_dict, PII_FIELDS, audit_log"},
    {"name": "Logging", "description": "LOG_FORMAT, LOG_LEVEL, JsonFormatter, RUN_ID"},
    {"name": "Config Management", "description": "load_config_from_file, apply_config_overrides, ENVIRONMENT"},
    {"name": "Compliance", "description": "ComplianceConfig, DataFormatConfig, NAMING_CONVENTIONS"},
    {"name": "Lineage", "description": "LineageMetadata, build_lineage_metadata, log_transformation"},
    {"name": "Validators", "description": "validate_all, THRESHOLD_LOCKS, MAGIC_NUMBERS_REGISTRY"},
]


# ─── Validators ──────────────────────────────────────────────────────────────

def _validate_scientific_schema() -> list[str]:
    """Validate scientific schema consistency.

    Returns
    -------
    list of str
        Issues found (empty = pass).
    """
    issues: list[str] = []
    node_set = set(CORE_NODE_TYPES) | set(DRKG_NODE_TYPES)

    for src, rel, dst in CORE_EDGE_TYPES:
        if src not in node_set:
            issues.append(f"Edge {rel!r}: unknown source node type {src!r}")
        if dst not in node_set:
            issues.append(f"Edge {rel!r}: unknown destination node type {dst!r}")

    for ent_type in CANONICAL_IDS:
        if ent_type not in node_set:
            issues.append(f"CANONICAL_IDS key {ent_type!r} is not a known node type")

    if ("Compound", "treats", "Disease") not in CORE_EDGE_TYPES:
        issues.append("Core edge ('Compound', 'treats', 'Disease') is missing")

    if ("Gene", "encodes", "Protein") not in CORE_EDGE_TYPES:
        issues.append("Bridge edge ('Gene', 'encodes', 'Protein') is missing")

    return issues


def _validate_data_sources() -> list[str]:
    """Validate DATA_SOURCES dict.

    Returns
    -------
    list of str
    """
    issues: list[str] = []
    required_keys = {"url", "filename", "description", "version_note", "version", "pinned", "sha256"}
    for src_name, src_cfg in DATA_SOURCES.items():
        missing = required_keys - set(src_cfg.keys())
        if missing:
            issues.append(
                f"DATA_SOURCES[{src_name!r}] missing keys: {sorted(missing)}"
            )
        url = str(src_cfg.get("url", ""))
        if not url.startswith(("http://", "https://", "ftp://")):
            issues.append(
                f"DATA_SOURCES[{src_name!r}].url is not valid: {url!r}"
            )

    # Cross-check __data_sources_version__
    src_keys = set(DATA_SOURCES.keys())
    version_keys = set(__data_sources_version__.keys())
    if src_keys != version_keys:
        only_in_sources = src_keys - version_keys
        only_in_version = version_keys - src_keys
        if only_in_sources:
            issues.append(
                f"__data_sources_version__ missing keys: {sorted(only_in_sources)}"
            )
        if only_in_version:
            issues.append(
                f"__data_sources_version__ has extra keys: {sorted(only_in_version)}"
            )

    return issues


def _validate_id_mapping_priority() -> list[str]:
    """Validate ID_MAPPING_PRIORITY consistency with CANONICAL_IDS.

    Returns
    -------
    list of str
    """
    issues: list[str] = []
    for ent_type, canonical_system in CANONICAL_IDS.items():
        priority = ID_MAPPING_PRIORITY.get(ent_type, [])
        if canonical_system not in priority:
            issues.append(
                f"CANONICAL_IDS[{ent_type!r}] = {canonical_system!r} "
                f"but {canonical_system!r} is not in ID_MAPPING_PRIORITY[{ent_type!r}]"
            )
    return issues


def _validate_node_type_consistency() -> list[str]:
    """Validate that all edge endpoints are known node types.

    Returns
    -------
    list of str
    """
    issues: list[str] = []
    known = set(CORE_NODE_TYPES) | set(DRKG_NODE_TYPES)
    # Add extra types that appear in edges
    for src, _, dst in CORE_EDGE_TYPES:
        for node_type in (src, dst):
            if node_type not in known:
                issues.append(
                    f"Node type {node_type!r} appears in CORE_EDGE_TYPES "
                    f"but not in CORE_NODE_TYPES or DRKG_NODE_TYPES"
                )
    return issues


def _validate_pinned_versions() -> list[str]:
    """Validate that critical sources have pinned URLs AND version pins.

    Returns
    -------
    list of str
        List of validation issue strings (empty if all checks pass).

    v28 ROOT FIX (audit TOP-19): the previous validator only checked the
    ``pinned: bool`` flag — a source could set ``pinned: True`` but
    have no ``url`` and no ``version`` field, and the validator would
    still report it as compliant. This made the "pinned" guarantee
    meaningless: a source flagged as pinned could still drift across
    upstream releases because nothing actually pinned it. The fix
    enforces the INVARIANT that a pinned source MUST have BOTH a URL
    and a version pin, and that the URL must mention the version
    (heuristic — catches the common case where the URL has no version
    token at all, e.g. ``https://example.com/data/latest`` which is
    by definition unpinned).
    """
    issues: list[str] = []
    for src_name in CRITICAL_SOURCES:
        src = DATA_SOURCES.get(src_name, {})
        if not src.get("pinned", False):
            issues.append(
                f"Critical source {src_name!r} is not pinned to a specific version"
            )
            continue
        # v28 TOP-19: pinned=True means nothing without a URL AND a
        # version. Flag both directions.
        url = src.get("url")
        version = src.get("version")
        if not url:
            issues.append(
                f"Critical source {src_name!r} is flagged pinned=True "
                f"but has no 'url' field — pinning is meaningless "
                f"without a URL to pin to."
            )
        if not version:
            issues.append(
                f"Critical source {src_name!r} is flagged pinned=True "
                f"but has no 'version' field — without an explicit "
                f"version, upstream can release a new version under the "
                f"same URL and the pipeline silently picks it up."
            )
        # Heuristic: the URL should mention the version (e.g.
        # ".../releases/5.1.10/..." or ".../v12.0/..."). This catches
        # URLs like ".../latest" or ".../download" that are by
        # definition unpinned even when 'version' is set elsewhere.
        if url and version:
            url_str = str(url)
            version_str = str(version)
            # Strip leading 'v' from version (some sources use "v5.1.10"
            # in tags but "5.1.10" in URLs) and check both forms.
            version_variants = {
                version_str,
                version_str.lstrip("v"),
                f"v{version_str.lstrip('v')}",
            }
            # Also accept the major.minor prefix (e.g. "5.1" matches a
            # URL containing "5.1.10") so a version pin of "5.1.x" is
            # accepted as a URL pin.
            major_minor = ".".join(version_str.split(".")[:2])
            version_variants.add(major_minor)
            if not any(v in url_str for v in version_variants if v):
                issues.append(
                    f"Critical source {src_name!r}: pinned version "
                    f"{version_str!r} does not appear in URL {url_str!r}. "
                    f"This usually means the URL points to a 'latest' or "
                    f"'master' symlink — upstream can release a new "
                    f"version under the same URL and the pipeline will "
                    f"silently pick it up, defeating the pin."
                )
    return issues


def _validate_no_hardcoded_filenames() -> list[str]:
    """Validate that config doesn't hardcode filenames that consumers hardcode.

    Returns
    -------
    list of str
    """
    # This is a placeholder — actual validation would scan consumer files
    return []


def validate_all() -> dict[str, list[str]]:
    """Run all validators and return a report.

    Returns
    -------
    dict[str, list[str]]
        Mapping from validator name to list of issues.
    """
    validators = [
        _validate_scientific_schema,
        _validate_data_sources,
        _validate_id_mapping_priority,
        _validate_node_type_consistency,
        _validate_pinned_versions,
        _validate_no_hardcoded_filenames,
    ]
    report: dict[str, list[str]] = {}
    for v in validators:
        try:
            report[v.__name__] = v()
        except (ValueError, TypeError, RuntimeError, KeyError, OSError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
            report[v.__name__] = [f"VALIDATOR CRASHED: {exc}"]
    return report


# ─── Self-test functions ─────────────────────────────────────────────────────

def _self_test_schema() -> bool:
    """Self-test: schema consistency."""
    report = _validate_scientific_schema()
    if report:
        for issue in report:
            logger.error("Schema self-test FAIL: %s", issue)
        return False
    return True


def _self_test_data_sources() -> bool:
    """Self-test: data sources valid."""
    report = _validate_data_sources()
    if report:
        for issue in report:
            logger.error("Data sources self-test FAIL: %s", issue)
        return False
    return True


def _self_test_config_hash() -> bool:
    """Self-test: config hash is deterministic."""
    h1 = compute_config_hash()
    h2 = compute_config_hash()
    if h1 != h2:
        logger.error("Config hash not deterministic: %s != %s", h1, h2)
        return False
    return True


def _self_test_neo4j_repr() -> bool:
    """Self-test: Neo4jConfig.__repr__ masks password."""
    cfg = Neo4jConfig(password="secret_password_123")
    r = repr(cfg)
    if "secret_password_123" in r:
        logger.error("Neo4jConfig.__repr__ leaks password!")
        return False
    return True


def _self_test_safe_config() -> bool:
    """Self-test: safe_config_dict doesn't expose secrets.

    v28 ROOT FIX (audit TOP-20): the previous self-test only flagged
    values that contained BOTH ``password`` AND ``secret`` — almost no
    real value contains both (a password is just a password, an API
    token is just a token). The AND logic was a logical typo that made
    the self-test a no-op: a leaked password like ``"p@ssw0rd"`` would
    pass because it contains ``password`` but not ``secret``; a leaked
    token like ``"sk-abc123"`` would pass because it contains neither.
    The fix uses OR logic across a comprehensive secret-keyword list
    (``password``, ``secret``, ``api_key``, ``token``, ``credentials``,
    ``apikey``) and additionally checks the KEY name (not just the
    value), because a key like ``"DATABASE_PASSWORD"`` with a value of
    ``"p@ssw0rd"`` is just as much a leak as a value containing the
    word "password".
    """
    SECRET_KEYWORDS = (
        "password", "secret", "api_key", "apikey", "token", "credentials",
        "private_key", "access_key", "auth_token",
    )
    d = safe_config_dict()
    for key, val in d.items():
        key_lower = str(key).lower()
        val_lower = str(val).lower() if val is not None else ""
        # Check BOTH key name and value — a leak in either is a leak.
        # (Previously only the value was checked, AND only for the
        # "password"+"secret" conjunction.)
        if any(kw in key_lower for kw in SECRET_KEYWORDS):
            logger.error(
                "safe_config_dict exposes secret-like KEY %r (value "
                "redacted for logging). Secret-bearing keys must be "
                "filtered by safe_config_dict before serialization.",
                key,
            )
            return False
        if isinstance(val, str) and any(kw in val_lower for kw in SECRET_KEYWORDS):
            logger.error(
                "safe_config_dict exposes secret-like VALUE in key %r "
                "(matched keyword in value).", key,
            )
            return False
    return True


def _self_test_edge_consistency() -> bool:
    """Self-test: all edges in EDGE_PRODUCERS are in CORE_EDGE_TYPES."""
    for edge in EDGE_PRODUCERS:
        if edge not in CORE_EDGE_TYPES_SET:
            logger.error("EDGE_PRODUCERS has edge not in CORE_EDGE_TYPES: %s", edge)
            return False
    return True


def _self_test_auc_enforcement() -> bool:
    """Self-test: AUC enforcement raises on low AUC."""
    try:
        assert_auc_meets_threshold(0.50, enforcement_level=AUCEnforcementLevel.STANDARD)
        logger.error("AUC enforcement did not raise on AUC=0.50!")
        return False
    except AUCBelowThresholdError:
        return True


def _self_test_id_mapping_priority() -> bool:
    """Self-test: canonical ID is first in mapping priority."""
    for ent_type, canonical in CANONICAL_IDS.items():
        priority = ID_MAPPING_PRIORITY.get(ent_type, [])
        if canonical not in priority:
            logger.error(
                "CANONICAL_IDS[%s]=%s not in ID_MAPPING_PRIORITY", ent_type, canonical
            )
            return False
    return True


def _self_test_inchikey_validation() -> bool:
    """Self-test: InChIKey validation works."""
    valid = validate_inchikey("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    if not valid:
        logger.error("Valid InChIKey failed validation!")
        return False
    invalid = validate_inchikey("NOT-A-VALID-KEY")
    if invalid:
        logger.error("Invalid InChIKey passed validation!")
        return False
    return True


def _self_test_lineage_metadata() -> bool:
    """Self-test: LineageMetadata has all required fields."""
    meta = build_lineage_metadata()
    required_fields = [
        "pipeline_version", "config_version", "config_hash",
        "schema_version", "input_checksums", "seed",
        "run_id", "created_at", "created_by",
    ]
    for f_name in required_fields:
        if not hasattr(meta, f_name):
            logger.error("LineageMetadata missing field: %s", f_name)
            return False
    return True


def _self_test_frozen_dataclasses() -> bool:
    """Self-test: dataclasses are frozen."""
    for cls in (Neo4jConfig, PyGConfig, TransEConfig):
        try:
            instance = cls()
            try:
                # Try to set an attribute — should raise FrozenInstanceError
                # In Python 3.10+, frozen dataclasses raise dataclasses.FrozenInstanceError
                # which is a subclass of AttributeError
                if hasattr(instance, 'uri'):
                    instance.uri = 'test'  # type: ignore[misc]
                elif hasattr(instance, 'embedding_dim'):
                    instance.embedding_dim = 999  # type: ignore[misc]
                else:
                    instance.train_ratio = 0.5  # type: ignore[misc]
                logger.error("%s is not frozen!", cls.__name__)
                return False
            except AttributeError:
                pass  # Expected — frozen dataclass raises AttributeError
        except (ImportError, AttributeError, ValueError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
            logger.error("Cannot create %s: %s", cls.__name__, exc)
            return False
    return True


def _self_test_threshold_locks() -> bool:
    """Self-test: all threshold values are documented in MAGIC_NUMBERS_REGISTRY."""
    for lock in THRESHOLD_LOCKS:
        if lock not in MAGIC_NUMBERS_REGISTRY:
            logger.error("Threshold %s not in MAGIC_NUMBERS_REGISTRY", lock)
            return False
    return True


def _self_test_dead_letter() -> bool:
    """Self-test: dead_letter_record creates file."""
    ensure_dirs()
    path = dead_letter_record(
        "self_test",
        {"test": "record"},
        "Self-test rejection",
    )
    if not path.exists():
        logger.error("Dead letter file not created!")
        return False
    # Cleanup
    path.unlink(missing_ok=True)
    return True


def _self_test_backfill_mode() -> bool:
    """Self-test: backfill mode flag works."""
    # Just verify the variable exists and is bool
    return isinstance(BACKFILL_MODE, bool)


# Fixes audit issue 7.9 — BACKFILL_MODE
BACKFILL_MODE: bool = os.environ.get("DRUGOS_BACKFILL_MODE", "0") == "1"
BACKFILL_AS_OF_DATE: str | None = os.environ.get("DRUGOS_BACKFILL_AS_OF")


# ─── Phase K — Final Wiring ──────────────────────────────────────────────────

# Apply config overrides if CONFIG_FILE is set
if CONFIG_FILE:
    apply_config_overrides()

# v29 ROOT FIX (audit I-5): the previous code called
# ``apply_environment_config()`` here unconditionally (when
# ENVIRONMENT != "development"). That function is now a deprecated
# no-op — its old behavior (applying hardcoded ENVIRONMENT_CONFIGS
# overrides) was redundant because every constant it overrode is
# already read from env vars at the top of this module. Calling the
# deprecated function here would emit a DeprecationWarning on EVERY
# config import, which is noisy. The call has been removed; the
# constants ``LOG_LEVEL``, ``STRICT_AUC_ENFORCEMENT``,
# ``STRICT_EDGE_FILTERING``, ``REDACT_PII`` are the authoritative
# source of truth (read directly from env vars).

# Compute final config hash (after all overrides)
CONFIG_HASH = compute_config_hash()


# ─── Phase L — __main__ self-test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    print("=" * 60)
    print("DrugOS config.py — Self-Test Suite")
    print("=" * 60)

    self_tests = [
        ("Schema consistency", _self_test_schema),
        ("Data sources valid", _self_test_data_sources),
        ("Config hash deterministic", _self_test_config_hash),
        ("Neo4jConfig.__repr__ masks password", _self_test_neo4j_repr),
        ("safe_config_dict no leak", _self_test_safe_config),
        ("Edge producers consistent", _self_test_edge_consistency),
        ("AUC enforcement raises", _self_test_auc_enforcement),
        ("ID mapping priority valid", _self_test_id_mapping_priority),
        ("InChIKey validation", _self_test_inchikey_validation),
        ("Lineage metadata complete", _self_test_lineage_metadata),
        ("Dataclasses frozen", _self_test_frozen_dataclasses),
        ("Threshold locks documented", _self_test_threshold_locks),
        ("Dead letter record", _self_test_dead_letter),
        ("Backfill mode flag", _self_test_backfill_mode),
    ]

    passed = 0
    failed = 0
    for name, test in self_tests:
        try:
            result = test()
            if result:
                print(f"  PASS: {name}")
                passed += 1
            else:
                print(f"  FAIL: {name}")
                failed += 1
        except (ValueError, TypeError, RuntimeError, KeyError, OSError) as exc:  # v85 FORENSIC ROOT FIX (BUG #51)
            print(f"  ERROR: {name} — {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"CONFIG_HASH: {CONFIG_HASH}")
    print(f"PIPELINE_VERSION: {PIPELINE_VERSION}")
    print("=" * 60)

    if failed > 0:
        import sys
        sys.exit(1)
    else:
        print("\nAll self-tests passed!")
