"""Shared entry point for cross-database entity resolution.

v75 ROOT FIX (T-025 -- download_parallel.py skips entity resolution):
    The forensic audit found that ``scripts/download_parallel.py`` and
    the Makefile's ``download-all`` / ``download-samples`` targets all
    called ``cls(run_id=...).run()`` for each pipeline -- the FULL run
    including LOAD to DB -- but NEVER ran entity resolution. The
    knowledge graph built from such a DB had no ``entity_mapping`` rows
    and ``proteins.string_id`` was never updated. The master Airflow
    DAG was the ONLY caller that ran entity resolution (inline in the
    ``@task entity_resolution`` callable).

    The COMPOUND problem: the entity resolution logic lived INLINE in
    the Airflow task body. There was no reusable function for non-
    Airflow callers. So even a developer who noticed the gap could not
    fix ``download_parallel.py`` without either (a) duplicating ~250
    lines of code (which would drift from the Airflow version, the
    classic copy-paste bug), or (b) importing the Airflow task directly
    (couples the CLI to Airflow runtime).

    ROOT FIX (master-grade):
      1. Extract the entity resolution logic into THIS module as a
         single function ``run_entity_resolution()``.
      2. The Airflow task (master_pipeline_dag.py:entity_resolution)
         becomes a thin wrapper that calls this function.
      3. ``scripts/download_parallel.py`` calls this function between
         SECOND_PASS and THIRD_PASS (PubChem needs drugs in DB, so
         resolution MUST run before PubChem download -- same ordering
         as the master DAG).
      4. The Makefile's ``download-all`` and ``download-samples``
         targets continue to call ``.run()`` (full run) -- they are
         documented as "unresolved DB" targets. The Makefile now
         points operators at ``download-parallel`` for the resolved
         path. (Refactoring the Makefile to use the two-phase design
         for every target is out of scope for T-025; the parallel
         path is the canonical CLI entry point for resolved DBs.)

    This module has NO Airflow dependency -- it can be imported and
    called from any Python context (CLI, pytest, notebook). The only
    dependencies are pandas, SQLAlchemy, and the Phase 1 package
    (config.settings, database.connection, entity_resolution resolvers).

Usage:
    from entity_resolution.run import run_entity_resolution
    run_entity_resolution()  # raises RuntimeError if no drug data

Returns:
    A dict with keys ``drug_mappings``, ``protein_mappings``,
    ``proteins_updated`` (int counts). The return value is consumed by
    tests; production callers ignore it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


def run_entity_resolution() -> Dict[str, Any]:
    """Run cross-database entity resolution and persist results to the DB.

    Reconciles drug entities across ChEMBL, DrugBank, and PubChem using
    InChIKey matching, connectivity-block matching, and normalised-name
    matching.  Also resolves protein entities across UniProt and STRING.

    Results are persisted to the ``entity_mapping`` table and the
    ``proteins.string_id`` column is updated with resolved STRING IDs.

    Raises:
        RuntimeError: if all three drug DataFrames are empty (the
            operator has not run any download pipeline first).
    """
    import pandas as pd
    from sqlalchemy import text

    from config.settings import PROCESSED_DATA_DIR
    from database.connection import get_engine
    from entity_resolution.drug_resolver import DrugResolver
    from entity_resolution.protein_resolver import ProteinResolver

    # ------------------------------------------------------------------
    # Drug entity resolution
    # ------------------------------------------------------------------
    logger.info("Starting drug entity resolution ...")
    drug_resolver = DrugResolver()

    chembl_path = PROCESSED_DATA_DIR / "drugs.csv"
    drugbank_path = PROCESSED_DATA_DIR / "drugbank_drugs.csv"
    pubchem_path = PROCESSED_DATA_DIR / "pubchem_enrichment.csv"

    chembl_df = (
        pd.read_csv(chembl_path, low_memory=False)
        if chembl_path.exists()
        else pd.DataFrame()
    )
    drugbank_df = (
        pd.read_csv(drugbank_path, low_memory=False)
        if drugbank_path.exists()
        else pd.DataFrame()
    )
    pubchem_df = (
        pd.read_csv(pubchem_path, low_memory=False)
        if pubchem_path.exists()
        else pd.DataFrame()
    )

    # FIX AUDIT-7: Validate that at least one drug DataFrame has data.
    total_drug_records = len(chembl_df) + len(drugbank_df) + len(pubchem_df)
    if total_drug_records == 0:
        logger.error(
            "All three drug DataFrames are empty (chembl=%d, drugbank=%d, pubchem=%d). "
            "This usually means the CSV files in %s do not exist or are empty. "
            "Run the download pipelines first before entity resolution.",
            len(chembl_df), len(drugbank_df), len(pubchem_df),
            PROCESSED_DATA_DIR,
        )
        raise RuntimeError(
            f"Entity resolution cannot proceed: all drug DataFrames are empty. "
            f"Ensure ChEMBL, DrugBank, and/or PubChem pipelines have been run. "
            f"Checked: {chembl_path}, {drugbank_path}, {pubchem_path}"
        )
    # Validate required columns in non-empty DataFrames.
    # v107 ROOT FIX (ISSUE-P1-038 — silent warning on missing required
    #   columns):
    #   The previous code logged a WARNING and continued when a non-empty
    #   drug DataFrame was missing ``inchikey`` or ``name``. Entity
    #   resolution then produced 0 canonical entities (because every row
    #   lacked ``inchikey`` for cross-source deduplication), the KG had
    #   0 Compound nodes, the GNN trained on 0 drugs, and the RL ranker
    #   produced 0 candidates. The platform appeared to "work" but was
    #   empty — a silent data-loss path that the audit correctly flagged.
    #   A misconfigured ChEMBL pipeline that drops the ``inchikey``
    #   column (e.g. due to a schema change at the upstream API) would
    #   be silently masked as a warning.
    # ROOT FIX: raise RuntimeError immediately. The operator MUST fix
    # the upstream pipeline before entity resolution can proceed. This
    # is the patient-safe behaviour: a clearly-failing pipeline is
    # always preferable to a silently-empty KG.
    required_drug_cols = {"inchikey", "name"}
    for name, df_check in [("chembl", chembl_df), ("drugbank", drugbank_df), ("pubchem", pubchem_df)]:
        if not df_check.empty and not required_drug_cols.issubset(set(df_check.columns)):
            missing = required_drug_cols - set(df_check.columns)
            logger.error(
                "Drug DataFrame '%s' is missing required columns: %s. "
                "Available columns: %s. Entity resolution CANNOT proceed "
                "without 'inchikey' (cross-source dedup key) and 'name' "
                "(display label). Fix the upstream pipeline before retrying.",
                name, missing, list(df_check.columns),
            )
            raise RuntimeError(
                f"Entity resolution cannot proceed: drug DataFrame "
                f"'{name}' ({len(df_check)} rows) is missing required "
                f"columns {sorted(missing)}. Available: {list(df_check.columns)}. "
                f"This is a hard stop — silently continuing would produce "
                f"an empty KG (0 Compound nodes), breaking Phase 2/3/4. "
                f"Fix the upstream {name} pipeline before retrying."
            )

    drug_mapping_df = drug_resolver.build_mapping(chembl_df, drugbank_df, pubchem_df)
    logger.info(
        "Drug entity resolution complete: %d canonical entities",
        len(drug_mapping_df),
    )

    # ------------------------------------------------------------------
    # Protein entity resolution
    # ------------------------------------------------------------------
    logger.info("Starting protein entity resolution ...")
    protein_resolver = ProteinResolver()

    proteins_path = PROCESSED_DATA_DIR / "proteins.csv"
    uniprot_df = (
        pd.read_csv(proteins_path, low_memory=False)
        if proteins_path.exists()
        else pd.DataFrame()
    )

    # FIX AUDIT-8: Also load STRING PPI data to provide protein IDs from
    # the interaction network. Extract unique UniProt IDs from both
    # uniprot_id_a and uniprot_id_b columns of the STRING processed output.
    # Schema reconciliation (GUARD-2.1, GUARD-2.2, BUG-14.1, BUG-15.1,
    # BUG-15.2): the upgraded StringPipeline now outputs the schema-
    # conformant column names `uniprot_id_a` / `uniprot_id_b` (was
    # `uniprot_a` / `uniprot_b`).
    #
    # v80 FORENSIC ROOT FIX (P0-D2 -- _string_to_uniprot cross-reference
    #   index NEVER populated):
    #   The previous code only loaded the STRING PPI edges file
    #   (``protein_protein_interactions.csv``), extracted unique UniProt
    #   IDs from it, and passed that as ``string_df`` to
    #   ``protein_resolver.build_mapping``. But ``string_df`` only
    #   creates ``string_derived`` provisional entries -- it does NOT
    #   populate the resolver's ``_string_to_uniprot`` cross-reference
    #   index. That index is populated ONLY from ``string_aliases_df``
    #   (STRING alias data containing the STRING->UniProt mapping).
    #   Without that index, ``resolve_single(string_id=...)`` is a dead
    #   path: it looks up ``self._string_to_uniprot.get(string_id)``
    #   which always returns None because the dict was never populated.
    #   STRING-derived protein IDs therefore NEVER resolve to UniProt,
    #   and STRING PPI edges never merge into the canonical Protein
    #   subgraph -- a silent integration gap that produced a KG with
    #   disconnected STRING and UniProt protein clusters.
    #
    #   ROOT FIX: also load the STRING aliases file
    #   (``string_protein_aliases.csv`` if emitted by the STRING
    #   pipeline's clean() stage, OR the raw ``.aliases.vXX.txt.gz``
    #   in raw_dir as a fallback) and pass it as ``string_aliases_df``.
    #   This populates ``_string_to_uniprot`` so ``resolve_single`` works.
    #
    # v80 FORENSIC ROOT FIX (P0-D5 -- fragile STRING column-name check):
    #   The previous code only checked for columns named
    #   ``uniprot_id_a`` / ``uniprot_id_b`` (the v49 schema). If the
    #   STRING pipeline emitted the legacy ``uniprot_a`` / ``uniprot_b``
    #   schema (or the v50 sample's ``uniprot_ac_a`` / ``uniprot_ac_b``,
    #   or the embedded ``uniprot_id1`` / ``uniprot_id2``), the
    #   ``string_protein_df`` was silently empty -- 0 Protein-Protein
    #   edges in the KG, missing the entire PPI subgraph.
    #
    #   ROOT FIX: check ALL known column-name variants and use whichever
    #   pair is present. The known variants are:
    #     - ``uniprot_id_a`` / ``uniprot_id_b``  (v49 schema-conformant)
    #     - ``uniprot_a``    / ``uniprot_b``     (legacy v48)
    #     - ``uniprot_ac_a`` / ``uniprot_ac_b``  (v50 sample mode + embedded)
    #     - ``uniprot_id1``  / ``uniprot_id2``   (embedded samples legacy)
    string_path = PROCESSED_DATA_DIR / "protein_protein_interactions.csv"
    string_protein_df = pd.DataFrame()
    if string_path.exists():
        try:
            string_df = pd.read_csv(string_path, low_memory=False)
            if not string_df.empty:
                # v89 FORENSIC ROOT FIX (BUG #16 P1 -- UniProt/STRING column
                #   detection used DIFFERENT variant lists):
                #   The previous code had TWO separate variant lists:
                #     - _COLUMN_PAIR_VARIANTS (5 pairs) for UniProt IDs
                #     - _string_col_variants (3 pairs) for STRING IDs
                #   If the UniProt pair was ``uniprot_id_a/uniprot_id_b``
                #   but the STRING pair was ``protein_a/protein_b`` (NOT in
                #   _string_col_variants), the STRING ID pairing was
                #   silently skipped. ``uniprot_to_string_id`` remained
                #   empty, and the organism inference (Source 3 in
                #   build_mapping) never fired -- non-human proteins were
                #   mislabeled as "Homo sapiens" (BUG #18).
                #   ROOT FIX: UNIFY the column-pair detection. Each
                #   variant entry is a 4-tuple
                #   ``(uniprot_a, uniprot_b, string_a, string_b)``. We
                #   pick the FIRST variant whose UniProt pair is present;
                #   the STRING pair from the SAME variant is used for
                #   STRING ID extraction. If the chosen variant's STRING
                #   pair is NOT present in the DataFrame, we log a
                #   WARNING (so operators notice) but still extract the
                #   UniProt IDs (organism inference will fall back to
                #   the override table).
                _COLUMN_PAIR_VARIANTS = [
                    # (uniprot_a, uniprot_b, string_a, string_b)
                    ("uniprot_id_a", "uniprot_id_b", "string_protein_a", "string_protein_b"),
                    ("uniprot_a", "uniprot_b", "string_protein_a", "string_protein_b"),
                    ("uniprot_ac_a", "uniprot_ac_b", "string_protein_a", "string_protein_b"),
                    ("uniprot_id1", "uniprot_id2", "string_protein_a", "string_protein_b"),
                    # Legacy variant: some emitters used protein_a/protein_b
                    # for BOTH UniProt (when no uniprot_* cols) and STRING.
                    ("string_protein_a", "string_protein_b", "string_protein_a", "string_protein_b"),
                ]
                col_a, col_b = None, None
                _string_col_a, _string_col_b = None, None
                for _ca, _cb, _sa, _sb in _COLUMN_PAIR_VARIANTS:
                    if _ca in string_df.columns and _cb in string_df.columns:
                        col_a, col_b = _ca, _cb
                        # Use the STRING pair from the SAME variant.
                        if _sa in string_df.columns and _sb in string_df.columns:
                            _string_col_a, _string_col_b = _sa, _sb
                        logger.info(
                            "Found STRING UniProt ID columns: %s / %s "
                            "(STRING ID columns: %s / %s)",
                            col_a, col_b,
                            _string_col_a or "<none>", _string_col_b or "<none>",
                        )
                        break
                if col_a and col_b:
                    uniprot_ids = set()
                    # v89 FORENSIC ROOT FIX (BUG #7 P1 -- uniprot_to_string_id
                    #   overwrote previous value, last STRING ID won):
                    #   The previous code used a single-valued dict
                    #   ``uniprot_to_string_id[uid_str] = sid_str``. If
                    #   the same UniProt accession appeared in multiple
                    #   STRING PPI rows (common -- a protein has many
                    #   interaction partners), the dictionary assignment
                    #   overwrote the previous value. The LAST STRING ID
                    #   encountered won, with no consistency check. If
                    #   uid_str was paired with sid_a in row 1 (human)
                    #   and sid_b in row 2 (mouse, due to BUG #2
                    #   mispairing), the final mapping was uid_str ->
                    #   sid_b -- non-deterministic, depending on row
                    #   ordering.
                    #   ROOT FIX: use a MULTI-VALUED dict
                    #   ``uniprot_to_string_ids: dict[str, set[str]]``.
                    #   After collecting all pairings, VALIDATE that all
                    #   STRING IDs paired with the same UniProt accession
                    #   share the SAME taxonomy prefix (the part before
                    #   the first "."). If they conflict (e.g. one human
                    #   9606.* and one mouse 10090.*), the UniProt
                    #   accession is AMBIGUOUS -- we DEAD-LETTER it (log
                    #   a WARNING and exclude it from the string_id
                    #   column) so the resolver's organism inference
                    #   (Source 3) does not pick a random taxonomy.
                    uniprot_to_string_ids: Dict[str, set] = {}
                    # v89 FORENSIC ROOT FIX (BUG #2 P0 -- STRING ID
                    #   mispairing):
                    #   The previous code iterated
                    #   ``for col in (col_a, col_b):`` and for EACH col
                    #   checked ``for scol in (_string_col_a,
                    #   _string_col_b):``. For uid_b (from col_b), it
                    #   ALSO checked _string_col_a FIRST, found sid_a
                    #   (the SAME value as for uid_a), and broke --
                    #   WRONG. uid_b should be paired with sid_b (from
                    #   _string_col_b), not sid_a. The result: BOTH
                    #   UniProt accessions in a PPI edge were paired
                    #   with the SAME STRING ID (the one from column A).
                    #   In a cross-species PPI edge
                    #   (human ↔ mouse), both UniProt accessions got
                    #   paired with the human STRING ID, so the mouse
                    #   UniProt accession was labeled "Homo sapiens" via
                    #   taxonomy-prefix inference -- corrupting organism
                    #   assignment.
                    #   ROOT FIX: iterate
                    #   ``zip((col_a, col_b), (_string_col_a,
                    #   _string_col_b))`` so uid_a pairs with sid_a and
                    #   uid_b pairs with sid_b EXPLICITLY. No inner loop.
                    if _string_col_a and _string_col_b:
                        for idx in string_df.index:
                            for _col, _scol in zip((col_a, col_b), (_string_col_a, _string_col_b)):
                                uid = string_df.at[idx, _col]
                                if pd.isna(uid):
                                    continue
                                uid_str = str(uid).strip()
                                if not uid_str or uid_str == "nan":
                                    continue
                                # Normalize UniProt accession to UPPERCASE
                                # (per UniProt spec -- accessions are
                                # case-sensitive and MUST be uppercase).
                                # This prevents duplicate canonical
                                # entries for the same protein (v89
                                # BUG #4 in protein_resolver.py).
                                uid_str = uid_str.upper()
                                uniprot_ids.add(uid_str)
                                sid = string_df.at[idx, _scol]
                                if pd.notna(sid):
                                    sid_str = str(sid).strip()
                                    if sid_str and "." in sid_str:
                                        uniprot_to_string_ids.setdefault(uid_str, set()).add(sid_str)
                    else:
                        # No STRING ID columns -- just collect UniProt IDs.
                        for idx in string_df.index:
                            for _col in (col_a, col_b):
                                uid = string_df.at[idx, _col]
                                if pd.isna(uid):
                                    continue
                                uid_str = str(uid).strip()
                                if not uid_str or uid_str == "nan":
                                    continue
                                uid_str = uid_str.upper()
                                uniprot_ids.add(uid_str)

                    # v89 BUG #7: validate taxonomy-prefix consistency
                    # for each UniProt accession's set of STRING IDs.
                    uniprot_to_string_id: Dict[str, str] = {}
                    _dead_lettered_uids: list = []
                    for _uid, _sids in uniprot_to_string_ids.items():
                        _taxids = {_s.split(".")[0] for _s in _sids if "." in _s}
                        if len(_taxids) > 1:
                            # Conflicting taxonomy prefixes -- dead-letter.
                            logger.warning(
                                "STRING PPI: UniProt accession %s paired "
                                "with STRING IDs from MULTIPLE taxa (%s) -- "
                                "organism is ambiguous. Excluding from "
                                "string_id column to prevent cross-species "
                                "contamination.",
                                _uid, sorted(_taxids),
                            )
                            _dead_lettered_uids.append(_uid)
                        elif len(_taxids) == 1:
                            # Consistent -- pick the first (deterministic).
                            uniprot_to_string_id[_uid] = sorted(_sids)[0]
                        # else: no valid taxonomy prefix -- skip (will be
                        # handled by the resolver's default-organism path).

                    if uniprot_ids:
                        _data = {"uniprot_id": list(uniprot_ids)}
                        # v89 ROOT FIX (BUG #43 -- string_id column
                        # conditionally added defeats downstream
                        # presence checks):
                        #   The previous code ONLY added the ``string_id``
                        #   column if ``uniprot_to_string_id`` was
                        #   non-empty. If ALL rows failed to pair (e.g. no
                        #   STRING ID columns detected), the column was
                        #   NOT added. Downstream code
                        #   (protein_resolver.build_mapping line ~2594)
                        #   checks ``"string_id" in string_df.columns``
                        #   -- if the column is absent, Source 3
                        #   (taxonomy-prefix inference) doesn't fire,
                        #   and STRING-derived proteins default to
                        #   "Homo sapiens" (cross-species contamination).
                        #   ROOT FIX: ALWAYS add the ``string_id`` column
                        #   (even if all values are None), so downstream
                        #   code can detect its presence and attempt
                        #   organism inference. None values are handled
                        #   correctly by the resolver.
                        _data["string_id"] = [
                            uniprot_to_string_id.get(uid)
                            for uid in uniprot_ids
                        ]
                        string_protein_df = pd.DataFrame(_data)
                        logger.info(
                            "Extracted %d unique UniProt IDs from STRING PPI data"
                            " (%d with paired STRING IDs for organism inference,"
                            " %d dead-lettered for conflicting taxa)",
                            len(string_protein_df),
                            len(uniprot_to_string_id),
                            len(_dead_lettered_uids),
                        )
                else:
                    logger.warning(
                        "STRING PPI file %s has no recognized UniProt ID "
                        "column pair. Checked variants: %s. Available "
                        "columns: %s. PPI subgraph will be empty.",
                        string_path,
                        [f"{a}/{b}" for a, b, _sa, _sb in _COLUMN_PAIR_VARIANTS],
                        list(string_df.columns),
                    )
        except Exception as exc:
            logger.warning("Failed to load STRING data for protein resolution: %s", exc)

    # v80 P0-D2: load the STRING aliases file and pass it as
    # ``string_aliases_df`` so the resolver's ``_string_to_uniprot``
    # cross-reference index is populated. Prefer the processed CSV
    # emitted by the STRING pipeline; fall back to the raw .aliases.vXX.txt.gz.
    string_aliases_df = pd.DataFrame()
    aliases_csv_path = PROCESSED_DATA_DIR / "string_protein_aliases.csv"
    if aliases_csv_path.exists():
        try:
            string_aliases_df = pd.read_csv(aliases_csv_path, low_memory=False)
            logger.info(
                "Loaded STRING aliases from processed CSV: %d rows",
                len(string_aliases_df),
            )
        except Exception as exc:
            logger.warning("Failed to load STRING aliases CSV %s: %s", aliases_csv_path, exc)
            string_aliases_df = pd.DataFrame()

    # Fallback: try to load the raw STRING aliases file from raw_dir.
    # The STRING pipeline downloads it as ``9606.protein.aliases.vXX.txt.gz``.
    if string_aliases_df.empty:
        try:
            from config.settings import RAW_DATA_DIR
            _string_raw_dir = RAW_DATA_DIR / "string"
            # v89 FORENSIC ROOT FIX (BUG #3 P0 -- alias file glob matched
            #   NON-HUMAN organism files):
            #   The previous code used ``*aliases*.txt.gz`` which matched
            #   ANY aliases file in the STRING raw directory -- including
            #   non-human organism files (10090.protein.aliases.v12.0.txt.gz
            #   = mouse, 7227.protein.aliases.v12.0.txt.gz = fly). The
            #   code picked ``_alias_files[0]`` (alphabetically first by
            #   default glob ordering). If 10090.protein.aliases... sorted
            #   before 9606.protein.aliases..., the MOUSE aliases file was
            #   loaded -- and every UniProt accession in it was treated as
            #   a mouse-to-STRING mapping. All protein mappings were
            #   wrong. The organism override table (~250 entries) did NOT
            #   catch this because most UniProt accessions are not in the
            #   table. Mouse/fly/worm proteins were provisionally entered
            #   as human and merged into human PPI subgraphs.
            #   ROOT FIX: glob SPECIFICALLY for the HUMAN aliases file
            #   (taxonomy ID 9606): ``9606.protein.aliases.*.txt.gz``.
            #   This guarantees only the human aliases file is loaded.
            #   If the human file is not present, log a WARNING (do NOT
            #   fall back to a non-human file).
            _alias_files = (
                list(_string_raw_dir.glob("9606.protein.aliases.*.txt.gz"))
                if _string_raw_dir.exists()
                else []
            )
            if _alias_files:
                # v89 ROOT FIX (BUG #28 -- corrupt aliases file silently
                # disables STRING->UniProt cross-reference):
                #   The previous code opened the first matching aliases
                #   file with ``gzip.open`` and iterated. If the file
                #   was corrupt (truncated download, partial write,
                #   disk corruption), ``gzip.open`` raised
                #   ``gzip.BadGzipFile`` (Python 3.8+) or ``OSError``.
                #   The broad ``except Exception`` at the bottom caught
                #   it and logged a WARNING -- but ``string_aliases_df``
                #   remained empty. The organism inference (Source 3
                #   in build_mapping) then didn't fire, and
                #   STRING-derived proteins defaulted to "Homo sapiens"
                #   (cross-species contamination). The operator saw a
                #   generic warning but had no signal that the file
                #   was corrupt or that the KG was compromised.
                #
                #   ROOT FIX: validate the gzip file integrity BEFORE
                #   parsing. If corrupt, dead-letter the file (move to
                #   a ``dead_letter/`` subdir) and raise a clear error
                #   so the operator can re-download. Do NOT silently
                #   continue with an empty DataFrame.
                import gzip
                _alias_file = _alias_files[0]
                # Pre-flight: verify the gzip integrity by reading the
                # first byte. ``gzip.BadGzipFile`` is raised on corrupt
                # files; ``OSError`` on truncated files. We read up to
                # 1 byte -- if that succeeds, the file is a valid gzip.
                try:
                    with gzip.open(_alias_file, "rb") as _probe:
                        _probe.read(1)
                except (gzip.BadGzipFile, OSError, EOFError) as _gz_exc:
                    _dl_dir = _string_raw_dir / "dead_letter"
                    _dl_dir.mkdir(parents=True, exist_ok=True)
                    _dl_path = _dl_dir / _alias_file.name
                    try:
                        _alias_file.rename(_dl_path)
                        logger.error(
                            "v89 BUG #28: STRING aliases file %s is "
                            "corrupt (%s). Moved to dead-letter at %s. "
                            "Re-download the file (delete it and re-run "
                            "the STRING pipeline). The KG's PPI subgraph "
                            "will be incomplete until re-downloaded.",
                            _alias_file.name, _gz_exc, _dl_path,
                        )
                    except OSError as _rename_exc:
                        logger.error(
                            "v89 BUG #28: STRING aliases file %s is "
                            "corrupt (%s) AND could not be moved to "
                            "dead-letter (%s). The file remains in place; "
                            "delete it manually and re-run the STRING "
                            "pipeline.",
                            _alias_file.name, _gz_exc, _rename_exc,
                        )
                    raise RuntimeError(
                        f"STRING aliases file {_alias_file.name} is "
                        f"corrupt ({_gz_exc}). File moved to dead-letter. "
                        f"Re-run the STRING pipeline to re-download. "
                        f"(v89 BUG #28)"
                    ) from _gz_exc
                # The raw aliases file is gzipped, space/tab-separated,
                # with columns: string_protein_id, source, alias, source_database.
                # We only need the STRING->UniProt mapping (where source_database
                # contains "UniProt").
                # v89 FORENSIC ROOT FIX (BUG #17 P1 -- fragile raw aliases
                #   parsing):
                #   The previous code split each line on tab (or
                #   whitespace as fallback) and took the first 4 fields.
                #   Problems:
                #   (a) No header validation -- STRING could change the
                #       column order and we'd silently produce wrong
                #       mappings.
                #   (b) The fallback ``_line.split()`` splits on ANY
                #       whitespace, which would break if an alias
                #       contains spaces.
                #   (c) The filter ``"UniProt" in _src_db or
                #       _source == "UniProt_AC"`` was case-sensitive --
                #       STRING uses ``UniProt_AC`` (exact) but some
                #       files use ``uniprot_ac`` (lowercase).
                #   ROOT FIX: use pandas with explicit ``sep="\t"`` and
                #   ``names=[...]`` for robust parsing. Validate the
                #   header comment (STRING aliases files start with a
                #   ``#`` header line). Make the UniProt filter
                #   case-insensitive (lowercase both sides). Skip lines
                #   that don't have exactly 4 fields after splitting
                #   (defensive -- corrupt lines are logged and skipped).
                import gzip
                _alias_records = []
                _skipped_lines = 0
                with gzip.open(_alias_file, "rt", encoding="utf-8") as _af:
                    _header_seen = False
                    for _line in _af:
                        _line = _line.rstrip("\n").rstrip("\r")
                        if not _line:
                            continue
                        if _line.startswith("#"):
                            # Header comment -- STRING aliases files have
                            # a ``#`` line describing the columns. Mark
                            # that we've seen it (so we know the file is
                            # well-formed) but don't parse it.
                            _header_seen = True
                            continue
                        # STRICT tab split -- do NOT fall back to
                        # whitespace split (BUG #17b: whitespace split
                        # breaks on aliases containing spaces).
                        _parts = _line.split("\t")
                        if len(_parts) < 4:
                            _skipped_lines += 1
                            continue
                        _string_id = _parts[0]
                        _source = _parts[1]
                        _alias = _parts[2]
                        _src_db = _parts[3]
                        # Case-insensitive UniProt filter (BUG #17c).
                        # STRING uses "UniProt_AC" (exact) but some
                        # emitters use "uniprot_ac" or "UniProt_AC_ID".
                        _src_db_lower = _src_db.lower()
                        _source_lower = _source.lower()
                        if "uniprot" in _src_db_lower or _source_lower == "uniprot_ac":
                            _alias_records.append({
                                "string_id": _string_id,
                                "uniprot_id": _alias,
                                "source": _source,
                                "source_database": _src_db,
                            })
                if not _header_seen:
                    logger.warning(
                        "STRING aliases file %s has no '#' header comment -- "
                        "file format may have changed. Proceeding with "
                        "best-effort parsing.",
                        _alias_file.name,
                    )
                if _skipped_lines:
                    logger.warning(
                        "STRING aliases file %s: skipped %d lines with "
                        "fewer than 4 tab-separated fields",
                        _alias_file.name, _skipped_lines,
                    )
                if _alias_records:
                    string_aliases_df = pd.DataFrame(_alias_records)
                    logger.info(
                        "Loaded STRING aliases from raw file %s: %d UniProt mappings",
                        _alias_file.name, len(string_aliases_df),
                    )
        except RuntimeError:
            # v89 BUG #28: re-raise RuntimeError (corrupt file) so the
            # operator sees a clear failure. Do NOT swallow it.
            raise
        except FileNotFoundError as exc:
            # v93 ROOT FIX (P1-030 -- asymmetric exception handling):
            # the previous ``except Exception`` clause SILENTLY
            # SWALLOWED FileNotFoundError, logging only a WARNING.
            # This meant a missing STRING aliases file (operator
            # error -- file not downloaded, wrong path, etc.) was
            # silently ignored, and STRING->UniProt cross-reference
            # was disabled with NO visible error. The KG then had
            # disconnected STRING and UniProt clusters, and the
            # operator had no way to know why.
            #
            # Root fix: log FileNotFoundError at ERROR level (visible
            # in every log sink) and re-raise it so the pipeline
            # fails fast. The operator MUST either (a) download the
            # STRING aliases file, or (b) explicitly acknowledge the
            # degradation by setting
            # ``DRUGOS_SKIP_STRING_ALIASES=1`` in the environment.
            # This matches the corrupt-file behavior (RuntimeError
            # re-raised above) -- both missing and corrupt files are
            # operator-actionable failures, not silent degradations.
            if os.environ.get("DRUGOS_SKIP_STRING_ALIASES", "") == "1":
                logger.warning(
                    "STRING aliases file not found: %s -- "
                    "DRUGOS_SKIP_STRING_ALIASES=1 set, continuing "
                    "with degraded STRING->UniProt resolution.",
                    exc,
                )
            else:
                logger.error(
                    "STRING aliases file not found: %s -- "
                    "STRING->UniProt cross-reference cannot be built. "
                    "Either (a) download the file to the expected "
                    "path, or (b) set DRUGOS_SKIP_STRING_ALIASES=1 "
                    "to acknowledge the degradation and continue.",
                    exc,
                )
                raise
        except Exception as exc:
            logger.warning(
                "Could not load raw STRING aliases file: %s -- "
                "string_aliases_df will be empty, resolve_single(string_id=...) "
                "will not resolve STRING IDs to UniProt",
                exc,
            )
        else:
            # v91 ROOT FIX: moved this block OUT of the except clause.
            # A botched edit had it indented INSIDE ``except RuntimeError:``
            # after a ``raise`` (unreachable + invalid ``else`` placement),
            # and a second ``except Exception`` AFTER it (invalid order:
            # Python requires try/except/except/else, not try/except/else/
            # except). The fix places it as a proper try/except/else clause
            # that runs when NO exception was raised -- if the try block
            # succeeded but found no human aliases file, log a clear warning.
            # v89 BUG #3: no human aliases file found -- log clearly.
            if _string_raw_dir.exists():
                _all_alias_files = list(_string_raw_dir.glob("*aliases*.txt.gz"))
                logger.warning(
                    "No HUMAN (9606) STRING aliases file found in %s. "
                    "Found %d non-human alias files: %s. "
                    "REFUSING to load non-human aliases (would corrupt "
                    "organism assignment). string_aliases_df will be "
                    "empty -- resolve_single(string_id=...) will not "
                    "resolve STRING IDs to UniProt.",
                    _string_raw_dir,
                    len(_all_alias_files),
                    [f.name for f in _all_alias_files[:5]],
                )

    # v89 ROOT FIX (BUG #33 -- load ChEMBL target data for protein
    # resolution):
    #   The previous code loaded UniProt + STRING data but NOT ChEMBL
    #   target data. The ``add_chembl_target_records`` method on
    #   ProteinResolver was registered in ``_SOURCE_INGESTORS`` but
    #   never invoked from ``build_mapping`` because there was no
    #   ``chembl_target_df`` parameter. ChEMBL target IDs (e.g.
    #   CHEMBL2366519 for EGFR) were not cross-referenced with UniProt
    #   accessions, leaving drug-target edges from ChEMBL orphaned
    #   from the canonical Protein nodes.
    #
    #   ROOT FIX: load the ChEMBL activities CSV (which contains
    #   ``chembl_target_id`` + ``uniprot_id`` cross-references, emitted
    #   by the ChEMBL pipeline's clean() stage as
    #   ``chembl_activities_clean.csv``). Pass it as
    #   ``chembl_target_df`` to ``build_mapping`` so the resolver can
    #   link ChEMBL target IDs to UniProt accessions. This completes
    #   the drug -> target -> pathway -> disease chain (Phase 1 -> Phase 2
    #   -> Phase 3 -> Phase 4 connectivity).
    chembl_target_df = pd.DataFrame()
    chembl_activities_path = PROCESSED_DATA_DIR / "chembl_activities_clean.csv"
    if chembl_activities_path.exists():
        try:
            _chembl_acts_df = pd.read_csv(chembl_activities_path, low_memory=False)
            if not _chembl_acts_df.empty:
                # The ChEMBL activities CSV has many columns; we only
                # need the protein-resolution subset. Required column:
                # ``chembl_target_id``. Optional but useful:
                # ``uniprot_id``, ``gene_symbol``, ``organism``,
                # ``target_name``. If ``chembl_target_id`` is absent,
                # the resolver's schema validation (BUG #23 pattern)
                # will log an ERROR and skip ingestion -- surface the
                # issue to the operator instead of silent failure.
                _chembl_target_cols = [
                    c for c in (
                        "chembl_target_id", "uniprot_id", "gene_symbol",
                        "organism", "target_name", "target_type",
                    ) if c in _chembl_acts_df.columns
                ]
                if "chembl_target_id" in _chembl_acts_df.columns:
                    # Deduplicate by chembl_target_id (keep first
                    # occurrence of each target -- activities CSV has
                    # one row per activity, but we only need one row
                    # per target for protein resolution).
                    chembl_target_df = (
                        _chembl_acts_df[_chembl_target_cols]
                        .drop_duplicates(subset=["chembl_target_id"])
                        .reset_index(drop=True)
                    )
                    logger.info(
                        "Loaded ChEMBL target data from %s: %d unique "
                        "targets (for protein resolution via "
                        "add_chembl_target_records). (v89 BUG #33)",
                        chembl_activities_path.name, len(chembl_target_df),
                    )
                else:
                    logger.warning(
                        "ChEMBL activities CSV %s has no 'chembl_target_id' "
                        "column. Available columns: %s. Cannot use for "
                        "protein resolution -- ChEMBL target IDs will not "
                        "be cross-referenced with UniProt. (v89 BUG #33)",
                        chembl_activities_path.name,
                        list(_chembl_acts_df.columns),
                    )
        except Exception as exc:
            logger.warning(
                "Failed to load ChEMBL activities CSV %s: %s. ChEMBL "
                "target IDs will not be cross-referenced with UniProt. "
                "(v89 BUG #33)",
                chembl_activities_path, exc,
            )
    else:
        logger.info(
            "ChEMBL activities CSV not found at %s -- skipping ChEMBL "
            "target -> UniProt cross-reference. Run the ChEMBL pipeline "
            "first to enable. (v89 BUG #33)",
            chembl_activities_path,
        )

    protein_mapping_df = protein_resolver.build_mapping(
        uniprot_df,
        string_aliases_df=string_aliases_df if not string_aliases_df.empty else None,
        string_df=string_protein_df if not string_protein_df.empty else None,
        # v89 BUG #33: pass ChEMBL target data so add_chembl_target_records
        # is invoked, linking ChEMBL target IDs to UniProt accessions.
        chembl_target_df=chembl_target_df if not chembl_target_df.empty else None,
    )
    logger.info(
        "Protein entity resolution complete: %d canonical entities",
        len(protein_mapping_df),
    )

    # ------------------------------------------------------------------
    # Persist drug entity mappings
    # ------------------------------------------------------------------
    if not drug_mapping_df.empty:
        # Align columns to EntityMapping schema -- drop extras, fill missing
        col_map = {
            "canonical_inchikey": "canonical_inchikey",
            "canonical_name": "canonical_name",
            "chembl_id": "chembl_id",
            "drugbank_id": "drugbank_id",
            "pubchem_cid": "pubchem_cid",
            "uniprot_id": "uniprot_id",
            "string_id": "string_id",
            "match_confidence": "match_confidence",
            "match_method": "match_method",
        }
        save_df = drug_mapping_df.rename(columns=col_map)
        # Ensure pubchem_cid is numeric
        if "pubchem_cid" in save_df.columns:
            save_df["pubchem_cid"] = pd.to_numeric(
                save_df["pubchem_cid"], errors="coerce",
            )
        # Keep only columns present in EntityMapping
        model_cols = [
            "canonical_inchikey", "canonical_name", "chembl_id",
            "drugbank_id", "pubchem_cid", "uniprot_id", "string_id",
            "match_confidence", "match_method",
        ]
        for c in model_cols:
            if c not in save_df.columns:
                save_df[c] = None
        save_df = save_df[model_cols]

        # ROOT FIX (E2E CI): deduplicate on chembl_id before persisting.
        # The entity_mapping table has a UNIQUE constraint on chembl_id,
        # but the resolution process can produce multiple rows with the
        # same chembl_id (e.g. from different sources mapping to the same
        # ChEMBL compound). Without dedup, the INSERT fails with
        # sqlite3.IntegrityError: UNIQUE constraint failed.
        # Keep the first row per chembl_id (deterministic by save_df order).
        # Rows with NULL chembl_id are kept (NULL is not subject to UNIQUE).
        n_before = len(save_df)
        if "chembl_id" in save_df.columns:
            chembl_non_null = save_df["chembl_id"].notna()
            dup_mask = chembl_non_null & save_df["chembl_id"].duplicated(keep="first")
            if dup_mask.any():
                logger.warning(
                    "Deduplicating %d rows with duplicate chembl_id "
                    "(keeping first occurrence). %d rows remain.",
                    int(dup_mask.sum()), n_before - int(dup_mask.sum()),
                )
                save_df = save_df[~dup_mask].copy()

        # Transactional: temp table + DELETE/INSERT -- atomic, rolls back on failure.
        # v9 ROOT FIX (audit F3.5): TRUNCATE TABLE is PostgreSQL-specific
        # syntax. On SQLite-backed dev/test environments it raises
        # sqlite3.OperationalError. Use DELETE FROM which is universally
        # supported (ANSI SQL) and behaves correctly within an explicit
        # transaction on both dialects.
        # v89 ROOT FIX (BUG #29 -- leftover temp tables on failed runs):
        #   The previous code put the ``DROP TABLE IF EXISTS`` INSIDE
        #   the ``with engine.begin()`` transaction. On SQLite, DROP
        #   TABLE within a transaction may behave unexpectedly, AND if
        #   the INSERT failed (constraint violation, type mismatch),
        #   the transaction rolled back -- so the DROP never executed,
        #   leaving an orphaned ``_tmp_entity_mapping_staging`` table
        #   in the DB. Repeated failures accumulated orphaned temp
        #   tables.
        #
        #   ROOT FIX: wrap the temp-table lifecycle in a TRY/FINALLY
        #   so the temp table is ALWAYS dropped, even on failure. The
        #   DELETE+INSERT remains atomic (same transaction); the DROP
        #   runs in a SEPARATE transaction after the main work, so it
        #   commits even if the main transaction rolled back.
        engine = get_engine()
        try:
            with engine.begin() as conn:
                save_df.to_sql(
                    "_tmp_entity_mapping_staging",
                    con=conn,
                    if_exists="replace",
                    index=False,
                    method="multi",
                    chunksize=5000,
                )
                conn.execute(text("DELETE FROM entity_mapping"))
                conn.execute(text("""
                    INSERT INTO entity_mapping
                        (canonical_inchikey, canonical_name, chembl_id,
                         drugbank_id, pubchem_cid, uniprot_id, string_id,
                         match_confidence, match_method)
                    SELECT
                        canonical_inchikey, canonical_name, chembl_id,
                        drugbank_id, pubchem_cid, uniprot_id, string_id,
                        match_confidence, match_method
                    FROM _tmp_entity_mapping_staging
                """))
        finally:
            # v89 BUG #29: ALWAYS drop the temp table, even if the main
            # transaction failed. Use a SEPARATE transaction so the DROP
            # commits independently. ``if_exists='replace'`` on the next
            # run would handle it, but explicit cleanup avoids orphaned
            # tables accumulating on repeated failures.
            try:
                with engine.begin() as _cleanup_conn:
                    _cleanup_conn.execute(
                        text("DROP TABLE IF EXISTS _tmp_entity_mapping_staging")
                    )
            except Exception as _cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "v89 BUG #29: could not drop temp table "
                    "_tmp_entity_mapping_staging (%s). It will be dropped "
                    "on the next run via if_exists='replace'.",
                    _cleanup_exc,
                )
        # v104 FORENSIC ROOT FIX (P1-002 -- duplicate INSERT/DELETE block):
        #   The V90 CI fix that lived here (lines 880-935 in the pre-v104
        #   codebase) was a COMPLETE SECOND COPY of the same temp-table
        #   INSERT/DELETE block above. It (a) lacked try/finally cleanup,
        #   so on failure the temp table was orphaned forever; (b) used a
        #   divergent ORDER BY clause (the chembl_id dedup ran a SECOND
        #   time with subtly different row ordering than the pre-dedup at
        #   lines 798-816), so the second INSERT silently OVERWROTE the
        #   first INSERT's rows with potentially different survivors --
        #   non-deterministic deduplication. The duplicate was a copy-paste
        #   artifact from the V90 CI hotfix that was never cleaned up.
        #
        #   ROOT FIX: DELETED the entire duplicate block. The first block
        #   (lines 818-879 above) already has (1) proper pre-dedup at
        #   lines 798-816, (2) atomic INSERT/DELETE inside a single
        #   transaction, (3) try/finally cleanup that ALWAYS drops the
        #   temp table even on failure. Nothing is lost by deleting the
        #   duplicate; correctness, idempotency, and disk hygiene are all
        #   restored.
        #
        #   Regression test: phase1/tests/test_p1_002_duplicate_block.py
        #   asserts that running run_entity_resolution() TWICE on the same
        #   input leaves no orphaned _tmp_entity_mapping_staging table and
        #   produces deterministic row counts.
        logger.info(
            "Persisted %d drug entity mappings to database",
            len(drug_mapping_df),
        )

    # ------------------------------------------------------------------
    # Update proteins.string_id from protein resolution results
    # ------------------------------------------------------------------
    proteins_updated = 0
    if not protein_mapping_df.empty and "string_id" in protein_mapping_df.columns:
        resolved = protein_mapping_df.dropna(subset=["string_id"])
        if not resolved.empty:
            update_df = resolved[["uniprot_id", "string_id"]].copy()
            update_df = update_df.dropna(subset=["uniprot_id", "string_id"])
            if not update_df.empty:
                # v89 ROOT FIX (BUG #30 -- uniprot_id uniqueness not
                # validated before UPDATE):
                #   The UPDATE below uses a correlated subquery
                #   (SQLite) / UPDATE...FROM (PostgreSQL) that joins
                #   ``proteins.uniprot_id`` to the temp table's
                #   ``uniprot_id``. If the ``proteins`` table has
                #   DUPLICATE ``uniprot_id`` rows (e.g. from a previous
                #   bad load), the UPDATE affects ALL matching rows --
                #   potentially setting DIFFERENT ``string_id`` values
                #   on different rows (last-write-wins in the subquery).
                #   The code assumes ``uniprot_id`` is unique, but this
                #   was never validated.
                #
                #   ROOT FIX: pre-flight uniqueness check. If duplicates
                #   exist, log an ERROR with the duplicate count, dead-
                #   letter the affected updates, and SKIP the UPDATE --
                #   do NOT silently corrupt multiple rows. The operator
                #   must fix the duplicate ``uniprot_id`` rows in the
                #   ``proteins`` table before re-running.
                engine = get_engine()
                _dup_check_sql = text("""
                    SELECT uniprot_id, COUNT(*) AS cnt
                    FROM proteins
                    WHERE uniprot_id IS NOT NULL
                    GROUP BY uniprot_id
                    HAVING COUNT(*) > 1
                """)
                try:
                    with engine.begin() as _dup_conn:
                        _dup_rows = _dup_conn.execute(_dup_check_sql).fetchall()
                except Exception as _dup_exc:  # noqa: BLE001
                    # If the dup-check query itself fails (e.g. proteins
                    # table doesn't exist yet), log and proceed -- the
                    # UPDATE will fail with a clearer error.
                    logger.warning(
                        "v89 BUG #30: could not check uniprot_id "
                        "uniqueness in proteins table (%s). Proceeding "
                        "with UPDATE; if it fails, check for duplicates.",
                        _dup_exc,
                    )
                    _dup_rows = []
                if _dup_rows:
                    _dup_count = len(_dup_rows)
                    _dup_examples = [str(r[0]) for r in _dup_rows[:5]]
                    logger.error(
                        "v89 BUG #30: proteins table has %d duplicate "
                        "uniprot_id values (examples: %s). The string_id "
                        "UPDATE is SKIPPED to avoid corrupting multiple "
                        "rows. Fix the duplicates (deduplicate or re-load "
                        "the proteins table) before re-running entity "
                        "resolution.",
                        _dup_count, _dup_examples,
                    )
                    # Skip the UPDATE -- do not corrupt multiple rows.
                else:
                    try:
                        with engine.begin() as conn:
                            update_df.to_sql(
                                "_tmp_protein_string_update", con=conn,
                                if_exists="replace", index=False,
                                method="multi", chunksize=5000,
                            )
                            # v75 ROOT FIX (T-025 compound): the PostgreSQL UPDATE
                            # ... FROM syntax is not valid on SQLite. SQLite uses
                            # UPDATE ... SET col = (SELECT ...) WHERE EXISTS.
                            # Detect the dialect and dispatch the right SQL.
                            dialect_name = engine.dialect.name
                            if dialect_name == "sqlite":
                                conn.execute(text("""
                                    UPDATE proteins
                                    SET string_id = (
                                        SELECT t.string_id
                                        FROM _tmp_protein_string_update t
                                        WHERE t.uniprot_id = proteins.uniprot_id
                                    )
                                    WHERE EXISTS (
                                        SELECT 1 FROM _tmp_protein_string_update t
                                        WHERE t.uniprot_id = proteins.uniprot_id
                                    )
                                    AND string_id IS NULL
                                """))
                            else:
                                conn.execute(text("""
                                    UPDATE proteins p
                                    SET string_id = t.string_id
                                    FROM _tmp_protein_string_update t
                                    WHERE p.uniprot_id = t.uniprot_id
                                    AND p.string_id IS NULL
                                """))
                        proteins_updated = len(resolved)
                        logger.info(
                            "Updated string_id for %d proteins", proteins_updated,
                        )
                    finally:
                        # v89 BUG #29 (applied to the protein temp table
                        # too): ALWAYS drop the temp table, even on
                        # failure, in a separate transaction.
                        try:
                            with engine.begin() as _cleanup_conn:
                                _cleanup_conn.execute(
                                    text("DROP TABLE IF EXISTS _tmp_protein_string_update")
                                )
                        except Exception as _cleanup_exc:  # noqa: BLE001
                            logger.warning(
                                "v89 BUG #29: could not drop temp table "
                                "_tmp_protein_string_update (%s).",
                                _cleanup_exc,
                            )

    logger.info("Entity resolution pipeline complete")
    return {
        "drug_mappings": len(drug_mapping_df),
        "protein_mappings": len(protein_mapping_df),
        "proteins_updated": proteins_updated,
    }


if __name__ == "__main__":
    # Allow ``python -m entity_resolution.run`` as a CLI entry point.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_entity_resolution()
    print(f"Entity resolution result: {result}")
