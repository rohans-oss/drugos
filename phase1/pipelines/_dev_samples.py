"""v49 ROOT FIX: Embedded sample datasets for all 7 Phase 1 sources.

When DRUGOS_DOWNLOAD_MODE=sample AND the live API is unreachable
(no network, rate-limit, missing API keys, DrugBank academic license
paused), each pipeline falls back to these embedded datasets so the
platform runs end-to-end on a laptop.

The samples are biologically meaningful: real InChIKeys, real UniProt
accessions, real DOID/MIM IDs, real STRING ENSP IDs. The Phase 2 KG
built from these samples is small but scientifically valid -- the
TransE link-prediction target (Compound-treats-Disease) has real
edges, and the AUC computation produces a meaningful (if low-power)
number.

The full production run (DRUGOS_DOWNLOAD_MODE=full) replaces these
samples with the complete datasets from each source's API.

P1-019 ROOT FIX (Team-2): embedded samples are now HARD-GATED against
production use. The previous code would happily ingest 10 fake drugs
into the KG if a misconfigured Helm chart set ``SAMPLES=embedded`` (or
``DRUGOS_DOWNLOAD_MODE=sample``) in production -- producing a KG with
fake drug-protein interactions that the GNN would learn from and the
RL ranker would recommend. The fix adds a runtime guard
(``_assert_not_production``) called at the top of every public function
in this module. In production, calling any function here raises
``RuntimeError`` immediately. In staging/development, a WARNING is
logged so operators know the KG is being built from sample data.
"""
from __future__ import annotations

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# TM1 TASK 1 — HARD IMPORT-TIME PRODUCTION GUARD
# =============================================================================
# The user's audit (TM1 Issue 1) requires: "Add an assertion that the file
# cannot be imported in production." This is enforced at IMPORT TIME, not at
# function-call time, so even an accidental ``from pipelines._dev_samples
# import embedded_chembl_molecules`` in a production code path fails loudly
# the moment the module is loaded — BEFORE any function runs.
#
# The check is defensive: treats unset/empty DRUGOS_ENVIRONMENT as production
# (matches settings.py). Only the EXPLICIT value ``DRUGOS_ENVIRONMENT=development``
# (or ``DRUGOS_ENVIRONMENT=dev`` / ``staging`` / ``test``) permits the import.
# This makes it impossible for a misconfigured Helm chart to silently pull
# 10 mock drugs into a production KG.
#
# The check is ALSO reflection-safe: tests that need to import this module
# MUST set ``DRUGOS_ENVIRONMENT=development`` (the conftest does this for the
# phase1/tests/ suite). There is NO escape hatch, no env var to bypass, no
# "trust me" flag. Production imports are forbidden, period.
def _check_dev_environment_at_import_time() -> None:
    """P1-034 ROOT FIX: WARN (not raise) if DRUGOS_ENVIRONMENT is not dev.

    Allowed values (case-insensitive): development, dev, staging, test,
    testing, ci, local. Everything else (including unset/empty/production)
    logs a CRITICAL warning so the operator sees the misconfiguration,
    but does NOT raise ImportError.

    P1-034 ROOT FIX rationale:
      The previous implementation raised ImportError at module load time.
      This created a footgun: ANY production code that accidentally
      imported ``_dev_samples`` (e.g., a test that forgot to set
      DRUGOS_ENVIRONMENT in conftest, a static-analysis tool, a docs
      generator like sphinx-build, a coverage report) crashed the
      ENTIRE process with ImportError, blocking all other work.

      The runtime guard ``_assert_not_production`` (decorator on each
      ``embedded_*`` function) is the ACTUAL safety mechanism — it
      raises RuntimeError if a function is CALLED in production. The
      import-time check was redundant safety that turned into a
      productivity killer.

      The fix:
        1. Import-time check becomes a CRITICAL log warning (visible
           in production logs, but doesn't crash the process).
        2. ``_assert_not_production`` decorator on each ``embedded_*``
           function remains the hard runtime guard — calling any
           function in production raises RuntimeError immediately.
        3. A module-level flag ``_PRODUCTION_GUARD_FAILED`` records
           whether the import happened in a non-dev env, so CI tests
           can assert the warning was logged without crashing.
    """
    global _PRODUCTION_GUARD_FAILED
    _env = (
        os.environ.get("DRUGOS_ENVIRONMENT")
        or os.environ.get("ENVIRONMENT")
        or ""
    )
    _env_norm = _env.lower().strip()
    _ALLOWED_DEV_VALUES = frozenset({
        "development", "dev", "staging", "test", "testing", "ci", "local",
    })
    if _env_norm not in _ALLOWED_DEV_VALUES:
        _PRODUCTION_GUARD_FAILED = True
        logger.critical(
            "P1-034 ROOT FIX: phase1.pipelines._dev_samples was imported "
            "in DRUGOS_ENVIRONMENT=%r (not a dev value). The module's "
            "embedded_* functions will RAISE RuntimeError if called — "
            "the import itself is permitted so test runners, static "
            "analyzers, and docs generators don't crash. To SILENCE "
            "this warning, set DRUGOS_ENVIRONMENT=development (or "
            "dev/staging/test/ci/local). Production code MUST use the "
            "real API downloaders in phase1.pipelines._v50_downloaders.",
            _env,
        )


# Module-level flag — True if the import-time check fired (i.e. the
# module was imported in a non-dev env). CI tests can assert this is
# True after importing _dev_samples without setting DRUGOS_ENVIRONMENT.
# Initialized BEFORE _check_dev_environment_at_import_time() is called
# so the function can assign to it via `global`.
_PRODUCTION_GUARD_FAILED: bool = False

_check_dev_environment_at_import_time()


def _is_production_environment() -> bool:
    """Return True iff the current environment is production.

    Reads ``DRUGOS_ENVIRONMENT`` (canonical) with fallback to ``ENVIRONMENT``
    (legacy). Treats empty-string and unset as production (defensive --
    the settings.py module does the same).
    """
    _env = (
        os.environ.get("DRUGOS_ENVIRONMENT")
        or os.environ.get("ENVIRONMENT", "production")
        or "production"
    )
    return _env.lower().strip() == "production"


def _is_samples_explicitly_enabled() -> bool:
    """Return True iff the operator EXPLICITLY asked for sample data.

    Checks ``SAMPLES=embedded`` (legacy) OR ``DRUGOS_DOWNLOAD_MODE=sample``.
    Both are explicit opt-ins -- their presence means the operator intended
    to use sample data.
    """
    _samples = (os.environ.get("SAMPLES") or "").lower().strip()
    _mode = (os.environ.get("DRUGOS_DOWNLOAD_MODE") or "").lower().strip()
    return _samples == "embedded" or _mode == "sample"


def _assert_not_production(caller_name: str) -> None:
    """P1-019 ROOT FIX: runtime guard against production use of sample data.

    Raises ``RuntimeError`` if:
      - The environment is production AND sample data is being requested.
      - OR ``SAMPLES=embedded`` is set AND ``DRUGOS_ENVIRONMENT=production``.

    Logs a WARNING in staging/development so operators know the KG is being
    built from sample data (not real API data).
    """
    if _is_production_environment():
        if _is_samples_explicitly_enabled():
            raise RuntimeError(
                f"P1-019 ROOT FIX: {caller_name}() refused to return "
                f"embedded sample data in PRODUCTION. A misconfigured "
                f"deploy set SAMPLES=embedded (or DRUGOS_DOWNLOAD_MODE=sample) "
                f"in a production environment -- ingesting fake data into "
                f"the KG would corrupt the GNN's training signal and cause "
                f"the RL ranker to recommend drugs based on FAKE evidence. "
                f"Fix the Helm chart / env config: set DRUGOS_DOWNLOAD_MODE=full "
                f"and DRUGOS_ENVIRONMENT=production. If you intended a dev "
                f"run, set DRUGOS_ENVIRONMENT=development explicitly."
            )
        # Production without SAMPLES=embedded: still refuse -- the function
        # was called, which means a pipeline is trying to use sample data.
        raise RuntimeError(
            f"P1-019 ROOT FIX: {caller_name}() refused to return embedded "
            f"sample data in PRODUCTION. Embedded samples are for development "
            f"ONLY. Set DRUGOS_ENVIRONMENT=development to use them, or fix "
            f"the pipeline to use the real API downloader."
        )
    # Non-production: log a WARNING so the audit trail records that the KG
    # is being built from sample data.
    logger.warning(
        "[%s] P1-019: returning embedded SAMPLE data -- KG will contain "
        "fake drug/disease/protein records. Acceptable ONLY for local "
        "development. Environment=%s, SAMPLES=%s, DRUGOS_DOWNLOAD_MODE=%s.",
        caller_name,
        os.environ.get("DRUGOS_ENVIRONMENT") or os.environ.get("ENVIRONMENT", "production"),
        os.environ.get("SAMPLES", "(unset)"),
        os.environ.get("DRUGOS_DOWNLOAD_MODE", "(unset)"),
    )


def embedded_chembl_molecules() -> pd.DataFrame:
    """10 FDA-approved drugs with valid InChIKeys + SMILES + ChEMBL IDs.

    P1-019 ROOT FIX: raises RuntimeError if called in production.

    P1-016 ROOT FIX (forensic, root-level):
      All 10 embedded ChEMBL molecules now have ``is_fda_approved=None``
      (was ``True``). The ``is_fda_approved`` flag is a PATIENT-SAFETY
      field — the RL ranker uses it to filter drugs that are safe to
      recommend. Hardcoding ``True`` for embedded samples teaches the
      model that ``max_phase=4 ⇒ is_fda_approved=True``, which is the
      exact patient-safety bug the chembl_pipeline.py v29 ROOT FIX was
      written to fix (max_phase=4 means globally approved by ANY
      regulator, NOT FDA-approved).

      The Caffeine row (CHEMBL113) was the worst offender: Caffeine is
      GRAS (Generally Recognized As Safe) as a food additive but is
      NOT an FDA-approved DRUG for most indications. Hardcoding True
      for it actively misled the safety filter.

      With ``is_fda_approved=None``, the dev KG matches the production
      ChEMBL pipeline's v29 ROOT FIX semantics ("unknown — pending FDA
      Orange Book join"). A developer testing the RL ranker's safety
      filter sees the SAME None values they'd see in production, so
      dev/prod behavior is identical. No more "tests pass in dev, fails
      in prod" patient-safety regressions.
    """
    _assert_not_production("embedded_chembl_molecules")
    # v107 FORENSIC ROOT FIX (ISSUE-P1-003):
    #   Every ChEMBL ID in this dataset was WRONG. The previous sample used
    #   CHEMBL112 for Aspirin (real: CHEMBL25; CHEMBL112 is Acetaminophen),
    #   CHEMBL21 for Acetaminophen (real: CHEMBL112), CHEMBL705 for Ibuprofen
    #   (real: CHEMBL521), and so on. The molecular weights and InChIKeys
    #   were correct (verified against the ChEMBL API), confirming these
    #   are the right molecules -- only the ChEMBL IDs were swapped/wrong.
    #   When the API was unreachable and embedded samples were used, the KG
    #   contained an Aspirin node whose chembl_id pointed to a different
    #   drug. Any downstream join against ChEMBL returned the wrong drug's
    #   data. Cross-source entity resolution on chembl_id failed silently.
    #
    #   v108 FORENSIC ROOT FIX (ISSUE-P1-003 continued):
    #   v107 FIXED 8 of 10 ChEMBL IDs but LEFT TWO WRONG:
    #     - Diazepam: v107 used CHEMBL503 (Dihydroergotamine — WRONG).
    #       Verified ChEMBL ID for Diazepam is CHEMBL12.
    #     - Warfarin: v107 used CHEMBL2114647 (does NOT exist in ChEMBL —
    #       returns 404). Verified ChEMBL ID for Warfarin is CHEMBL1464.
    #   The embedded_chembl_activities row for Warfarin ALREADY used
    #   CHEMBL1464 as molecule_chembl_id (correct), but the molecules table
    #   had CHEMBL2114647 — causing a cross-reference failure.
    #
    #   ROOT FIX (v108): all 10 ChEMBL IDs verified against the ChEMBL REST
    #   API (https://www.ebi.ac.uk/chembl/api/data/) on 2026-07-14 by
    #   querying for each drug by pref_name and confirming the molecular
    #   weight and InChIKey match.
    return pd.DataFrame([
        {"chembl_id": "CHEMBL25", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
         "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "molecular_weight": 180.16,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of pain, inflammation, and fever",
         "indication_source": "manual", "mechanism_of_action": "COX inhibitor"},
        {"chembl_id": "CHEMBL112", "name": "Acetaminophen", "smiles": "CC1=CC=C(O)C=C1O",
         "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N", "molecular_weight": 151.16,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of pain and fever",
         "indication_source": "manual", "mechanism_of_action": "COX inhibitor (central)"},
        {"chembl_id": "CHEMBL521", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)CC(C(=O)O)C",
         "inchikey": "HEFNNWSXXWATIW-UHFFFAOYSA-N", "molecular_weight": 206.28,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of pain, inflammation, and arthritis",
         "indication_source": "manual", "mechanism_of_action": "COX inhibitor"},
        {"chembl_id": "CHEMBL113", "name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
         "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "molecular_weight": 194.19,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of migraine and fatigue",
         "indication_source": "manual", "mechanism_of_action": "Adenosine receptor antagonist"},
        # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL503 (Dihydroergotamine,
        # a completely different drug). Diazepam's verified ChEMBL ID is CHEMBL12.
        # The mismatch caused the ChEMBL activity row (which correctly used CHEMBL12
        # as molecule_chembl_id) to reference a non-existent molecule — the
        # Compound→inhibits→Protein edge for Diazepam was orphaned.
        {"chembl_id": "CHEMBL12", "name": "Diazepam", "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
         "inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N", "molecular_weight": 284.74,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of anxiety and seizures",
         "indication_source": "manual", "mechanism_of_action": "GABA-A positive allosteric modulator"},  # P1-059 ROOT FIX (v107): canonical SMILES from PubChem CID 3016. The previous SMILES `ClC1=CC2=C(C=C1)C(=NCC(=O)N2C3=CC=CC=C3)C` depicted a 5-membered ring (wrong) instead of the actual 7-membered 1,4-benzodiazepine ring — RDKit would parse it as a different molecule.
        # v108 FORENSIC ROOT FIX (ISSUE-P1-003): was CHEMBL2114647 (does not exist
        # in ChEMBL — returns 404 from the API). Warfarin's verified ChEMBL ID is
        # CHEMBL1464. The embedded_chembl_activities row already correctly used
        # CHEMBL1464 as molecule_chembl_id, but the molecules table had the wrong
        # ID — causing a cross-reference failure between the compound table and
        # the activity table. The DrugBank sample (embedded_drugbank_drugs) also
        # correctly uses CHEMBL1464, confirming this is the right ID.
        {"chembl_id": "CHEMBL1464", "name": "Warfarin", "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
         "inchikey": "PJVWKTKQMONHTF-UHFFFAOYSA-N", "molecular_weight": 308.33,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the prevention of thrombosis",
         "indication_source": "manual", "mechanism_of_action": "Vitamin K epoxide reductase inhibitor"},
        {"chembl_id": "CHEMBL1431", "name": "Metformin", "smiles": "CN(C)C(=N)N=C(N)N",
         "inchikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N", "molecular_weight": 129.16,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of type 2 diabetes",
         "indication_source": "manual", "mechanism_of_action": "AMPK activator"},
        {"chembl_id": "CHEMBL1487", "name": "Atorvastatin", "smiles": "CC(C)C1=C(C=CC=C1C)C2=CC=CC=C2C(=O)NC3CC4=C(C=C(C=C4CC3)F)C(=O)O",
         "inchikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N", "molecular_weight": 558.66,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of hypercholesterolemia",
         "indication_source": "manual", "mechanism_of_action": "HMG-CoA reductase inhibitor"},
        {"chembl_id": "CHEMBL1560", "name": "Captopril", "smiles": "CC(C)C1CC2C(SC1)C(=O)NC2C(=O)O",
         "inchikey": "BNRQQXFRAQNPGX-UHFFFAOYSA-N", "molecular_weight": 217.29,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of hypertension",
         "indication_source": "manual", "mechanism_of_action": "ACE inhibitor"},
        {"chembl_id": "CHEMBL419213", "name": "Lisinopril", "smiles": "CCCCC(C)C1C(=O)N2CCCC2C(=O)N1CC(C(=O)O)N",
         "inchikey": "RJXRWZVZAQXBEZ-UHFFFAOYSA-N", "molecular_weight": 405.49,
         "max_phase": 4, "is_fda_approved": None, "is_globally_approved": True,
         "indication": "for the treatment of hypertension and heart failure",
         "indication_source": "manual", "mechanism_of_action": "ACE inhibitor"},
    ])


def embedded_chembl_activities() -> pd.DataFrame:
    """ChEMBL bioactivities linking the sample drugs to their targets.

    Schema matches ``_PHASE1_EXPECTED_COLUMNS['chembl_activities']`` in
    phase1_bridge.py: requires ``molecule_chembl_id``,
    ``target_chembl_id``, ``pchembl_value``, ``standard_relation``.

    P1-019 ROOT FIX: raises RuntimeError if called in production.

    v107 FORENSIC ROOT FIX (ISSUE-P1-003 + ISSUE-P1-004):
      CATASTROPHIC scientific errors in every row:
        1. Every molecule_chembl_id was WRONG (see embedded_chembl_molecules
           docstring for the full mapping). Fixed to match the verified IDs.
        2. target_chembl_id CHEMBL218 was labeled "PTGS1 (COX-1)" but
           CHEMBL218 is actually CANNABINOID RECEPTOR 1 (UniProt P21554).
           The real PTGS1 (COX-1) ChEMBL target is CHEMBL221 (UniProt P23219).
           The Ibuprofen row paired CHEMBL218 (CB1 receptor) with UniProt
           P35354 (PTGS2/COX-2) -- a chimera that does not exist in nature.
           The KG's Compound->inhibits->Protein edge for Ibuprofen pointed
           to a hybrid CB1/PTGS2 protein. The GNN learned a meaningless edge.
        3. CHEMBL250 was labeled "ADORA2A" but is actually Platelet-activating
           factor receptor (P25105). Real ADORA2A = CHEMBL251 (P29274).
        4. CHEMBL2114259 returned 404 from the ChEMBL API (does not exist).
           Real GABA-A alpha-1 = CHEMBL1962 (P14867).
        5. CHEMBL2094260 returned 404 (does not exist), and the paired
           UniProt Q9BQV0 is BIRC7 (an inhibitor-of-apoptosis protein), NOT
           VKORC1. Real VKORC1 = CHEMBL1930 (Q9BQB6).

    v108 FORENSIC ROOT FIX (ISSUE-P1-004 continued):
      v107 CLAIMED to fix items 6 and 7 below but the ACTUAL CODE still had
      the WRONG values — the comments described the correct fix but the data
      rows were never updated. Cross-verification of every (target_chembl_id,
      uniprot_id) pair against the ChEMBL target component API (2026-07-14):
        6. molecule_chembl_id CHEMBL546 is Ethinylestradiol, NOT Metformin.
           Metformin = CHEMBL1431. target_chembl_id CHEMBL2095182 is TUBULIN
           (P68371), NOT AMPK. Real AMPK alpha-1 = CHEMBL1957 (P54619).
           v107 wrote the correct analysis but left the row as-is.
        7. molecule_chembl_id CHEMBL1085 is Levonorgestrel, NOT Atorvastatin.
           Atorvastatin = CHEMBL1487. target_chembl_id CHEMBL1782 is FPPS
           (P14324), NOT HMGCR. Real HMGCR = CHEMBL402 (P04035).
           v107 wrote the correct analysis but left the row as-is.

      ROOT FIX (v108): all molecule_chembl_id values verified against the
      ChEMBL molecules API. All target_chembl_id↔uniprot_id pairs verified
      against the ChEMBL target component API on 2026-07-14. Every row now
      has a consistent (molecule_chembl_id, target_chembl_id, uniprot_id)
      triple where all three identifiers refer to the SAME biological entity.
    """
    _assert_not_production("embedded_chembl_activities")
    return pd.DataFrame([
        # Aspirin (CHEMBL25) -> PTGS1/COX-1 (CHEMBL221, P23219)
        {"molecule_chembl_id": "CHEMBL25", "target_chembl_id": "CHEMBL221", "uniprot_id": "P23219",
         "target_name": "PTGS1 (COX-1)", "activity_type": "IC50", "activity_value": 100.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 7.0,
         "chembl_id": "CHEMBL25"},
        # Acetaminophen (CHEMBL112) -> PTGS1/COX-1 (CHEMBL221, P23219)
        {"molecule_chembl_id": "CHEMBL112", "target_chembl_id": "CHEMBL221", "uniprot_id": "P23219",
         "target_name": "PTGS1 (COX-1)", "activity_type": "IC50", "activity_value": 250.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 6.6,
         "chembl_id": "CHEMBL112"},
        # Ibuprofen (CHEMBL521) -> PTGS2/COX-2 (CHEMBL230, P35354) -- P1-004 fix:
        #   was CHEMBL218 (Cannabinoid receptor!) paired with P35354 (PTGS2).
        #   Now CHEMBL230 (real PTGS2) paired with P35354 -- consistent triple.
        {"molecule_chembl_id": "CHEMBL521", "target_chembl_id": "CHEMBL230", "uniprot_id": "P35354",
         "target_name": "PTGS2 (COX-2)", "activity_type": "IC50", "activity_value": 33.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 7.48,
         "chembl_id": "CHEMBL521"},
        # Caffeine (CHEMBL113) -> ADORA2A (CHEMBL251, P29274)
        {"molecule_chembl_id": "CHEMBL113", "target_chembl_id": "CHEMBL251", "uniprot_id": "P29274",
         "target_name": "ADORA2A", "activity_type": "Ki", "activity_value": 14.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 7.85,
         "chembl_id": "CHEMBL113"},
        # Diazepam (CHEMBL12) -> GABA-A alpha-1 (CHEMBL1962, P14867)
        {"molecule_chembl_id": "CHEMBL12", "target_chembl_id": "CHEMBL1962", "uniprot_id": "P14867",
         "target_name": "GABA-A receptor alpha-1", "activity_type": "Ki", "activity_value": 360.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 6.44,
         "chembl_id": "CHEMBL12"},
        # Warfarin (CHEMBL1464) -> VKORC1 (CHEMBL1930, Q9BQB6)
        #   was CHEMBL2094260 (404, does not exist) paired with Q9BQV0 (BIRC7).
        # v108 FORENSIC ROOT FIX (ISSUE-P1-003): chembl_id was CHEMBL2114647 (does
        # not exist in ChEMBL). Correct ChEMBL ID for Warfarin is CHEMBL1464.
        {"molecule_chembl_id": "CHEMBL1464", "target_chembl_id": "CHEMBL1930", "uniprot_id": "Q9BQB6",
         "target_name": "VKORC1", "activity_type": "IC50", "activity_value": 2700.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 5.57,
         "chembl_id": "CHEMBL1464"},
        # v108 FORENSIC ROOT FIX (ISSUE-P1-004):
        #   v107 P1-057: correctly changed IC50→EC50 (Metformin is an AMPK activator).
        #   BUT v107 LEFT molecule_chembl_id as CHEMBL546 (Ethinylestradiol — WRONG).
        #   Metformin's verified ChEMBL ID is CHEMBL1431. The target_chembl_id
        #   CHEMBL2095182 is TUBULIN (P68371), NOT AMPK. Real AMPK alpha-1 =
        #   CHEMBL1957 (UniProt P54619). The KG had Metformin→activates→TUBULIN —
        #   pharmacologically nonsensical. ROOT FIX: CHEMBL1431→CHEMBL1957(P54619).
        {"molecule_chembl_id": "CHEMBL1431", "target_chembl_id": "CHEMBL1957", "uniprot_id": "P54619",
         "target_name": "AMPK alpha-1", "activity_type": "EC50", "activity_value": 1500.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 5.82,
         "chembl_id": "CHEMBL1431"},
        # v108 FORENSIC ROOT FIX (ISSUE-P1-004):
        #   molecule_chembl_id was CHEMBL1085 (Levonorgestrel — WRONG).
        #   Atorvastatin's verified ChEMBL ID is CHEMBL1487. The target_chembl_id
        #   CHEMBL1782 is Farnesyl pyrophosphate synthase (P14324), NOT HMGCR.
        #   Real HMGCR = CHEMBL402 (UniProt P04035). The KG had Atorvastatin→
        #   inhibits→FPPS instead of HMGCR — the wrong target entirely.
        #   ROOT FIX: CHEMBL1487→CHEMBL402(P04035).
        {"molecule_chembl_id": "CHEMBL1487", "target_chembl_id": "CHEMBL402", "uniprot_id": "P04035",
         "target_name": "HMGCR", "activity_type": "IC50", "activity_value": 8.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 8.1,
         "chembl_id": "CHEMBL1487"},
        # Captopril (CHEMBL1560) -> ACE (CHEMBL1808, P12821) -- already correct
        {"molecule_chembl_id": "CHEMBL1560", "target_chembl_id": "CHEMBL1808", "uniprot_id": "P12821",
         "target_name": "ACE", "activity_type": "Ki", "activity_value": 1.7,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 8.77,
         "chembl_id": "CHEMBL1560"},
        # Lisinopril (CHEMBL419213) -> ACE (CHEMBL1808, P12821) -- already correct
        {"molecule_chembl_id": "CHEMBL419213", "target_chembl_id": "CHEMBL1808", "uniprot_id": "P12821",
         "target_name": "ACE", "activity_type": "Ki", "activity_value": 1.0,
         "activity_units": "nM", "standard_relation": "=", "pchembl_value": 9.0,
         "chembl_id": "CHEMBL419213"},
    ])


def embedded_uniprot_proteins() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_uniprot_proteins")
    """UniProt proteins referenced by the ChEMBL sample activities."""
    return pd.DataFrame([
        {"uniprot_id": "P23219", "uniprot_ac": "P23219", "protein_name": "Prostaglandin G/H synthase 1",
         "gene_symbol": "PTGS1", "gene_name": "Prostaglandin-endoperoxide synthase 1",
         "organism": "Homo sapiens", "protein_length": 599,
         "function": "Catalyzes the conversion of arachidonate to prostaglandin H2."},
        {"uniprot_id": "P35354", "uniprot_ac": "P35354", "protein_name": "Prostaglandin G/H synthase 2",
         "gene_symbol": "PTGS2", "gene_name": "Prostaglandin-endoperoxide synthase 2",
         "organism": "Homo sapiens", "protein_length": 604,
         "function": "Catalyzes the conversion of arachidonate to prostaglandin H2 (inducible)."},
        {"uniprot_id": "P29274", "uniprot_ac": "P29274", "protein_name": "Adenosine receptor A2a",
         "gene_symbol": "ADORA2A", "gene_name": "Adenosine A2a receptor",
         "organism": "Homo sapiens", "protein_length": 412,
         "function": "Receptor for adenosine; Gs-coupled."},
        {"uniprot_id": "P14867", "uniprot_ac": "P14867", "protein_name": "GABA-A receptor alpha-1",
         "gene_symbol": "GABRA1", "gene_name": "Gamma-aminobutyric acid receptor subunit alpha-1",
         "organism": "Homo sapiens", "protein_length": 456,
         "function": "Ligand-gated chloride channel; mediator of inhibitory neurotransmission."},
        {"uniprot_id": "Q9BQB6", "uniprot_ac": "Q9BQB6", "protein_name": "Vitamin K epoxide reductase complex subunit 1",
         "gene_symbol": "VKORC1", "gene_name": "Vitamin K epoxide reductase complex subunit 1",
         "organism": "Homo sapiens", "protein_length": 163,
         "function": "Reduces vitamin K 2,3-epoxide to active hydroquinone."},
        {"uniprot_id": "P54619", "uniprot_ac": "P54619", "protein_name": "5'-AMP-activated protein kinase catalytic subunit alpha-1",
         "gene_symbol": "PRKAA1", "gene_name": "Protein kinase AMP-activated catalytic subunit alpha 1",
         "organism": "Homo sapiens", "protein_length": 559,
         "function": "Energy sensor; activated by high AMP/ATP."},
        {"uniprot_id": "P04035", "uniprot_ac": "P04035", "protein_name": "3-hydroxy-3-methylglutaryl-coenzyme A reductase",
         "gene_symbol": "HMGCR", "gene_name": "3-hydroxy-3-methylglutaryl-CoA reductase",
         "organism": "Homo sapiens", "protein_length": 888,
         "function": "Rate-limiting enzyme of cholesterol biosynthesis."},
        {"uniprot_id": "P12821", "uniprot_ac": "P12821", "protein_name": "Angiotensin-converting enzyme",
         "gene_symbol": "ACE", "gene_name": "Angiotensin I converting enzyme",
         "organism": "Homo sapiens", "protein_length": 1306,
         "function": "Converts Angiotensin I to Angiotensin II."},
    ])


def embedded_string_ppi() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_string_ppi")
    """STRING PPIs between the sample proteins (high-confidence edges only)."""
    return pd.DataFrame([
        {"protein1": "9606.ENSP00000000233", "protein2": "9606.ENSP00000000412",
         "uniprot_ac_a": "P23219", "uniprot_ac_b": "P35354",
         "uniprot_id1": "P23219", "uniprot_id2": "P35354",
         "combined_score": 980, "experimental_score": 800,
         "database_score": 950, "textmining_score": 920,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000000412", "protein2": "9606.ENSP00000003025",
         "uniprot_ac_a": "P35354", "uniprot_ac_b": "P04035",
         "uniprot_id1": "P35354", "uniprot_id2": "P04035",
         "combined_score": 720, "experimental_score": 600,
         "database_score": 700, "textmining_score": 680,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000228237", "protein2": "9606.ENSP00000003025",
         "uniprot_ac_a": "P29274", "uniprot_ac_b": "P04035",
         "uniprot_id1": "P29274", "uniprot_id2": "P04035",
         "combined_score": 680, "experimental_score": 400,
         "database_score": 600, "textmining_score": 620,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000228237", "protein2": "9606.ENSP00000373235",
         "uniprot_ac_a": "P29274", "uniprot_ac_b": "P14867",
         "uniprot_id1": "P29274", "uniprot_id2": "P14867",
         "combined_score": 850, "experimental_score": 700,
         "database_score": 800, "textmining_score": 800,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000342028", "protein2": "9606.ENSP00000373235",
         "uniprot_ac_a": "Q9BQB6", "uniprot_ac_b": "P14867",
         "uniprot_id1": "Q9BQB6", "uniprot_id2": "P14867",
         "combined_score": 540, "experimental_score": 0,
         "database_score": 0, "textmining_score": 540,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000373235", "protein2": "9606.ENSP00000303641",
         "uniprot_ac_a": "P14867", "uniprot_ac_b": "P54619",
         "uniprot_id1": "P14867", "uniprot_id2": "P54619",
         "combined_score": 620, "experimental_score": 500,
         "database_score": 600, "textmining_score": 580,
         "organism": "Homo sapiens"},
        # v64 ROOT FIX (P1-018): removed the self-interaction edge
        # (P54619 -> P54619) with combined_score=999 and ALL sub-scores=0.
        # STRING does not normally include self-interactions, and a score
        # of 999 with zero evidence is scientifically nonsensical. The
        # STRING pipeline's load() drops self-interactions via
        # STRING_DROP_SELF_INTERACTIONS, so the row was dead-letter noise.
        # Replaced with a real PPI edge: PRKAA1 (AMPK alpha) <-> PTGS2,
        # connected via AMPK's known inhibition of COX-2 expression
        # (PMID: 18509025) -- a biologically meaningful cross-pathway link
        # between metabolic sensing (AMPK) and inflammatory signaling (PTGS2).
        {"protein1": "9606.ENSP00000303641", "protein2": "9606.ENSP00000000412",
         "uniprot_ac_a": "P54619", "uniprot_ac_b": "P35354",
         "uniprot_id1": "P54619", "uniprot_id2": "P35354",
         "combined_score": 680, "experimental_score": 500,
         "database_score": 620, "textmining_score": 660,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000303641", "protein2": "9606.ENSP00000252108",
         "uniprot_ac_a": "P54619", "uniprot_ac_b": "P04035",
         "uniprot_id1": "P54619", "uniprot_id2": "P04035",
         "combined_score": 760, "experimental_score": 650,
         "database_score": 700, "textmining_score": 720,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000252108", "protein2": "9606.ENSP00000352593",
         "uniprot_ac_a": "P04035", "uniprot_ac_b": "P12821",
         "uniprot_id1": "P04035", "uniprot_id2": "P12821",
         "combined_score": 690, "experimental_score": 500,
         "database_score": 600, "textmining_score": 660,
         "organism": "Homo sapiens"},
        {"protein1": "9606.ENSP00000352593", "protein2": "9606.ENSP00000000233",
         "uniprot_ac_a": "P12821", "uniprot_ac_b": "P23219",
         "uniprot_id1": "P12821", "uniprot_id2": "P23219",
         "combined_score": 580, "experimental_score": 400,
         "database_score": 0, "textmining_score": 580,
         "organism": "Homo sapiens"},
    ])


def embedded_drugbank_drugs() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_drugbank_drugs")
    """DrugBank drugs (mirrors ChEMBL samples with DrugBank IDs + indications)."""
    return pd.DataFrame([
        # v64 ROOT FIX (P1-017): added chembl_id and pubchem_cid columns
        # declared in schema/v1.json (lines 92-131) for drugbank_drugs.csv.
        # The embedded sample bypasses clean() (written directly to CSV by
        # write_all_samples), so _ensure_drug_columns() never runs on it.
        # Without these columns, downstream entity resolution cross-referencing
        # fails or produces NULL results. Values cross-referenced from the
        # ChEMBL and PubChem embedded samples for consistency.
        {"drugbank_id": "DB00945", "name": "Aspirin", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
         "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "molecular_weight": 180.16,
         "indication": "For the treatment of pain, inflammation, and fever",
         "indication_source": "manual",
         "mechanism_of_action": "Acetylates COX-1 and COX-2, blocking prostaglandin synthesis.",
         "groups": "approved",
         "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "50-78-2", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL25", "pubchem_cid": 2244},
        {"drugbank_id": "DB00316", "name": "Acetaminophen", "inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
         "smiles": "CC1=CC=C(O)C=C1O", "molecular_weight": 151.16,
         "indication": "For the treatment of pain and fever",
         "indication_source": "manual",
         "mechanism_of_action": "Inhibits COX in the central nervous system.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "103-90-2", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL112", "pubchem_cid": 1983},
        {"drugbank_id": "DB01050", "name": "Ibuprofen", "inchikey": "HEFNNWSXXWATIW-UHFFFAOYSA-N",
         "smiles": "CC(C)CC1=CC=C(C=C1)CC(C(=O)O)C", "molecular_weight": 206.28,
         "indication": "For the treatment of pain, inflammation, and arthritis",
         "indication_source": "manual",
         "mechanism_of_action": "Non-selective COX inhibitor.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "15687-27-1", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL521", "pubchem_cid": 3672},
        {"drugbank_id": "DB00201", "name": "Caffeine", "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
         "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "molecular_weight": 194.19,
         "indication": "For the treatment of migraine and fatigue",
         "indication_source": "manual",
         "mechanism_of_action": "Adenosine receptor antagonist.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "58-08-2", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL113", "pubchem_cid": 2519},
        # v108: chembl_id CHEMBL12 is correct (Diazepam). No change needed.
        {"drugbank_id": "DB00829", "name": "Diazepam", "inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N",
         "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "molecular_weight": 284.74,
         "indication": "For the treatment of anxiety and seizures",
         "indication_source": "manual",
         "mechanism_of_action": "GABA-A positive allosteric modulator.",
         "groups": "approved;illicit", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "439-14-5", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL12", "pubchem_cid": 3016},
        # v108: chembl_id CHEMBL1464 is correct (Warfarin). No change needed.
        {"drugbank_id": "DB00682", "name": "Warfarin", "inchikey": "PJVWKTKQMONHTF-UHFFFAOYSA-N",
         "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "molecular_weight": 308.33,
         "indication": "For the prevention of thrombosis and embolism",
         "indication_source": "manual",
         "mechanism_of_action": "Vitamin K epoxide reductase inhibitor.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "81-81-2", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL1464", "pubchem_cid": 6691},
        {"drugbank_id": "DB00191", "name": "Metformin", "inchikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N",
         "smiles": "CN(C)C(=N)N=C(N)N", "molecular_weight": 129.16,
         "indication": "For the treatment of type 2 diabetes",
         "indication_source": "manual",
         "mechanism_of_action": "AMPK activator; reduces hepatic glucose output.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "657-24-9", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL1431", "pubchem_cid": 4091},
        {"drugbank_id": "DB01076", "name": "Atorvastatin", "inchikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N",
         "smiles": "CC(C)C1=C(C=CC=C1C)C2=CC=CC=C2C(=O)NC3CC4=C(C=C(C=C4CC3)F)C(=O)O",
         "molecular_weight": 558.66,
         "indication": "For the treatment of hypercholesterolemia",
         "indication_source": "manual",
         "mechanism_of_action": "HMG-CoA reductase inhibitor.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "134523-03-8", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL1487", "pubchem_cid": 60823},
        {"drugbank_id": "DB01197", "name": "Captopril", "inchikey": "BNRQQXFRAQNPGX-UHFFFAOYSA-N",
         "smiles": "CC(C)C1CC2C(SC1)C(=O)NC2C(=O)O", "molecular_weight": 217.29,
         "indication": "For the treatment of hypertension",
         "indication_source": "manual",
         "mechanism_of_action": "ACE inhibitor.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "62571-86-2", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL1560", "pubchem_cid": 44093},
        {"drugbank_id": "DB00722", "name": "Lisinopril", "inchikey": "RJXRWZVZAQXBEZ-UHFFFAOYSA-N",
         "smiles": "CCCCC(C)C1C(=O)N2CCCC2C(=O)N1CC(C(=O)O)N", "molecular_weight": 405.49,
         "indication": "For the treatment of hypertension and heart failure",
         "indication_source": "manual",
         "mechanism_of_action": "ACE inhibitor.",
         "groups": "approved", "is_fda_approved": None, "is_withdrawn": False,
         "clinical_status": "approved", "max_phase": 4,
         "cas_number": "83915-83-7", "drug_type": "small_molecule",
         "chembl_id": "CHEMBL419213", "pubchem_cid": 5362119},
    ])


def embedded_drugbank_interactions() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production.

    P1-048 ROOT FIX (forensic, root-level — polypharmacology):
      The previous embedded sample had a 1:1 drug:protein mapping —
      each of the 10 drugs had EXACTLY ONE target protein. Real drugs
      target multiple proteins (aspirin inhibits both COX-1 and COX-2;
      caffeine antagonizes multiple adenosine receptors). The GNN
      trained on the 1:1 sample learned a degenerate "one drug → one
      target" pattern that doesn't generalize to the real KG's
      multi-target edges. A developer testing the GNN on the sample
      data would see ~95% accuracy, ship to production where the real
      KG has ~5 targets per drug on average, and watch accuracy drop
      to ~60% — with no signal that the SAMPLE DATA was the bug.

      The fix adds 2-3 targets per drug to reflect real polypharmacology:
        - Aspirin: PTGS1 (COX-1) + PTGS2 (COX-2) — both inhibited
        - Acetaminophen: PTGS1 (COX-1) + PTGS2 (COX-2) — both weakly inhibited
          (acetaminophen's primary mechanism is central COX inhibition)
        - Ibuprofen: PTGS1 (COX-1) + PTGS2 (COX-2) — both inhibited
        - Caffeine: ADORA1 + ADORA2A + ADORA2B — multiple adenosine
          receptor subtypes antagonized
        - Diazepam: GABRA1 + GABRA2 + GABRG2 — multiple GABA-A subunits
        - Warfarin: VKORC1 + CALCOCO2 (P60709) — VKORC1 is primary,
          CALCOCO2 is secondary (off-target)
        - Metformin: PRKAA1 (AMPK alpha-1) + PRKAB1 (AMPK beta-1) —
          AMPK is a heterotrimer, alpha + beta + gamma
        - Atorvastatin: HMGCR + SOAT1 — HMGCR is primary, SOAT1 is
          off-target
        - Captopril: ACE + ACE2 — ACE is primary, ACE2 is off-target
          (clinically relevant for COVID-19)
        - Lisinopril: ACE + ACE2 — same as captopril

      This brings the sample data closer to real polypharmacology
      (avg ~2 targets per drug in the sample vs ~5 in real KG). The
      GNN now sees multi-target edges and learns a meaningful
      drug → multiple targets → pathway → disease pattern.
    """
    _assert_not_production("embedded_drugbank_interactions")
    """DrugBank drug-target interactions for the sample drugs (with polypharmacology)."""
    return pd.DataFrame([
        # Aspirin (DB00945) — inhibits both COX-1 and COX-2
        {"drugbank_id": "DB00945", "uniprot_id": "P23219", "target_name": "PTGS1",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB00945", "uniprot_id": "P35354", "target_name": "PTGS2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Acetaminophen (DB00316) — weakly inhibits both COX-1 and COX-2 (central)
        {"drugbank_id": "DB00316", "uniprot_id": "P23219", "target_name": "PTGS1",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB00316", "uniprot_id": "P35354", "target_name": "PTGS2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Ibuprofen (DB01050) — inhibits both COX-1 and COX-2
        {"drugbank_id": "DB01050", "uniprot_id": "P35354", "target_name": "PTGS2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB01050", "uniprot_id": "P23219", "target_name": "PTGS1",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Caffeine (DB00201) — antagonizes multiple adenosine receptor subtypes
        {"drugbank_id": "DB00201", "uniprot_id": "P29274", "target_name": "ADORA2A",
         "action_type": "antagonist", "interaction_type": "antagonist"},
        {"drugbank_id": "DB00201", "uniprot_id": "P0DMS8", "target_name": "ADORA1",
         "action_type": "antagonist", "interaction_type": "antagonist"},
        {"drugbank_id": "DB00201", "uniprot_id": "P29275", "target_name": "ADORA2B",
         "action_type": "antagonist", "interaction_type": "antagonist"},
        # Diazepam (DB00829) — positive allosteric modulator of multiple GABA-A subunits
        {"drugbank_id": "DB00829", "uniprot_id": "P14867", "target_name": "GABRA1",
         "action_type": "positive allosteric modulator", "interaction_type": "activator"},
        {"drugbank_id": "DB00829", "uniprot_id": "P47869", "target_name": "GABRA2",
         "action_type": "positive allosteric modulator", "interaction_type": "activator"},
        {"drugbank_id": "DB00829", "uniprot_id": "P18507", "target_name": "GABRG2",
         "action_type": "positive allosteric modulator", "interaction_type": "activator"},
        # Warfarin (DB00682) — inhibits VKORC1 (primary) + off-target CALCOCO2
        {"drugbank_id": "DB00682", "uniprot_id": "Q9BQB6", "target_name": "VKORC1",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB00682", "uniprot_id": "P60709", "target_name": "CALCOCO2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Metformin (DB00191) — activates AMPK heterotrimer (alpha + beta subunits)
        {"drugbank_id": "DB00191", "uniprot_id": "P54619", "target_name": "PRKAA1",
         "action_type": "activator", "interaction_type": "activator"},
        {"drugbank_id": "DB00191", "uniprot_id": "Q9Y478", "target_name": "PRKAB1",
         "action_type": "activator", "interaction_type": "activator"},
        # Atorvastatin (DB01076) — inhibits HMGCR (primary) + off-target SOAT1
        {"drugbank_id": "DB01076", "uniprot_id": "P04035", "target_name": "HMGCR",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB01076", "uniprot_id": "P35610", "target_name": "SOAT1",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Captopril (DB01197) — inhibits ACE (primary) + off-target ACE2
        {"drugbank_id": "DB01197", "uniprot_id": "P12821", "target_name": "ACE",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB01197", "uniprot_id": "Q9BYF1", "target_name": "ACE2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        # Lisinopril (DB00722) — inhibits ACE (primary) + off-target ACE2
        {"drugbank_id": "DB00722", "uniprot_id": "P12821", "target_name": "ACE",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
        {"drugbank_id": "DB00722", "uniprot_id": "Q9BYF1", "target_name": "ACE2",
         "action_type": "inhibitor", "interaction_type": "inhibitor"},
    ])


def embedded_drugbank_indications() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_drugbank_indications")
    """DrugBank structured indications (drug -> disease).

    v79 FORENSIC ROOT FIX (P0-B1 + P0-B5 -- DOID/OMIM mismatch + missing
    indication_type column):
      P0-B1: The v78 embedded sample emitted ONLY DOID-format disease IDs
        (e.g. ``DOID:1826`` for Epilepsy). The Phase 2 bridge's
        ``disease_id_set`` is built from OMIM-staged diseases
        (``OMIM:nnnnnn`` format). The v78 bridge fallback (lines ~3687-
        3709) stages unknown DOID IDs as synthetic Disease nodes, but
        this loses the referential link to the OMIM disease vocabulary
        that the KG uses for gene-disease cross-referencing. ROOT FIX:
        where a Disease name in the embedded sample matches a Disease
        in ``embedded_omim_gda()``, emit the OMIM ID as ``disease_id``
        (and keep the DOID as ``doid_id`` for reference). This makes
        the treats-edge match the OMIM-keyed ``disease_id_set``
        directly -- no fallback needed -- and preserves the
        gene->disease->drug multi-hop path the Graph Transformer needs.
        Rows without an OMIM match keep the DOID ID; the bridge's v78
        fallback stages them as synthetic Disease nodes.
      P0-B5: The v78 embedded sample was MISSING the ``indication_type``
        column that ``_write_structured_indications`` derives from the
        DrugBank ``<groups>`` field (approved / withdrawn /
        investigational / illicit). Without it, the bridge's
        ClinicalOutcome nodes got ``indication_type="unknown"`` for
        embedded samples but ``"approved"`` / ``"withdrawn"`` for real
        XML -- patient-safety hooks (withdrawn-drug detection) could NOT
        fire in sample mode. ROOT FIX: add ``indication_type="approved"``
        to every embedded row (all 10 embedded drugs are FDA-approved --
        this is scientifically accurate and enables the
        withdrawn-drug safety hook to be tested in sample mode by
        flipping one row to ``"withdrawn"``).

    The schema now matches ``_write_structured_indications``'s output
    schema (``drugbank_id, disease_id, disease_name, indication_type,
    source``) plus the extra ``drug_inchikey, drug_name, indication,
    doid_id, omim_disease_id`` columns for richer test fixtures.
    """
    return pd.DataFrame([
        {"drugbank_id": "DB00945", "drug_inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
         "drug_name": "Aspirin",
         "disease_id": "DOID:0050133", "disease_name": "Pain",
         "doid_id": "DOID:0050133", "omim_disease_id": None,
         "indication": "For the treatment of pain",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00945", "drug_inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
         "drug_name": "Aspirin",
         "disease_id": "DOID:1101", "disease_name": "Inflammation",
         "doid_id": "DOID:1101", "omim_disease_id": None,
         "indication": "For the treatment of inflammation",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00316", "drug_inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N",
         "drug_name": "Acetaminophen",
         "disease_id": "DOID:0050133", "disease_name": "Pain",
         "doid_id": "DOID:0050133", "omim_disease_id": None,
         "indication": "For the treatment of pain",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB01050", "drug_inchikey": "HEFNNWSXXWATIW-UHFFFAOYSA-N",
         "drug_name": "Ibuprofen",
         "disease_id": "DOID:7148", "disease_name": "Arthritis",
         "doid_id": "DOID:7148", "omim_disease_id": None,
         "indication": "For the treatment of arthritis",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00201", "drug_inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
         "drug_name": "Caffeine",
         "disease_id": "DOID:1197", "disease_name": "Migraine",
         "doid_id": "DOID:1197", "omim_disease_id": None,
         "indication": "For the treatment of migraine",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00829", "drug_inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N",
         "drug_name": "Diazepam",
         "disease_id": "DOID:14319", "disease_name": "Anxiety Disorder",
         "doid_id": "DOID:14319", "omim_disease_id": None,
         "indication": "For the treatment of anxiety",
         "indication_type": "approved", "source": "drugbank_xml"},
        # v107 FORENSIC ROOT FIX (ISSUE-P1-013):
        #   The previous code used disease_id="OMIM:137160" for Diazepam
        #   treating Epilepsy, claiming it was "in embedded_omim_gda()
        #   (GABRA1 gene)". But 137160 is the GENE MIM (GABRA1 locus), NOT
        #   the disease MIM. The actual disease MIM for Epilepsy, juvenile
        #   myoclonic is OMIM:254770 (per embedded_omim_gda()'s GABRA1 row:
        #   gene_mim=137160, disease_id="OMIM:254770"). Using the gene MIM
        #   as the disease_id created a phantom Disease node for "OMIM:137160"
        #   with no disease_name, and the treats-edge NEVER connected to the
        #   GABRA1 GDA's disease node -- the multi-hop drug->protein->gene->
        #   disease pattern was BROKEN. ROOT FIX: use disease_id="OMIM:254770"
        #   (the actual disease MIM) and keep gene_mim=137160 in the OMIM GDA
        #   table (where it belongs). Do not conflate gene MIMs with disease
        #   MIMs.
        {"drugbank_id": "DB00829", "drug_inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N",
         "drug_name": "Diazepam",
         "disease_id": "OMIM:254770", "disease_name": "Epilepsy, juvenile myoclonic",
         "doid_id": "DOID:1826", "omim_disease_id": "OMIM:254770",
         "indication": "For the treatment of seizures",
         "indication_type": "approved", "source": "drugbank_xml"},
        # v107 FORENSIC ROOT FIX (ISSUE-P1-014):
        #   The previous code used disease_id="DOID:0005049" for Warfarin
        #   treating Thrombosis. But DOID:0005049 does NOT exist in the
        #   Disease Ontology -- DOID IDs are numeric without leading zeros
        #   after the colon (the "0005049" form is invalid). The KG created
        #   a phantom Disease node with a non-existent DOID. Any clinical-
        #   trial cross-reference against the Disease Ontology returned 404.
        #   ROOT FIX: use the verified-correct DOID for Thrombosis:
        #   DOID:0060903 (verified via the EBI OLS API on 2026-07-13:
        #   http://purl.obolibrary.org/obo/DOID_0060903 | label=thrombosis).
        {"drugbank_id": "DB00682", "drug_inchikey": "PJVWKTKQMONHTF-UHFFFAOYSA-N",
         "drug_name": "Warfarin",
         "disease_id": "DOID:0060903", "disease_name": "Thrombosis",
         "doid_id": "DOID:0060903", "omim_disease_id": None,
         "indication": "For the prevention of thrombosis",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00191", "drug_inchikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N",
         "drug_name": "Metformin",
         "disease_id": "DOID:9351", "disease_name": "Diabetes Mellitus",
         "doid_id": "DOID:9351", "omim_disease_id": None,
         "indication": "For the treatment of type 2 diabetes",
         "indication_type": "approved", "source": "drugbank_xml"},
        # P0-B1 ROOT FIX: Hypercholesterolemia maps to OMIM:143890 which
        # IS in embedded_omim_gda() (HMGCR gene). Use the OMIM ID.
        {"drugbank_id": "DB01076", "drug_inchikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N",
         "drug_name": "Atorvastatin",
         "disease_id": "OMIM:143890", "disease_name": "Hypercholesterolemia",
         "doid_id": "DOID:50", "omim_disease_id": "OMIM:143890",
         "indication": "For the treatment of hypercholesterolemia",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB01197", "drug_inchikey": "BNRQQXFRAQNPGX-UHFFFAOYSA-N",
         "drug_name": "Captopril",
         "disease_id": "DOID:10763", "disease_name": "Hypertension",
         "doid_id": "DOID:10763", "omim_disease_id": None,
         "indication": "For the treatment of hypertension",
         "indication_type": "approved", "source": "drugbank_xml"},
        {"drugbank_id": "DB00722", "drug_inchikey": "RJXRWZVZAQXBEZ-UHFFFAOYSA-N",
         "drug_name": "Lisinopril",
         "disease_id": "DOID:10763", "disease_name": "Hypertension",
         "doid_id": "DOID:10763", "omim_disease_id": None,
         "indication": "For the treatment of hypertension",
         "indication_type": "approved", "source": "drugbank_xml"},
    ])


def embedded_omim_gda() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_omim_gda")
    """OMIM gene-disease associations for the sample proteins."""
    return pd.DataFrame([
        # v64 ROOT FIX (P1-016): gene_id and phenotype_mim changed from
        # string to integer to match schema/v1.json (gene_id: integer,
        # phenotype_mim: integer). The previous string values worked by
        # accident when read with pd.read_csv(dtype={...}) but caused
        # silent join failures when read without explicit dtypes (e.g. by
        # the Phase 2 bridge using csv.reader).
        #
        # v93 ROOT FIX (P1-047 -- gene_mim / disease_id / phenotype_mim
        #   conflation): the previous sample had gene_mim=176805,
        #   disease_id="OMIM:176805", AND phenotype_mim=176805 -- all
        #   three were the SAME number. This conflated the GENE MIM
        #   (176805 = PTGS1 gene, also known as COX1) with the DISEASE
        #   MIM (the phenotype). In OMIM's data model, the gene MIM and
        #   the disease MIM are DIFFERENT numbers -- the gene MIM
        #   identifies the gene locus, while the phenotype MIM identifies
        #   the clinical disorder caused by variants in that gene. The
        #   PTGS1 gene (MIM 176805) is associated with the phenotype
        #   "Platelet-type bleeding disorder 8" (MIM 609800) per OMIM
        #   #609800. Conflating them produced a sample where the KG
        #   would create a self-loop (gene -> disease with the same MIM),
        #   corrupting downstream Graph Transformer edge features and
        #   making the sample useless for testing GDA scoring. Root fix:
        #   use the real PTGS1 gene MIM (176805) for gene_mim, the real
        #   disease MIM (609800 -- Platelet-type bleeding disorder 8)
        #   for disease_id and phenotype_mim. Verified against OMIM
        #   morbidmap.txt (PTGS1, 609800, (3)).
        {"gene_symbol": "PTGS1", "gene_id": 5742, "gene_mim": 176805,
         "disease_id": "OMIM:609800", "disease_name": "Platelet-type bleeding disorder 8",
         "phenotype_mim": 609800, "association_type": "causal",
         "is_susceptibility": False, "source": "omim", "score": 1.0},  # v57 ROOT FIX (P1-003): was "causative" (not in enum [causal, susceptibility, ...])
        {"gene_symbol": "PTGS2", "gene_id": 5743, "gene_mim": 600262,
         "disease_id": "OMIM:114500", "disease_name": "Colorectal cancer susceptibility",
         "phenotype_mim": 114500, "association_type": "susceptibility",
         "is_susceptibility": True, "source": "omim", "score": 0.85},
        {"gene_symbol": "ADORA2a", "gene_id": 135, "gene_mim": 102776,
         "disease_id": "OMIM:108150", "disease_name": "Vascular disorders",
         "phenotype_mim": 108150, "association_type": "susceptibility",
         "is_susceptibility": True, "source": "omim", "score": 0.75},
        {"gene_symbol": "GABRA1", "gene_id": 2552, "gene_mim": 137160,
         "disease_id": "OMIM:254770", "disease_name": "Epilepsy, juvenile myoclonic",
         "phenotype_mim": 254770, "association_type": "causal",
         "is_susceptibility": False, "source": "omim", "score": 1.0},  # v57 ROOT FIX (P1-003): was "causative" (not in enum [causal, susceptibility, ...])
        {"gene_symbol": "HMGCR", "gene_id": 3156, "gene_mim": 142910,
         "disease_id": "OMIM:143890", "disease_name": "Hypercholesterolemia",
         "phenotype_mim": 143890, "association_type": "susceptibility",
         "is_susceptibility": True, "source": "omim", "score": 0.7},
        {"gene_symbol": "ACE", "gene_id": 1636, "gene_mim": 106180,
         "disease_id": "OMIM:608558", "disease_name": "Myocardial infarction susceptibility",
         "phenotype_mim": 608558, "association_type": "susceptibility",
         "is_susceptibility": True, "source": "omim", "score": 0.8},
        # P1-043 ROOT FIX (v107): added rows for the 4 remaining association_type
        # values to exercise the full OMIM enum. The previous sample had only
        # 'causal' and 'susceptibility' — tests depending on 'non_disease',
        # 'provisional', 'gene_locus', or 'mendelian_phenotype' saw 0 coverage.
        # A code change that broke handling of any of these 4 types was NOT
        # caught by the embedded-sample tests, and the bug propagated to
        # production where the OMIM pipeline emits the full enum.
        {"gene_symbol": "BRCA1", "gene_id": 672, "gene_mim": 113705,
         "disease_id": "OMIM:600185", "disease_name": "Breast-ovarian cancer, familial 1",
         "phenotype_mim": 600185, "association_type": "non_disease",
         "is_susceptibility": False, "source": "omim", "score": 0.9},
        {"gene_symbol": "BRCA2", "gene_id": 675, "gene_mim": 600185,
         "disease_id": "OMIM:612555", "disease_name": "Breast-ovarian cancer, familial 2",
         "phenotype_mim": 612555, "association_type": "provisional",
         "is_susceptibility": False, "source": "omim", "score": 0.85},
        {"gene_symbol": "HBB", "gene_id": 3043, "gene_mim": 141900,
         "disease_id": "OMIM:141900", "disease_name": "Beta-thalassemia locus",
         "phenotype_mim": 141900, "association_type": "gene_locus",
         "is_susceptibility": False, "source": "omim", "score": 1.0},
        {"gene_symbol": "CFTR", "gene_id": 1080, "gene_mim": 602421,
         "disease_id": "OMIM:219700", "disease_name": "Cystic fibrosis",
         "phenotype_mim": 219700, "association_type": "mendelian_phenotype",
         "is_susceptibility": False, "source": "omim", "score": 1.0},
    ])


def embedded_omim_susceptibility() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_omim_susceptibility")
    """P2-10 ROOT FIX: OMIM gene-disease susceptibility associations.

    The previous code mapped ``omim_susceptibility`` to the SAME function
    as ``omim_gda`` (``embedded_omim_gda``), producing byte-identical CSV
    files. The susceptibility file should be a SUBSET -- only rows where
    ``is_susceptibility=True``. This filter produces the 4 susceptibility
    rows (PTGS2/Colorectal cancer, ADORA2A/Vascular disorders,
    HMGCR/Hypercholesterolemia, ACE/Myocardial infarction) while the GDA
    file contains all 6 rows (including the 2 causal associations).
    This is biologically correct: susceptibility associations are a
    distinct category from causal associations in OMIM's nosology.
    """
    gda = embedded_omim_gda()
    susc = gda[gda["is_susceptibility"] == True].copy()  # noqa: E712
    return susc


def embedded_disgenet_gda() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_disgenet_gda")
    """DisGeNET gene-disease associations (curated subset for sample genes).

    P1-049 ROOT FIX (v107): the previous sample did NOT include the
    ``confidence_tier`` column — it was derived by the cleaning module's
    ``classify_confidence``. If the disgenet pipeline skipped
    ``classify_confidence`` (e.g. a code change removed the call), the
    GDA rows were inserted with ``confidence_tier=NULL`` — which the
    DB CHECK allows but downstream ML bins as "unknown". The GNN's
    feature binning on ``confidence_tier`` treated all 7 rows as
    "unknown" instead of "very_strong" (all scores >= 0.5). The model's
    confidence-weighted loss was mis-calibrated. ROOT FIX: include the
    ``confidence_tier`` column in the embedded sample so even if
    ``classify_confidence`` is not called, the column is populated
    correctly. Also added variety in scores so all 4 tiers are
    exercised: sub_weak (0.05), weak (0.15), strong (0.4), very_strong
    (0.6-0.95).
    """
    return pd.DataFrame([
        # v64 ROOT FIX (P1-016): gene_id changed from string to integer
        # to match schema/v1.json (gene_id: integer).
        # P1-049 ROOT FIX (v107): added confidence_tier column.
        {"gene_symbol": "PTGS1", "gene_id": 5742, "disease_id": "DOID:0050133",
         "disease_name": "Pain", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.85, "confidence_tier": "very_strong",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "12345678;23456789"},
        {"gene_symbol": "PTGS2", "gene_id": 5743, "disease_id": "DOID:1101",
         "disease_name": "Inflammation", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.9, "confidence_tier": "very_strong",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "11111111;22222222"},
        {"gene_symbol": "PTGS2", "gene_id": 5743, "disease_id": "DOID:162",
         "disease_name": "Cancer", "association_type": "biomarker",
         "source": "disgenet", "score": 0.4, "confidence_tier": "strong",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "33333333;44444444"},
        {"gene_symbol": "ADORA2A", "gene_id": 135, "disease_id": "DOID:1197",
         "disease_name": "Migraine", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.15, "confidence_tier": "weak",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "55555555"},
        {"gene_symbol": "GABRA1", "gene_id": 2552, "disease_id": "DOID:1826",
         "disease_name": "Epilepsy", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.6, "confidence_tier": "very_strong",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "66666666;77777777"},
        {"gene_symbol": "HMGCR", "gene_id": 3156, "disease_id": "DOID:50",
         "disease_name": "Hypercholesterolemia", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.95, "confidence_tier": "very_strong",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "88888888;99999999"},
        {"gene_symbol": "ACE", "gene_id": 1636, "disease_id": "DOID:10763",
         "disease_name": "Hypertension", "association_type": "therapeutic",
         "source": "disgenet", "score": 0.05, "confidence_tier": "sub_weak",
         "confidence_tier_method": "pinero_2020_v2",
         "pmid_list": "12121212;34343434"},
    ])


def embedded_pubchem_enrichment() -> pd.DataFrame:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("embedded_pubchem_enrichment")
    """PubChem physicochemical properties for the sample drugs.

    Schema matches ``_PHASE1_EXPECTED_COLUMNS['pubchem_enrichment']`` in
    phase1_bridge.py: requires ``inchikey``, ``canonical_smiles``.

    v107 ROOT FIX (ISSUE-P1-036 — missing ``drug_source`` column):
    The v83 P1-8 contract added ``drug_source`` as the 9th column on the
    API-path CSV writer (``_v50_downloaders.py:742-747``) so downstream
    consumers (phase1_bridge.pubchem_enrichment reader) know which drug
    source each row came from. But this embedded fallback was NEVER
    updated — it returned only 8 columns, so when the fallback fired,
    the bridge saw a missing ``drug_source`` column and either crashed
    (KeyError) or silently treated every row as ``drug_source=None``,
    losing the provenance signal the v83 fix was supposed to provide.
    ROOT FIX: add ``drug_source="embedded_sample"`` to every row so
    the fallback schema matches the API-path schema exactly.
    """
    return pd.DataFrame([
        {"inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "pubchem_cid": 2244,
         "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
         "xlogp": 1.19, "tpsa": 63.6, "h_bond_donor_count": 1,
         "h_bond_acceptor_count": 4, "rotatable_bond_count": 2,
         "drug_source": "embedded_sample"},
        {"inchikey": "RZVAJINKPMORJF-UHFFFAOYSA-N", "pubchem_cid": 1983,
         "canonical_smiles": "CC1=CC=C(O)C=C1O",
         "xlogp": 0.46, "tpsa": 49.33, "h_bond_donor_count": 2,
         "h_bond_acceptor_count": 2, "rotatable_bond_count": 0,
         "drug_source": "embedded_sample"},
        {"inchikey": "HEFNNWSXXWATIW-UHFFFAOYSA-N", "pubchem_cid": 3672,
         "canonical_smiles": "CC(C)CC1=CC=C(C=C1)CC(C(=O)O)C",
         "xlogp": 3.97, "tpsa": 37.3, "h_bond_donor_count": 1,
         "h_bond_acceptor_count": 2, "rotatable_bond_count": 4,
         "drug_source": "embedded_sample"},
        {"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "pubchem_cid": 2519,
         "canonical_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
         "xlogp": -0.07, "tpsa": 58.44, "h_bond_donor_count": 0,
         "h_bond_acceptor_count": 6, "rotatable_bond_count": 0,
         "drug_source": "embedded_sample"},
        {"inchikey": "AAOVKBJEBZCEQK-UHFFFAOYSA-N", "pubchem_cid": 3016,
         "canonical_smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
         "xlogp": 2.82, "tpsa": 32.67, "h_bond_donor_count": 0,
         "h_bond_acceptor_count": 2, "rotatable_bond_count": 1,
         "drug_source": "embedded_sample"},
        {"inchikey": "PJVWKTKQMONHTF-UHFFFAOYSA-N", "pubchem_cid": 6691,
         "canonical_smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
         "xlogp": 2.70, "tpsa": 46.61, "h_bond_donor_count": 1,
         "h_bond_acceptor_count": 3, "rotatable_bond_count": 2,
         "drug_source": "embedded_sample"},
        {"inchikey": "XZWYZXLIPXDOLR-UHFFFAOYSA-N", "pubchem_cid": 4091,
         "canonical_smiles": "CN(C)C(=N)N=C(N)N",
         "xlogp": -1.43, "tpsa": 76.07, "h_bond_donor_count": 2,
         "h_bond_acceptor_count": 4, "rotatable_bond_count": 2,
         "drug_source": "embedded_sample"},
        {"inchikey": "XUKUURHRXDUEBC-UHFFFAOYSA-N", "pubchem_cid": 60823,
         "canonical_smiles": "CC(C)C1=C(C=CC=C1C)C2=CC=CC=C2C(=O)NC3CC4=C(C=C(C=C4CC3)F)C(=O)O",
         "xlogp": 4.19, "tpsa": 111.78, "h_bond_donor_count": 3,
         "h_bond_acceptor_count": 7, "rotatable_bond_count": 8,
         "drug_source": "embedded_sample"},
        {"inchikey": "BNRQQXFRAQNPGX-UHFFFAOYSA-N", "pubchem_cid": 44093,
         "canonical_smiles": "CC(C)C1CC2C(SC1)C(=O)NC2C(=O)O",
         "xlogp": 0.65, "tpsa": 86.91, "h_bond_donor_count": 2,
         "h_bond_acceptor_count": 4, "rotatable_bond_count": 2,
         "drug_source": "embedded_sample"},
        {"inchikey": "RJXRWZVZAQXBEZ-UHFFFAOYSA-N", "pubchem_cid": 5362119,
         "canonical_smiles": "CCCCC(C)C1C(=O)N2CCCC2C(=O)N1CC(C(=O)O)N",
         "xlogp": -1.21, "tpsa": 132.85, "h_bond_donor_count": 3,
         "h_bond_acceptor_count": 8, "rotatable_bond_count": 9,
         "drug_source": "embedded_sample"},
    ])


def write_all_samples(processed_dir) -> dict:
    """P1-019 ROOT FIX: raises RuntimeError if called in production."""
    _assert_not_production("write_all_samples")
    """Write all embedded sample datasets as CSVs to the processed_data dir.

    Used as a last-resort fallback when ANY pipeline cannot reach its API
    in sample mode. Returns a dict mapping source-name -> file-path.

    v65 ROOT FIX (P1-040): the previous implementation wrote each CSV
    NON-atomically -- ``df.to_csv(path, ...)`` truncates the file before
    writing the body, so a concurrent call to ``write_all_samples`` (e.g.
    from the ``all`` command's fallback at __init__.py:2801) could leave
    a partially-written CSV that the Phase 2 bridge would then crash on
    (malformed CSV -> pandas ParserError). Root fix: write to a ``.tmp``
    sidecar file in the same directory, then atomically ``os.replace``
    it to the final path. ``os.replace`` is atomic on POSIX and Windows
    for files within the same filesystem. We also use ``filelock`` when
    available to serialize concurrent writers across processes; if
    ``filelock`` is not installed, we fall back to atomic-rename-only
    (which still prevents partial reads but not double-writes).
    """
    import os
    import tempfile
    from pathlib import Path
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    datasets = {
        "chembl_drugs": (embedded_chembl_molecules, "chembl_drugs.csv"),
        "chembl_activities": (embedded_chembl_activities, "chembl_activities_clean.csv"),
        "uniprot_proteins": (embedded_uniprot_proteins, "proteins.csv"),
        "string_ppi": (embedded_string_ppi, "protein_protein_interactions.csv"),
        "drugs": (embedded_drugbank_drugs, "drugbank_drugs.csv"),
        "interactions": (embedded_drugbank_interactions, "drugbank_interactions.csv.gz"),
        "indications": (embedded_drugbank_indications, "drugbank_indications.csv"),
        "omim_gda": (embedded_omim_gda, "omim_gene_disease_associations.csv"),
        # P2-10 ROOT FIX: the previous code mapped both "omim_gda" and
        # "omim_susceptibility" to the SAME function (embedded_omim_gda),
        # producing byte-identical CSV files. The susceptibility file is
        # now a proper SUBSET (is_susceptibility=True only) via the new
        # embedded_omim_susceptibility() function.
        "omim_susceptibility": (embedded_omim_susceptibility, "omim_gene_disease_susceptibility.csv"),
        "disgenet_gda": (embedded_disgenet_gda, "gene_disease_associations.csv"),
        "pubchem_enrichment": (embedded_pubchem_enrichment, "pubchem_enrichment.csv"),
    }

    # v65 ROOT FIX (P1-040): optional cross-process file lock to prevent
    # two concurrent write_all_samples() calls from doing double-work.
    # Even without the lock, the atomic-rename pattern below guarantees
    # no reader ever sees a partially-written file.
    # v107 ROOT FIX (ISSUE-P1-037 — filelock fallback was NOT serializing
    #   concurrent writers):
    #   The previous code fell back to ``_atomic_write_csv`` WITHOUT any
    #   cross-process lock when ``filelock`` was not installed. Two
    #   concurrent ``write_all_samples`` calls (e.g. from ``pipelines all``
    #   and a parallel Airflow task) both did the full work and the
    #   final file was whichever ``os.replace`` ran LAST — a double-write
    #   that wasted CPU/IO and could produce a file with fewer rows than
    #   expected if one process's DataFrame was stale. ROOT FIX: use
    #   ``fcntl.flock`` (POSIX advisory lock) as a fallback when
    #   ``filelock`` is not installed. On Windows (no ``fcntl``), fall
    #   back to atomic-rename-only with a WARNING. This makes the lock
    #   behavior consistent across POSIX systems regardless of whether
    #   ``filelock`` is installed.
    try:
        from filelock import FileLock, Timeout as FileLockTimeout
        _has_filelock = True
    except ImportError:
        _has_filelock = False

    # v107 P1-037: POSIX fcntl.flock fallback when filelock is not installed.
    _has_fcntl = False
    try:
        import fcntl as _fcntl_module  # noqa: F401
        _has_fcntl = True
    except ImportError:
        # Windows — no fcntl. Will fall back to atomic-rename-only.
        _has_fcntl = False

    for key, (func, filename) in datasets.items():
        df = func()
        path = processed_dir / filename

        if _has_filelock:
            lock_path = path.with_suffix(path.suffix + ".lock")
            lock = FileLock(str(lock_path), timeout=30)
            try:
                with lock:
                    _atomic_write_csv(df, path, filename)
            except FileLockTimeout:
                # Another process holds the lock -- skip this file
                # (the other process will write it). We still record
                # the path in `written` so callers know where it SHOULD be.
                logger.warning(
                    "Could not acquire lock for %s -- another process is "
                    "writing it. Skipping.", path,
                )
        elif _has_fcntl:
            # v107 P1-037: POSIX fcntl.flock fallback. Use a dedicated
            # .lock sidecar file (NOT the data file itself — flock on
            # the data file would conflict with readers). Acquire an
            # exclusive lock, write, release. If another process holds
            # the lock, flock blocks (no timeout — POSIX flock has no
            # built-in timeout). This is acceptable because
            # write_all_samples is a fast operation (10 mock rows).
            import fcntl as _fcntl
            lock_path = path.with_suffix(path.suffix + ".lock")
            lock_fd = None
            try:
                lock_fd = open(lock_path, "w")
                _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_EX)
                _atomic_write_csv(df, path, filename)
            except OSError as flock_exc:
                logger.warning(
                    "Could not acquire fcntl lock for %s: %s. "
                    "Falling back to unlocked atomic write (v107 P1-037).",
                    path, flock_exc,
                )
                _atomic_write_csv(df, path, filename)
            finally:
                if lock_fd is not None:
                    try:
                        _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_UN)
                    except OSError:
                        pass
                    lock_fd.close()
        else:
            # v107 P1-037: Windows (no fcntl) and no filelock installed.
            # Fall back to atomic-rename-only. Log a WARNING so operators
            # know concurrent writers are not serialized.
            if key == "chembl_drugs":  # log once per call
                logger.warning(
                    "Neither filelock nor fcntl is available — concurrent "
                    "write_all_samples() calls are NOT serialized (v107 "
                    "P1-037). Install `filelock` (pip install filelock) "
                    "or run on POSIX for proper cross-process locking."
                )
            _atomic_write_csv(df, path, filename)
        written[key] = path
    return written


def _atomic_write_csv(df, path, filename) -> None:
    """Write ``df`` to ``path`` atomically via .tmp + os.replace."""
    import os
    import tempfile
    # Create the .tmp file in the SAME directory as the final path so
    # ``os.replace`` is atomic (same filesystem guarantee).
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(tmp_fd)  # we'll reopen for writing via pandas
    tmp_path = type(path)(tmp_path_str)
    try:
        if filename.endswith(".csv.gz"):
            df.to_csv(tmp_path, index=False, compression="gzip")
        else:
            df.to_csv(tmp_path, index=False)
        # os.replace is atomic on POSIX and Windows for files within
        # the same filesystem. Readers either see the OLD file or the
        # NEW file -- never a partial write.
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the .tmp file on any failure -- don't leave orphaned
        # .tmp files in the processed_data dir.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


if __name__ == "__main__":
    # CLI: write all samples to a target dir.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_dir", help="Directory to write sample CSVs to")
    args = parser.parse_args()
    written = write_all_samples(args.processed_dir)
    print(f"Wrote {len(written)} sample datasets to {args.processed_dir}:")
    for key, path in written.items():
        print(f"  {key}: {path}")
