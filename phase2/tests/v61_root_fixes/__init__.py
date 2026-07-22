"""v61 ROOT FIX verification tests -- 3 silent break points in the bridge.

Each test verifies ONE root-cause fix from the v61 audit. Tests are
designed to FAIL if a regression re-introduces the bug. They use NO
network access, NO real databases -- pure Python assertions on the actual
code.

Issues verified:
  #1  _phase1_db_available() swallows ALL exceptions with broad except
      Exception and (in v58/v60) re-raises in production, crashing the
      bridge for the COMMON configuration error of "SQLite file exists
      but no schema migrated".
  #2  read_phase1_outputs() adds a SECOND layer of silent fallback that
      also crashes for the same configuration errors.
  #3  run_unified.py Phase 1 auto-invocation fails with NO fallback when
      the sample-mode API calls fail (no network, missing API keys).
  #4  Phase1StagedData.total_nodes MUST include pathway_nodes (regression
      guard for the v57 fix that was unverifiable because of bug #1).
  #5  Phase 1 ↔ Phase 2 connection produces ALL 5 node types mandated
      by the DOCX Phase 2 spec (Compound, Protein, Gene, Disease,
      ClinicalOutcome, Pathway).
  #6  nodes_staged == nodes_loaded (no under-reporting).
"""
