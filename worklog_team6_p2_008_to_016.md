---
Task ID: team6-p2-008-to-016
Agent: Team Member 6 (Claude Code agent)
Task: Fix 9 Phase 2 data-loader issues (P2-008 through P2-016) — STRING, STITCH, DrugBank, ClinicalTrials, OMIM, OpenTargets, DRKG, GEO, SIDER loaders.

Work Log:
- Read project docx (Team_Cosmic_Build_Process_Updated.docx) to understand the 4-phase architecture: KG (Neo4j) -> Graph Transformer (PyTorch+PyG) -> RL Ranker -> Clinical Decision Layer.
- Cloned repo at /home/z/my-project/repo and created branch fix/team6-p2-008-to-p2-016-forensic-root-fixes.
- Read each loader file line-by-line (real code, not comments/tests) per user instruction: string_loader.py, stitch_loader.py, drugbank_parser.py, clinicaltrials_loader.py, omim_loader.py, opentargets_loader.py, drkg_loader.py, geo_loader.py, sider_loader.py, and config.py.
- P2-008: Added _looks_like_canonical_string_id + _warn_if_gene_symbol_mode helpers in string_loader.py. _canonicalize_pair_order now warns when gene-symbol mode is detected (cross-species PPI collapse risk). Updated emit_both_directions docstring.
- P2-009: Expanded _stitch_stereo_label docstring in stitch_loader.py with full citation to Kuhn et al. 2008 (doi:10.1093/nar/gkm858) and per-code explanation of all 6+1 stereo codes (sm, s, f, m, 0, 1, "").
- P2-010: Added _parse_food_interactions, _parse_herb_interactions, _classify_food_herb_severity, drugbank_to_food_herb_edges in drugbank_parser.py. Added food_interactions and herb_interactions fields to DrugRecord. Wired parsers into _parse_drug_fields. Emits (Compound)-[:causes_adverse_event]->(Food|Herb) edges.
- P2-011: Added _detect_ctgov_schema_version, _parse_ctgov_v1_study, _parse_ctgov_v2_study, parse_ctgov_study, fetch_ctgov_studies in clinicaltrials_loader.py. Handles both legacy v1 (StudyFieldsSection) and current v2 (protocolSection) schemas with cursor-based pagination.
- P2-012: Added leading-digit validation [1-6] to _safe_gene_id_from_mim in omim_loader.py. MIMs with leading 7/8/9 fall back to SYM:<symbol>. Leading-0 case is conceptually unreachable (int(float("099999"))=99999 fails 6-digit range first) — documented this in code comment + test.
- P2-013: Added fetch_opentargets_associations + _build_opentargets_associations_query in opentargets_loader.py. Follows GraphQL cursor until all rows fetched or max_pages reached. Truncation WARNING emitted when max_pages is hit before completion.
- P2-014: Added "A" -> "Compound-affects-Gene" to DRKG_RELATION_ABBREV_TO_NAME and ("A", "Compound", "Gene")/("A", "Compound", "Disease") to DRKG_VALID_TRIPLE_SCHEMAS in config.py. Added parse_drkg_relation_head_tail (colon-safe split) and canonical_drkg_relation_name (case-insensitive lookup) helpers. Updated split_drkg_relation docstring to document both DRKG formats.
- P2-015: Added optional DRUGOS_GEO_CA_BUNDLE env var for CA pinning in geo_loader._create_ssl_context. Added _verify_tls_strict regression-test helper. The loader has always used HTTPS with verify_mode=CERT_REQUIRED (audit was a misattribution) but the fix adds defence-in-depth via CA pinning + regression hook.
- P2-016: Added PT-by-name dedup to sider_to_node_records in sider_loader.py. Deduplicates AdverseEvent nodes by lowercased side_effect_name (column 4 in meddra.tsv) with PT-preferential ordering. Gated by the `dedup` parameter. Prevents duplicate AE nodes for the same condition (e.g. Nausea PT vs LLT).
- Wrote 9 regression test files (170 tests total) in phase2/tests/team6_p2_008_to_016/. All tests pass: 170 passed in 1.97s.
- Real-code smoke test: all 9 loaders import and execute real functions correctly. End-to-end DrugBank XML parse with food/herb interactions verified: atorvastatin -> grapefruit juice edge correctly classified as 'severe' (rhabdomyolysis risk).

Stage Summary:
- 9 issues fixed at root level (no surface-level patches).
- 170 regression tests, all passing.
- Real code execution verified end-to-end.
- Branch: fix/team6-p2-008-to-p2-016-forensic-root-fixes
- Files modified: string_loader.py, stitch_loader.py, drugbank_parser.py, clinicaltrials_loader.py, omim_loader.py, opentargets_loader.py, drkg_loader.py (via config.py), geo_loader.py, sider_loader.py, config.py
- Files added: 9 test files in phase2/tests/team6_p2_008_to_016/
- No breaking changes to public APIs (all new code is additive or backward-compatible).
- All commit messages use the required format: fix(P2-XXX): <description>.
