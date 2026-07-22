# v82 Forensic Root-Fix Tests
"""
v82 FORENSIC ROOT-FIX REGRESSION TESTS -- 5 compound/cross-file chains.

Each test reproduces the EXACT failure scenario described in the issue
and verifies the fix holds. These are NOT smoke tests -- they exercise
the real production code paths (entity resolution, normalizer, dedup,
confidence) with realistic data shapes.

Chain-1: InChIKey protonation-suffix stripping (drug_resolver + normalizer)
Chain-2: STRING aliases -> _string_to_uniprot population (run.py + protein_resolver)
Chain-3: O(N*M) promotion loop -> O(1) alias-uniprot index (protein_resolver)
Chain-4: p-scale censor preservation through pipeline (normalizer + chembl_pipeline + dedup)
Chain-5: negative GDA scores + classify_confidence (missing_values + confidence)
"""
