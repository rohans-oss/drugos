# AUDIT TRAIL — `entity_resolution/drug_resolver.py`

This file is the **canonical audit trail** for the drug-resolution logic
in `phase1/entity_resolution/drug_resolver.py`. It consolidates the
inline "v[0-9]+ FORENSIC ROOT FIX (BUG #X)" / "ROOT FIX" comment blocks
that were previously scattered across the 6,600+ line file into a single
indexed reference. The inline comments remain in the source for code-level
context, but this file is the authoritative index for understanding
*why* each fix was made, *what* the previous behavior was, and *what* the
root fix changed.

> **P1-038 ROOT FIX (audit-trail extraction):** the previous
> `drug_resolver.py` had ~57 inline "FORENSIC ROOT FIX (BUG #X)" comment
> blocks (often 20–50 lines each), making the file nearly unmaintainable.
> A future maintainer could not find the actual resolution logic without
> scrolling past hundreds of lines of audit-trail comments. The audit
> comments are valuable for traceability but should be indexed in a
> separate file. This file is that index. Keep only the essential
> "why this code exists" comments inline; the full forensic context
> lives here.

---

## How to use this file

1. **When investigating a bug in `drug_resolver.py`:** find the relevant
   BUG # below, read the root-fix description, then jump to the cited
   line range in the source for the code-level fix.
2. **When adding a new fix:** append a new entry to this file (do NOT
   add a 50-line inline comment block to the source). In the source, add
   a one-line comment like `# vXXX ROOT FIX (BUG #Y): see AUDIT_TRAIL.md`
   pointing to the entry here.
3. **When refactoring:** preserve the BUG # references in the source so
   the index stays navigable.

---

## Index of Forensic Root Fixes (ordered by BUG #)

### BUG #5 (P0) — `_match_by_name` ignored `confidence_score` from callers

- **Source line:** ~4490 in `drug_resolver.py`
- **Symptom:** `_match_by_name` accepted a `confidence_score` parameter
  but never used it — it computed its own score internally and overrode
  the caller's value. Callers that passed a high confidence (e.g. 0.95
  for an exact InChIKey match) had their score silently discarded.
- **Root cause:** historical — the parameter was added for forward-compat
  but the internal computation was never wired to respect it.
- **Root fix (v89):** `_match_by_name` now uses the caller-provided
  `confidence_score` when it is non-None, falling back to internal
  computation only when the caller passes None. This makes the
  confidence score a true input rather than an output.

### BUG #9 (P1) — SMILES not canonicalized via RDKit before comparison

- **Source lines:** ~4659, ~4721, ~5420, ~5737, ~5949 in `drug_resolver.py`
- **Symptom:** when comparing two SMILES strings for equality, the
  previous code did a string-strip comparison. Two SMILES representing
  the same molecule but with different atom ordering (e.g.
  `C(C)(C)O` vs `C(C)(O)C`) compared unequal — causing the resolver to
  miss matches and create duplicate drug entries.
- **Root cause:** RDKit was not always available (ARM64 wheels missing
  in some CI environments), so the code fell back to strip-only
  comparison without warning.
- **Root fix (v89):** centralize SMILES canonicalization in a single
  helper that uses RDKit when available and logs a WARNING when falling
  back to strip-only. Every SMILES comparison site calls the helper.

### BUG #10 (P1) — Deterministic tie-break was non-deterministic

- **Source line:** ~4583 in `drug_resolver.py`
- **Symptom:** when two candidate matches had the same fuzzy-match
  score, the previous code picked `matches[0]` — but `matches` was a
  set, so the order was non-deterministic across Python runs. The same
  input could resolve to different drug IDs on different runs.
- **Root cause:** set iteration order is implementation-defined; the
  previous code did not impose a deterministic order.
- **Root fix (v89):** sort tied candidates by (drug_id, name) before
  picking the first. Also refuse to match if the tie is a NEAR-TIE
  (within 0.05 score) — surface as a manual-resolution case instead of
  silently picking one.

### BUG #14 (P1) — InChIKey collision (different molecules, same InChIKey)

- **Source line:** ~5668 in `drug_resolver.py`
- **Symptom:** two genuinely different molecules had the same InChIKey
  (rare but possible — InChIKey is a hash of the connectivity layer,
  and collisions exist). The previous code merged them into one drug
  entry, corrupting both.
- **Root cause:** InChIKey was treated as a unique key without a
  secondary check (e.g. molecular formula, SMILES).
- **Root fix (v89):** when an InChIKey collision is detected, compare
  molecular formulas; if they differ, refuse to merge and log a
  WARNING for manual resolution.

### BUG #22 (P1) — Method NAME inconsistency

- **Source line:** ~1976 in `drug_resolver.py`
- **Symptom:** the `method` field on resolved entries used different
  names for the same logical method (e.g. "inchikey_exact" vs
  "inchikey_match" vs "inchikey") depending on which code path set it.
  Downstream consumers filtering on `method` missed entries.
- **Root cause:** method names were string literals duplicated across
  code paths with no central registry.
- **Root fix (v89):** centralize all method names in a `_RESOLUTION_METHODS`
  frozenset; every code path uses the constant. Add a CI check that
  asserts the `method` field on every resolved entry is in the set.

### BUG #34 — `_create_canonical_entry` does not [redacted for brevity]

- **Source line:** ~5174 in `drug_resolver.py`
- **Symptom:** (see source for full context — the inline comment at
  line 5174 has the complete description)
- **Root fix (v89):** (see source)

### BUG #41 — SYNTH key fallback + connectivity index population

- **Source lines:** ~5383, ~5917 in `drug_resolver.py`
- **Symptom:** the SYNTH InChIKey fallback path (used when a
  canonical InChIKey is unavailable) populated the connectivity index
  even when the SYNTH key was itself missing — creating index entries
  pointing to None.
- **Root cause:** the gate on SYNTH key presence was missing.
- **Root fix (v89):** gate the connectivity index population on
  `synth_key is not None`.

---

## Index of Forensic Root Fixes (ordered by audit ID)

### P0-D1 — Silent InChIKey index-key mismatch

- **Source line:** ~1339 in `drug_resolver.py`
- **Symptom:** the InChIKey index used the full InChIKey (14-char
  connectivity + 1-char version + 1-char proton layer) as the key, but
  the lookup code passed only the 14-char connectivity prefix — so
  every lookup missed.
- **Root fix (v80):** normalize the index key to the 14-char
  connectivity prefix at both index-build and lookup time.

### P1-4 — SYNTH key fallback produces [redacted for brevity]

- **Source lines:** ~5332, ~5917 in `drug_resolver.py`
- **Root fix (v82 → v89):** see source for full context.

### P1-10 — `CANONICAL_SYNTHETIC_INCHIKEY_REGEX` placement

- **Source lines:** ~292, ~4393 in `drug_resolver.py`
- **Symptom:** the regex was defined inside a method, so it was
  recompiled on every call (10K+ calls per run). Profile showed 30% of
  resolver wall-clock time was regex compilation.
- **Root fix (v82):** move the regex to module level so it is compiled
  once at import.

### P1-14 — SYNTH key fallback produces [redacted for brevity]

- **Source line:** ~5332 in `drug_resolver.py`
- **Root fix (v82):** see source.

### P1C-009 — Method/confidence self-consistency

- **Source lines:** ~4363, ~4391 in `drug_resolver.py`
- **Symptom:** SYNTH-keyed entries got a `method` label of
  "inchikey_exact" (same as canonical-keyed entries) but a lower
  `confidence_score`. Downstream consumers filtering on
  `method == "inchikey_exact"` got the lower-confidence entries mixed
  in with the high-confidence ones.
- **Root fix (v65):** SYNTH keys get their OWN method label
  ("inchikey_synth") so downstream filters can distinguish.

### P1C-019 — Empty-name collision

- **Source line:** ~5214 in `drug_resolver.py`
- **Symptom:** two drugs with empty `name` fields were treated as
  name-matches and merged — even though empty names carry no
  identifying information.
- **Root fix (v66):** skip name-matching when either name is empty.

---

## Index of Other Root Fixes (non-BUG-ID, non-audit-ID)

### v16 SW-8 — `_resolve_inchikey_type` returned "canonical" for SYNTH keys

- **Source line:** ~1448 in `drug_resolver.py`
- **Root fix (v16):** return "synthetic" for SYNTH keys.

### v16 SW-9 / SW-10 — Salt-form and metal-cation coverage

- **Source lines:** ~356, ~387 in `drug_resolver.py`
- **Root fix (v16):** added 9 common pharmaceutical salt forms and 8
  additional metal cations to the salt-stripping logic.

### v29 C-1 / C-2 / C-3 / C-6 — Confidence-score and InChIKey-class fixes

- **Source lines:** ~451, ~4355, ~5583 in `drug_resolver.py`
- **Root fix (v29):** audit C-1/C-2 fixed confidence-score inversion;
  C-3 added SYNTH-key classification; C-6 prevented certain edge-case
  duplicate merges.

### v43 P2 — Fuzzy match silently picks first of equal-score ties

- **Source line:** ~4578 in `drug_resolver.py`
- **Root fix (v43):** see BUG #10 above for the v89 deepening.

### v65 P1C-009 — see above

### v66 P1C-019 — see above

### v74 T-022 — Silent RDKit degradation on ARM64

- **Source line:** ~312 in `drug_resolver.py`
- **Symptom:** on ARM64 (e.g. Apple Silicon), RDKit wheels were missing
  in some CI environments. The previous code silently fell back to
  strip-only SMILES comparison without warning — causing missed matches.
- **Root fix (v74):** log a WARNING when RDKit is unavailable; surface
  the degradation in the pipeline run's metadata.

### v80 P0-D1 — see above

### v82 P1-4 / P1-10 / P1-14 — see above

### v89 BUG #5 / #9 / #10 / #14 / #22 / #34 / #41 — see above

---

## Maintenance protocol

1. **When a new fix is added:** append a new entry to the appropriate
   index above. Use the next available BUG # (or the audit ID from the
   issue tracker). In the source, add a one-line comment
   `# vXXX ROOT FIX (BUG #Y): see AUDIT_TRAIL.md` and include any
   essential "why this code exists" context (1–3 lines max).
2. **When refactoring a fix:** preserve the BUG # reference. Update the
   entry here if the fix is materially changed.
3. **When removing dead code from a fix:** remove the entry here too —
   but only if the fix is genuinely dead (no live code path references
   it). If unsure, leave the entry and mark it "[DEAD CODE — kept for
   history]".

---

## Cross-references

- **Issue tracker:** see the `P1-038` issue for the original audit
  finding that motivated this file.
- **Source file:** `phase1/entity_resolution/drug_resolver.py`
- **Related modules:**
  - `phase1/cleaning/normalizer.py` — InChIKey/SMILES normalization
  - `phase1/database/loaders.py` — Drug upsert with resolution
  - `phase2/graph_builder.py` — Consumes resolved drug IDs
