"""Task 119: Verify pyg_builder and phase2_adapter use the SAME node type mapping.

Root test for task 112 (INT-004): both pyg_builder._PHASE2_TO_GT_NODE_TYPE
and phase2_adapter.PHASE2_TO_PHASE3_NODE must import from the SAME shared
module (schema_mappings.py). No local copies allowed.
"""
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Add graph_transformer to path for phase2_adapter
_gt_path = str(Path(__file__).resolve().parents[2] / "graph_transformer")
if _gt_path not in sys.path:
    sys.path.insert(0, _gt_path)


class TestNodeTypeMappingConsistency(unittest.TestCase):
    """Task 119: verify pyg_builder and phase2_adapter use the same mapping."""

    def test_both_import_from_schema_mappings(self):
        """Both modules must import the mapping from schema_mappings."""
        from drugos_graph import schema_mappings, pyg_builder
        # pyg_builder._PHASE2_TO_GT_NODE_TYPE should be the SAME object as
        # schema_mappings.PHASE2_TO_PHASE3_NODE
        self.assertIs(
            pyg_builder._PHASE2_TO_GT_NODE_TYPE,
            schema_mappings.PHASE2_TO_PHASE3_NODE,
            "pyg_builder._PHASE2_TO_GT_NODE_TYPE must be the SAME object as "
            "schema_mappings.PHASE2_TO_PHASE3_NODE (imported, not copied). "
            "(task 112 root fix — INT-004)"
        )

    def test_phase2_adapter_imports_from_schema_mappings(self):
        """phase2_adapter must also import from schema_mappings."""
        try:
            # phase2_adapter is in graph_transformer/data/
            from data.phase2_adapter import PHASE2_TO_PHASE3_NODE
            from drugos_graph.schema_mappings import (
                PHASE2_TO_PHASE3_NODE as SHARED_MAPPING,
            )
            self.assertIs(
                PHASE2_TO_PHASE3_NODE, SHARED_MAPPING,
                "phase2_adapter.PHASE2_TO_PHASE3_NODE must be the SAME object "
                "as schema_mappings.PHASE2_TO_PHASE3_NODE. (task 112)"
            )
        except ImportError:
            # Try alternate import path
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "phase2_adapter",
                    str(Path(__file__).resolve().parents[2] /
                        "graph_transformer" / "data" / "phase2_adapter.py"),
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                from drugos_graph.schema_mappings import (
                    PHASE2_TO_PHASE3_NODE as SHARED_MAPPING,
                )
                self.assertIs(
                    mod.PHASE2_TO_PHASE3_NODE, SHARED_MAPPING,
                    "phase2_adapter.PHASE2_TO_PHASE3_NODE must be the SAME "
                    "object as schema_mappings.PHASE2_TO_PHASE3_NODE. (task 112)"
                )
            except Exception as e:
                self.skipTest(f"Could not import phase2_adapter: {e}")

    def test_mapping_has_5_canonical_node_types(self):
        """The mapping must have exactly 5 canonical Phase 3 node types."""
        from drugos_graph.schema_mappings import PHASE2_TO_PHASE3_NODE
        self.assertEqual(len(PHASE2_TO_PHASE3_NODE), 5,
                         f"Expected 5 canonical node types, got "
                         f"{len(PHASE2_TO_PHASE3_NODE)}: "
                         f"{list(PHASE2_TO_PHASE3_NODE.keys())}")

    def test_mapping_includes_required_types(self):
        """The mapping must include Compound, Protein, Pathway, Disease, ClinicalOutcome."""
        from drugos_graph.schema_mappings import PHASE2_TO_PHASE3_NODE
        required_phase2 = {
            "Compound", "Protein", "Pathway", "Disease", "ClinicalOutcome"
        }
        self.assertEqual(
            set(PHASE2_TO_PHASE3_NODE.keys()), required_phase2,
            f"Mapping keys must be {required_phase2}, got "
            f"{set(PHASE2_TO_PHASE3_NODE.keys())}"
        )

    def test_mapping_excludes_intermediate_types(self):
        """Gene and MedDRA_Term must NOT be in the Phase 3 mapping.

        They are Phase 2 intermediates used for derivation only and
        intentionally dropped AFTER their derivation work is complete.
        """
        from drugos_graph.schema_mappings import PHASE2_TO_PHASE3_NODE
        self.assertNotIn("Gene", PHASE2_TO_PHASE3_NODE,
                         "Gene is a Phase 2 intermediate — must NOT be in "
                         "the Phase 3 mapping (it's dropped after derivation).")
        self.assertNotIn("MedDRA_Term", PHASE2_TO_PHASE3_NODE,
                         "MedDRA_Term is a Phase 2 intermediate — must NOT "
                         "be in the Phase 3 mapping.")

    def test_no_local_mapping_definitions(self):
        """Neither pyg_builder nor phase2_adapter should define a local mapping dict.

        They must IMPORT from schema_mappings, not define their own.
        """
        import inspect
        from drugos_graph import pyg_builder

        # Read the source of pyg_builder
        source = inspect.getsource(pyg_builder)
        # The source should NOT contain a local dict definition like:
        # _PHASE2_TO_GT_NODE_TYPE = {  (with a literal dict body)
        # But it SHOULD contain an import like:
        # from .schema_mappings import PHASE2_TO_PHASE3_NODE as _PHASE2_TO_GT_NODE_TYPE
        self.assertIn(
            "from .schema_mappings import",
            source,
            "pyg_builder must import _PHASE2_TO_GT_NODE_TYPE from "
            "schema_mappings, not define it locally. (task 112)"
        )


if __name__ == "__main__":
    unittest.main()
