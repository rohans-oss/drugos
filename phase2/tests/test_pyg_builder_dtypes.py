"""Task 118: Verify PyG builder dtypes — edge_index int64, node_features float32.

Root tests for tasks 110 and 111:
  - edge_index must be torch.int64 (NOT int32)
  - node features must be torch.float32 (NOT float64 or float16)
"""
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugos_graph.pyg_builder import PyGBuilder


class TestPyGBuilderDtypes(unittest.TestCase):
    """Task 118: verify edge_index int64 and node_features float32."""

    def setUp(self):
        os.environ.pop("DRUGOS_ENVIRONMENT", None)
        os.environ.pop("DRUGOS_ALLOW_XAVIER_FALLBACK", None)

    def test_edge_index_int32_rejected(self):
        """Task 110: int32 edge_index must be REJECTED."""
        builder = PyGBuilder()
        # Build a minimal HeteroData with int32 edge_index
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["drug"].num_nodes = 5
        data["disease"].num_nodes = 5
        # Intentionally use int32 (wrong dtype)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.int32)
        data["drug", "treats", "disease"].edge_index = edge_index

        with self.assertRaises(ValueError) as ctx:
            builder._validate_heterodata(data)
        self.assertIn("task 110", str(ctx.exception).lower() + str(ctx.exception))
        self.assertIn("int64", str(ctx.exception))

    def test_edge_index_int64_accepted(self):
        """Task 110: int64 edge_index is accepted."""
        builder = PyGBuilder()
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["drug"].num_nodes = 5
        data["disease"].num_nodes = 5
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
        data["drug", "treats", "disease"].edge_index = edge_index

        # Should not raise
        try:
            builder._validate_heterodata(data)
        except ValueError as e:
            if "dtype" in str(e) and "int64" in str(e):
                self.fail(f"int64 edge_index was rejected: {e}")
            # Other validation errors are OK (we only care about dtype)

    def test_node_features_float64_rejected(self):
        """Task 111: float64 node features must be REJECTED."""
        builder = PyGBuilder()
        from torch_geometric.data import HeteroData
        data = HeteroData()
        # float64 features (wrong dtype)
        data["drug"].x = torch.randn(5, 8, dtype=torch.float64)
        data["drug"].num_nodes = 5
        data["disease"].num_nodes = 3
        edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data["drug", "treats", "disease"].edge_index = edge_index

        with self.assertRaises(ValueError) as ctx:
            builder._validate_heterodata(data)
        self.assertIn("task 111", str(ctx.exception).lower() + str(ctx.exception))
        self.assertIn("float32", str(ctx.exception))

    def test_node_features_float32_accepted(self):
        """Task 111: float32 node features are accepted."""
        builder = PyGBuilder()
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["drug"].x = torch.randn(5, 8, dtype=torch.float32)
        data["drug"].num_nodes = 5
        data["disease"].num_nodes = 3
        edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data["drug", "treats", "disease"].edge_index = edge_index

        # Should not raise for dtype reasons
        try:
            builder._validate_heterodata(data)
        except ValueError as e:
            if "float32" in str(e) and "dtype" in str(e):
                self.fail(f"float32 features were rejected: {e}")

    def test_node_features_float16_rejected(self):
        """Task 111: float16 node features must be REJECTED (causes gradient underflow)."""
        builder = PyGBuilder()
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["drug"].x = torch.randn(5, 8, dtype=torch.float16)
        data["drug"].num_nodes = 5
        data["disease"].num_nodes = 3
        edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        data["drug", "treats", "disease"].edge_index = edge_index

        with self.assertRaises(ValueError) as ctx:
            builder._validate_heterodata(data)
        self.assertIn("task 111", str(ctx.exception).lower() + str(ctx.exception))

    def test_chemberta_fallback_always_raises(self):
        """Task 109: missing node_features must ALWAYS raise (no Xavier fallback)."""
        builder = PyGBuilder()
        from torch_geometric.data import HeteroData

        entity_maps = {"drug": {f"D{i}": i for i in range(5)}}
        edge_maps = {}

        # No node_features, no feature_provider → must raise
        with self.assertRaises(RuntimeError) as ctx:
            builder.build_from_drkg(entity_maps, edge_maps)
        self.assertIn("task 109", str(ctx.exception).lower() + str(ctx.exception))

    def test_chemberta_fallback_with_explicit_allow_flag(self):
        """Task 109: DRUGOS_ALLOW_XAVIER_FALLBACK=1 allows fallback (with WARNING)."""
        os.environ["DRUGOS_ALLOW_XAVIER_FALLBACK"] = "1"
        try:
            builder = PyGBuilder()
            entity_maps = {"drug": {f"D{i}": i for i in range(5)}}
            edge_maps = {}
            # Should NOT raise (fallback allowed with WARNING)
            try:
                data = builder.build_from_drkg(entity_maps, edge_maps)
                # Verify features were created (Xavier fallback)
                self.assertIsNotNone(data["drug"].x)
                # Verify dtype is float32 (task 111)
                self.assertEqual(data["drug"].x.dtype, torch.float32)
            except RuntimeError as e:
                if "task 109" in str(e).lower():
                    self.fail(f"DRUGOS_ALLOW_XAVIER_FALLBACK=1 should allow "
                              f"fallback, but got: {e}")
        finally:
            os.environ.pop("DRUGOS_ALLOW_XAVIER_FALLBACK", None)


if __name__ == "__main__":
    unittest.main()
