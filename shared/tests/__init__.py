"""
shared.tests — cross-phase integration tests.

These tests exercise the FULL data flywheel (validate → writeback → trainer
fine-tune → RL ranker loads new bonuses/penalties), not just individual
components. They are the acceptance criteria for issues 349-351.

Tests:
    test_data_flywheel_e2e.py        — issue #349: end-to-end flywheel.
    test_flywheel_toxic_penalty.py   — issue #350: toxic → negative reward.
    test_flywheel_checkpoint_atomic.py — issue #351: atomic checkpoint save.
    test_contract_consistency.py     — Task 330: writer/reader contract match.
"""
