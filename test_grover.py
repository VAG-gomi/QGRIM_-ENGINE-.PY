"""Tests for the Grover search circuit and related search primitives."""
import pytest
from conftest import run, assert_norm, Q412_TOL, STAT_TOL
from QGRIM_ENGINE import QGRIMSim, assemble, BUILTIN_CIRCUITS


GROVER_SRC = BUILTIN_CIRCUITS["grover"][1]
BV_SRC     = BUILTIN_CIRCUITS["bv"][1]     # used as a reliable oracle reference


class TestGrover:
    def test_grover_runs_without_error(self):
        prog = assemble(GROVER_SRC)
        sim = QGRIMSim(); sim.run(prog)
        assert sim.halted

    def test_grover_norm_preserved(self):
        """State is normalised after Grover circuit."""
        assert_norm(run(GROVER_SRC))

    def test_grover_all_amplitudes_nonzero(self):
        """After 1 Grover iteration all amplitudes still exist (uniform amplification)."""
        sim = run(GROVER_SRC)
        non_zero = sum(1 for a in sim.state if abs(a) > 1e-4)
        assert non_zero > 0, "No non-zero amplitudes after Grover"

    def test_grover_sum_probabilities_equals_one(self):
        """ΣP = 1 after Grover."""
        sim = run(GROVER_SRC)
        total_prob = sum(abs(a) ** 2 for a in sim.state)
        assert abs(total_prob - 1.0) < 0.002

    def test_grover_no_negative_probs(self):
        """All probabilities ≥ 0."""
        sim = run(GROVER_SRC)
        for i, a in enumerate(sim.state):
            p = abs(a) ** 2
            assert p >= -1e-9, f"Negative probability at |{i:04b}⟩: {p}"

    def test_grover_consistent_over_runs(self):
        """Same seed → same output every time (deterministic before measure)."""
        src = GROVER_SRC
        prog = assemble(src)
        sim1 = QGRIMSim(seed=0); sim1.run(prog)
        sim2 = QGRIMSim(seed=0); sim2.run(prog)
        for i in range(16):
            assert abs(sim1.state[i] - sim2.state[i]) < Q412_TOL

    def test_grover_depth_is_reasonable(self):
        """Grover circuit assembles to >10 instructions (non-trivial depth)."""
        prog = assemble(GROVER_SRC)
        assert len(prog) >= 10, f"Grover unexpectedly short: {len(prog)} words"


class TestBernsteinVazirani:
    """BV is a more reliable oracle test than Grover (always deterministic)."""

    def test_bv_finds_hidden_string_101(self):
        """BV circuit reveals hidden string s=101: q0=1, q1=0, q2=1."""
        sim = run(BV_SRC)
        assert sim.measurements.get(0) == 1, "BV: q0 should be 1"
        assert sim.measurements.get(1) == 0, "BV: q1 should be 0"
        assert sim.measurements.get(2) == 1, "BV: q2 should be 1"

    def test_bv_single_query(self):
        """BV solves in one oracle query — norm preserved."""
        assert_norm(run(BV_SRC))

    def test_bv_100_shots_deterministic(self):
        """BV always produces 101 — deterministic, not statistical."""
        from QGRIM_ENGINE import sample_circuit
        counts = sample_circuit(BV_SRC, shots=100, seed=0)
        # Only one outcome expected
        assert len(counts) == 1, f"BV gave non-deterministic outcomes: {list(counts.keys())}"
        outcome = list(counts.keys())[0]
        # q0=1, q1=0, q2=1, q3=0 → outcome string "1010" (q0,q1,q2,q3)
        assert outcome[0] == '1', f"q0 wrong: {outcome}"
        assert outcome[1] == '0', f"q1 wrong: {outcome}"
        assert outcome[2] == '1', f"q2 wrong: {outcome}"
