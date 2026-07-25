"""Tests for the Bell state circuit."""
import math
import pytest
from conftest import run, assert_norm, assert_amp, assert_prob, Q412_TOL, STAT_TOL
from QGRIM_ENGINE import QGRIMSim, assemble, BUILTIN_CIRCUITS


INV_SQRT2 = 1.0 / math.sqrt(2)

BELL_SRC       = "INIT\nH 0\nCNOT 0 1\nHALT"
BELL_MEAS_SRC  = BUILTIN_CIRCUITS["bell"][1]


class TestBellState:
    def test_bell_amplitudes(self):
        """|Φ+⟩ = (|0000⟩+|0011⟩)/√2 — two equal real amplitudes."""
        sim = run(BELL_SRC)
        assert_amp(sim, 0b0000, INV_SQRT2)
        assert_amp(sim, 0b0011, INV_SQRT2)

    def test_bell_only_two_nonzero(self):
        """Only |0000⟩ and |0011⟩ are non-zero in the Bell state."""
        sim = run(BELL_SRC)
        for i in range(16):
            if i not in (0b0000, 0b0011):
                assert abs(sim.state[i]) < Q412_TOL, \
                    f"Unexpected non-zero at |{i:04b}⟩: {sim.state[i]}"

    def test_bell_norm(self):
        assert_norm(run(BELL_SRC))

    def test_bell_equal_probabilities(self):
        """P(|0000⟩) = P(|0011⟩) = 0.5."""
        sim = run(BELL_SRC)
        assert_prob(sim, 0b0000, 0.5)
        assert_prob(sim, 0b0011, 0.5)

    def test_bell_measurement_correlated(self):
        """500 shots: q0 and q1 always agree (00 or 11), never 01 or 10."""
        for seed in range(500):
            sim = QGRIMSim(seed=seed)
            sim._hadamard(0); sim._cnot(0, 1)
            r0 = sim._measure(0)
            r1 = sim._measure(1)
            assert r0 == r1, f"Anti-correlated at seed={seed}: q0={r0} q1={r1}"

    def test_bell_builtin_circuit_runs(self):
        """Built-in bell circuit assembles and runs without error."""
        prog = assemble(BELL_MEAS_SRC)
        sim = QGRIMSim(); sim.run(prog)
        assert 0 in sim.measurements
        assert 1 in sim.measurements
        assert sim.measurements[0] == sim.measurements[1]

    def test_bell_statistical_distribution(self):
        """Over 2000 shots: P(|0000⟩) and P(|1100⟩) each ≈ 50%."""
        from QGRIM_ENGINE import sample_circuit
        counts = sample_circuit(BELL_MEAS_SRC, shots=2000, seed=42)
        total = sum(counts.values())
        p00 = counts.get("0000", 0) / total
        p11 = counts.get("1100", 0) / total
        assert abs(p00 - 0.5) < STAT_TOL, f"P(00)={p00:.3f}"
        assert abs(p11 - 0.5) < STAT_TOL, f"P(11)={p11:.3f}"
        # No other outcomes
        for k in counts:
            assert k in ("0000", "1100"), f"Unexpected outcome {k}"

    def test_bell_entanglement_entropy(self):
        """S(q0 | q1q2q3) = 1 bit for Bell state."""
        sim = run(BELL_SRC)
        S = sim.entanglement_entropy([0])
        assert abs(S - 1.0) < 0.01, f"Bell entropy = {S:.4f}, expected 1.0"
