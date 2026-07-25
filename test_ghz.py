"""Tests for the GHZ (Greenberger–Horne–Zeilinger) state circuit."""
import math
import pytest
from conftest import run, assert_norm, assert_amp, assert_prob, Q412_TOL, STAT_TOL
from QGRIM_ENGINE import QGRIMSim, assemble, BUILTIN_CIRCUITS


INV_SQRT2 = 1.0 / math.sqrt(2)
GHZ_SRC = "INIT\nH 0\nCNOT 0 1\nCNOT 1 2\nHALT"  # no-measure version


class TestGHZState:
    def test_ghz_amplitudes(self):
        """GHZ = (|0000⟩+|0111⟩)/√2."""
        sim = run(GHZ_SRC)
        assert_amp(sim, 0b0000, INV_SQRT2)
        assert_amp(sim, 0b0111, INV_SQRT2)

    def test_ghz_only_two_nonzero(self):
        sim = run(GHZ_SRC)
        for i in range(16):
            if i not in (0b0000, 0b0111):
                assert abs(sim.state[i]) < Q412_TOL

    def test_ghz_norm(self):
        assert_norm(run(GHZ_SRC))

    def test_ghz_equal_probabilities(self):
        sim = run(GHZ_SRC)
        assert_prob(sim, 0b0000, 0.5)
        assert_prob(sim, 0b0111, 0.5)

    def test_ghz_measurement_all_same(self):
        """200 shots: q0, q1, q2 always all-0 or all-1."""
        for seed in range(200):
            sim = QGRIMSim(seed=seed)
            sim._hadamard(0); sim._cnot(0, 1); sim._cnot(1, 2)
            r0 = sim._measure(0)
            r1 = sim._measure(1)
            r2 = sim._measure(2)
            assert r0 == r1 == r2, \
                f"GHZ mismatch at seed={seed}: {r0},{r1},{r2}"

    def test_ghz_builtin_circuit(self):
        """Built-in GHZ circuit runs and produces correct measurements."""
        prog = assemble(BUILTIN_CIRCUITS["ghz"][1])
        sim = QGRIMSim(); sim.run(prog)
        m = sim.measurements
        assert m[0] == m[1] == m[2], f"GHZ measurements {m} not all equal"

    def test_ghz_entanglement_entropy(self):
        """S(q0 | rest) = 1 bit for GHZ."""
        sim = run(GHZ_SRC)
        S = sim.entanglement_entropy([0])
        assert abs(S - 1.0) < 0.02, f"GHZ entropy(q0)={S:.4f}"

    def test_ghz_statistical(self):
        """1000 shots: only 000 and 111 outcomes (in measurement bits)."""
        from QGRIM_ENGINE import sample_circuit
        counts = sample_circuit(BUILTIN_CIRCUITS["ghz"][1], shots=1000, seed=42)
        for outcome in counts:
            # measurements are q0,q1,q2 (q3 unmeasured stays 0)
            assert outcome in ("0000", "1110"), \
                f"Unexpected GHZ outcome: {outcome}"
