"""Tests for the MEASURE gate (Born-rule collapse)."""
import pytest
from conftest import fresh, run, assert_norm, Q412_TOL, STAT_TOL
from QGRIM_ENGINE import QGRIMSim, assemble, STATES


class TestMeasure:
    def test_measure_zero_gives_zero(self):
        """Measuring |0⟩ always returns 0."""
        for _ in range(20):
            sim = fresh()
            result = sim._measure(0)
            assert result == 0, "MEASURE on |0⟩ must always yield 0"

    def test_measure_one_gives_one(self):
        """Measuring |1⟩ always returns 1."""
        for _ in range(20):
            sim = fresh(); sim._pauli_x(0)
            result = sim._measure(0)
            assert result == 1, "MEASURE on |1⟩ must always yield 1"

    def test_measure_collapses_superposition(self):
        """After measuring |+⟩ the state collapses to a definite eigenstate."""
        sim = fresh(); sim._hadamard(0)
        result = sim._measure(0)
        assert result in (0, 1)
        # Post-collapse: the measured qubit must have probability 1
        non_zero = [i for i in range(STATES) if abs(sim.state[i]) > 1e-4]
        for idx in non_zero:
            qubit_val = (idx >> 0) & 1
            assert qubit_val == result, \
                f"Post-collapse state |{idx:04b}⟩ inconsistent with measured {result}"

    def test_measure_repeated_is_idempotent(self):
        """Measuring the same qubit twice gives the same result."""
        sim = fresh(); sim._hadamard(0)
        r1 = sim._measure(0)
        r2 = sim._measure(0)
        assert r1 == r2, f"Repeated MEASURE gave {r1} then {r2}"

    def test_measure_preserves_norm(self):
        """State is renormalised after collapse."""
        sim = fresh(); sim._hadamard(0); sim._cnot(0, 1)
        sim._measure(0)
        assert_norm(sim)

    def test_measure_statistical_50_50(self):
        """Measuring |+⟩ gives 0 and 1 each ~50% of the time."""
        n = 2000; zeros = 0
        for i in range(n):
            sim = QGRIMSim(seed=i)
            sim._hadamard(0)
            zeros += (sim._measure(0) == 0)
        p0 = zeros / n
        assert abs(p0 - 0.5) < STAT_TOL, \
            f"P(0) after H was {p0:.3f}, expected 0.5 ± {STAT_TOL}"

    def test_measure_updates_measurements_dict(self):
        """MEASURE records result in sim.measurements."""
        sim = fresh(); sim._measure(0)
        assert 0 in sim.measurements
        assert sim.measurements[0] == 0

    def test_measure_all_qubits(self):
        """MEASURE on each qubit independently works."""
        for q in range(4):
            sim = fresh()
            result = sim._measure(q)
            assert result == 0
            assert sim.measurements[q] == 0

    def test_measure_via_assembly(self):
        """Assembled MEASURE instruction records correct result."""
        prog = assemble("INIT\nX 0\nMEASURE 0\nHALT")
        sim = QGRIMSim(); sim.run(prog)
        assert sim.measurements[0] == 1

    def test_measure_entangled_pair_consistent(self):
        """Bell state: when q0=0 then q1=0, when q0=1 then q1=1 (always)."""
        for seed in range(200):
            sim = QGRIMSim(seed=seed)
            sim._hadamard(0); sim._cnot(0, 1)
            r0 = sim._measure(0)
            r1 = sim._measure(1)
            assert r0 == r1, f"Bell mismatch: q0={r0}, q1={r1} at seed={seed}"
