"""Tests for entanglement_entropy() — von Neumann entropy via Schmidt decomp."""
import math
import pytest
from conftest import run, fresh, Q412_TOL
from QGRIM_ENGINE import QGRIMSim, assemble


class TestEntropy:
    # ── separable (product) states — entropy must be 0 ──────────────────────

    def test_entropy_zero_state(self):
        """S(|0000⟩) = 0 for any bipartition."""
        sim = fresh()
        for q in range(4):
            S = sim.entanglement_entropy([q])
            assert abs(S) < 0.01, f"Product |0000⟩ entropy q{q}={S:.4f}"

    def test_entropy_x_state_separable(self):
        """Single X gate: |0001⟩ is a product state — S = 0."""
        sim = fresh(); sim._pauli_x(0)
        S = sim.entanglement_entropy([0])
        assert abs(S) < 0.01, f"S(|0001⟩)={S:.4f}"

    def test_entropy_superposition_all_separable(self):
        """H on all qubits gives product state — S(q0|rest)≈0."""
        sim = run("INIT\nH 0\nH 1\nH 2\nH 3\nHALT")
        for q in range(4):
            S = sim.entanglement_entropy([q])
            assert abs(S) < 0.02, f"Uniform superposition not separable? S(q{q})={S:.4f}"

    # ── maximally entangled states — entropy must be 1 bit ──────────────────

    def test_entropy_bell_q0(self):
        """Bell state: S(q0 | q1q2q3) = 1 bit."""
        sim = run("INIT\nH 0\nCNOT 0 1\nHALT")
        S = sim.entanglement_entropy([0])
        assert abs(S - 1.0) < 0.02, f"Bell S(q0)={S:.4f}"

    def test_entropy_bell_q1(self):
        """Bell state: S(q1 | q0q2q3) = 1 bit."""
        sim = run("INIT\nH 0\nCNOT 0 1\nHALT")
        S = sim.entanglement_entropy([1])
        assert abs(S - 1.0) < 0.02, f"Bell S(q1)={S:.4f}"

    def test_entropy_bell_q2_zero(self):
        """Bell state: S(q2 | q0q1q3) = 0 (q2 uninvolved)."""
        sim = run("INIT\nH 0\nCNOT 0 1\nHALT")
        S = sim.entanglement_entropy([2])
        assert abs(S) < 0.02, f"Bell S(q2 uninvolved)={S:.4f}"

    def test_entropy_ghz_q0(self):
        """GHZ: S(q0 | rest) = 1 bit."""
        sim = run("INIT\nH 0\nCNOT 0 1\nCNOT 1 2\nHALT")
        S = sim.entanglement_entropy([0])
        assert abs(S - 1.0) < 0.02, f"GHZ S(q0)={S:.4f}"

    def test_entropy_ghz_all_entangled(self):
        """GHZ: qubits 0,1,2 all show S=1, qubit 3 shows S=0."""
        sim = run("INIT\nH 0\nCNOT 0 1\nCNOT 1 2\nHALT")
        for q in [0, 1, 2]:
            S = sim.entanglement_entropy([q])
            assert abs(S - 1.0) < 0.02, f"GHZ S(q{q})={S:.4f}"
        S3 = sim.entanglement_entropy([3])
        assert abs(S3) < 0.02, f"GHZ uninvolved q3 S={S3:.4f}"

    def test_entropy_cluster_state(self):
        """4-qubit cluster state: all single-qubit cuts show entanglement."""
        sim = run("INIT\nH 0\nH 1\nH 2\nH 3\nCZ 0 1\nCZ 1 2\nCZ 2 3\nHALT")
        for q in range(4):
            S = sim.entanglement_entropy([q])
            assert S > 0.5, f"Cluster S(q{q})={S:.4f} unexpectedly low"

    # ── non-trivial partition ─────────────────────────────────────────────────

    def test_entropy_two_qubit_partition(self):
        """Bell state: S(q0,q1 | q2,q3) = 0 (entanglement internal to A)."""
        sim = run("INIT\nH 0\nCNOT 0 1\nHALT")
        S = sim.entanglement_entropy([0, 1])
        assert abs(S) < 0.02, f"Bell S(q01|q23)={S:.4f} should be 0"

    # ── performance check ─────────────────────────────────────────────────────

    def test_entropy_1qubit_faster_than_half_second(self):
        """Analytic 2×2 path should be fast (<500ms for 100 calls)."""
        import time
        sim = run("INIT\nH 0\nCNOT 0 1\nHALT")
        t0 = time.perf_counter()
        for _ in range(100):
            sim.entanglement_entropy([0])
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, \
            f"1-qubit entropy too slow: {elapsed:.3f}s for 100 calls"
