"""Tests for the CNOT (controlled-NOT) gate."""
import pytest
from conftest import fresh, run, assert_norm, assert_amp, assert_prob, Q412_TOL
from QGRIM_ENGINE import QGRIMSim, STATES


class TestCNOT:
    # ── truth table ────────────────────────────────────────────────────────────
    def test_cnot_00_unchanged(self):
        """CNOT|00⟩ = |00⟩ (control=0, target unchanged)."""
        sim = fresh(); sim._cnot(0, 1)
        assert_amp(sim, 0b0000, 1.0)

    def test_cnot_01_unchanged(self):
        """CNOT|01⟩ = |01⟩ (control=0, no flip)."""
        sim = fresh(); sim._pauli_x(1); sim._cnot(0, 1)
        assert_amp(sim, 0b0010, 1.0)

    def test_cnot_10_flips_target(self):
        """CNOT|10⟩ = |11⟩ (control=1, target flips 0→1)."""
        sim = fresh(); sim._pauli_x(0); sim._cnot(0, 1)
        assert_amp(sim, 0b0011, 1.0)

    def test_cnot_11_flips_target(self):
        """CNOT|11⟩ = |10⟩ (control=1, target flips 1→0)."""
        sim = fresh(); sim._pauli_x(0); sim._pauli_x(1); sim._cnot(0, 1)
        assert_amp(sim, 0b0001, 1.0)

    # ── algebraic properties ──────────────────────────────────────────────────
    def test_cnot_squared_is_identity(self):
        """CNOT²=I."""
        sim = fresh(); sim._pauli_x(0)
        sim._cnot(0, 1); sim._cnot(0, 1)
        assert_amp(sim, 0b0001, 1.0)

    def test_cnot_preserves_norm(self):
        """‖CNOT|ψ⟩‖ = 1."""
        sim = fresh(); sim._hadamard(0); sim._cnot(0, 1)
        assert_norm(sim)

    # ── Bell state creation ───────────────────────────────────────────────────
    def test_cnot_creates_bell_state(self):
        """H+CNOT on |00⟩ creates (|00⟩+|11⟩)/√2."""
        import math
        sim = fresh(); sim._hadamard(0); sim._cnot(0, 1)
        inv_sqrt2 = 1.0 / math.sqrt(2)
        assert_amp(sim, 0b0000, inv_sqrt2)
        assert_amp(sim, 0b0011, inv_sqrt2)
        assert abs(sim.state[0b0001]) < Q412_TOL
        assert abs(sim.state[0b0010]) < Q412_TOL

    def test_cnot_all_qubit_pairs(self):
        """CNOT works correctly for all distinct control/target pairs."""
        for c in range(4):
            for t in range(4):
                if c == t:
                    continue
                sim = fresh(); sim._pauli_x(c)
                sim._cnot(c, t)
                expected = (1 << c) | (1 << t)
                assert abs(sim.state[expected] - 1.0) < Q412_TOL, \
                    f"CNOT c={c} t={t}: expected |{expected:04b}⟩"

    def test_cnot_via_assembly(self):
        """Assembled CNOT produces Bell state."""
        import math
        prog_src = "INIT\nH 0\nCNOT 0 1\nHALT"
        sim = run(prog_src)
        assert_prob(sim, 0b0000, 0.5)
        assert_prob(sim, 0b0011, 0.5)

    def test_cnot_bounds_error(self):
        """Assembling CNOT with out-of-range qubit raises AsmError."""
        from QGRIM_ENGINE import AsmError, assemble
        with pytest.raises(AsmError):
            assemble("CNOT 0 5")
        with pytest.raises(AsmError):
            assemble("CNOT 4 1")
