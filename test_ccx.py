"""Tests for the Toffoli (CCX) gate."""
import pytest
from conftest import fresh, run, assert_norm, assert_amp, assert_prob, Q412_TOL
from QGRIM_ENGINE import QGRIMSim


class TestToffoli:
    # ── truth table (c1=q0, c2=q1, target=q2) ────────────────────────────────
    def test_ccx_000_unchanged(self):
        sim = fresh(); sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0000, 1.0)

    def test_ccx_100_unchanged(self):
        """Only c1=1, c2=0: target NOT flipped."""
        sim = fresh(); sim._pauli_x(0); sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0001, 1.0)

    def test_ccx_010_unchanged(self):
        """Only c1=0, c2=1: target NOT flipped."""
        sim = fresh(); sim._pauli_x(1); sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0010, 1.0)

    def test_ccx_110_flips_target(self):
        """c1=1, c2=1, t=0: target flips 0→1."""
        sim = fresh(); sim._pauli_x(0); sim._pauli_x(1); sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0111, 1.0)

    def test_ccx_111_flips_target(self):
        """c1=1, c2=1, t=1: target flips 1→0."""
        sim = fresh()
        sim._pauli_x(0); sim._pauli_x(1); sim._pauli_x(2)
        sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0011, 1.0)

    # ── algebraic properties ──────────────────────────────────────────────────
    def test_ccx_squared_is_identity(self):
        """CCX²=I."""
        sim = fresh()
        sim._pauli_x(0); sim._pauli_x(1)
        sim._ccx(0, 1, 2); sim._ccx(0, 1, 2)
        assert_amp(sim, 0b0011, 1.0)

    def test_ccx_preserves_norm(self):
        """CCX is unitary — norm preserved."""
        sim = fresh()
        sim._hadamard(0); sim._hadamard(1)
        sim._ccx(0, 1, 2)
        assert_norm(sim)

    def test_ccx_via_assembly(self):
        """Assembled CCX works correctly."""
        prog_src = "INIT\nX 0\nX 1\nCCX 0 1 2\nMEASURE 0\nMEASURE 1\nMEASURE 2\nHALT"
        sim = run(prog_src)
        assert sim.measurements[0] == 1
        assert sim.measurements[1] == 1
        assert sim.measurements[2] == 1

    def test_ccx_toffoli_alias(self):
        """TOFFOLI macro assembles identically to CCX."""
        from QGRIM_ENGINE import assemble
        prog_ccx     = assemble("INIT\nCCX 0 1 2\nHALT")
        prog_toffoli = assemble("INIT\nTOFFOLI 0 1 2\nHALT")
        assert prog_ccx == prog_toffoli

    def test_ccx_bounds_error(self):
        from QGRIM_ENGINE import AsmError, assemble
        with pytest.raises(AsmError):
            assemble("CCX 0 1 8")
