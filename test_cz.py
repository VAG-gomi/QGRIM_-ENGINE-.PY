"""Tests for the Controlled-Z (CZ) gate."""
import pytest
from conftest import fresh, run, assert_norm, assert_amp, assert_prob, Q412_TOL
from QGRIM_ENGINE import QGRIMSim


class TestCZ:
    def test_cz_00_unchanged(self):
        """CZ|00⟩ = |00⟩ (no phase on |00⟩)."""
        sim = fresh(); sim._cz(0, 1)
        assert_amp(sim, 0b0000, 1.0)

    def test_cz_10_unchanged(self):
        """CZ|10⟩ = |10⟩ (phase only when both=1)."""
        sim = fresh(); sim._pauli_x(0); sim._cz(0, 1)
        assert_amp(sim, 0b0001, 1.0)

    def test_cz_01_unchanged(self):
        """CZ|01⟩ = |01⟩."""
        sim = fresh(); sim._pauli_x(1); sim._cz(0, 1)
        assert_amp(sim, 0b0010, 1.0)

    def test_cz_11_flips_phase(self):
        """CZ|11⟩ = -|11⟩ — the phase kick on |11⟩."""
        sim = fresh(); sim._pauli_x(0); sim._pauli_x(1); sim._cz(0, 1)
        assert_amp(sim, 0b0011, -1.0)

    def test_cz_squared_is_identity(self):
        """CZ²=I on |11⟩ (double phase flip = no flip)."""
        sim = fresh(); sim._pauli_x(0); sim._pauli_x(1)
        sim._cz(0, 1); sim._cz(0, 1)
        assert_amp(sim, 0b0011, 1.0)

    def test_cz_preserves_norm(self):
        """CZ is unitary — preserves norm."""
        sim = fresh(); sim._hadamard(0); sim._hadamard(1); sim._cz(0, 1)
        assert_norm(sim)

    def test_cz_on_plus_plus(self):
        """|++⟩ after CZ: |11⟩ component gets -1 phase."""
        import math
        sim = fresh(); sim._hadamard(0); sim._hadamard(1); sim._cz(0, 1)
        # (|00⟩+|01⟩+|10⟩-|11⟩)/2
        half = 0.5
        assert_amp(sim, 0b0000,  half)
        assert_amp(sim, 0b0010,  half)
        assert_amp(sim, 0b0001,  half)
        assert_amp(sim, 0b0011, -half)

    def test_cz_symmetric(self):
        """CZ(c,t) = CZ(t,c) — symmetric gate."""
        import math
        sim1 = fresh(); sim1._hadamard(0); sim1._hadamard(1); sim1._cz(0, 1)
        sim2 = fresh(); sim2._hadamard(0); sim2._hadamard(1); sim2._cz(1, 0)
        for i in range(16):
            assert abs(sim1.state[i] - sim2.state[i]) < Q412_TOL, \
                f"CZ not symmetric at basis {i:04b}"

    def test_cz_via_assembly(self):
        """Assembled CZ instruction correct."""
        prog_src = "INIT\nX 0\nX 1\nCZ 0 1\nHALT"
        sim = run(prog_src)
        assert_amp(sim, 0b0011, -1.0)
