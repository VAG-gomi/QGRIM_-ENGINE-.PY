"""Tests for the SWAP gate."""
import pytest
from conftest import fresh, run, assert_norm, assert_amp, assert_prob, Q412_TOL
from QGRIM_ENGINE import QGRIMSim


class TestSWAP:
    def test_swap_01_gives_10(self):
        """SWAP|01⟩ = |10⟩."""
        sim = fresh(); sim._pauli_x(1); sim._swap(0, 1)
        # q0=1 was originally in q1 position; after swap q0 is set
        # |0001⟩ (q1=1) → |0010⟩ (q0=1) — wait, bit 0 = q0, bit 1 = q1
        # SWAP(0,1): bit 0 ↔ bit 1 → |0001⟩ becomes |0010⟩? Let me check:
        # state[i]: bit position q means qubit q = (i >> q) & 1
        # X on q1: sets bit 1 → state[0b0010] = 1
        # SWAP(0,1): exchanges bits 0 and 1
        # 0b0010 (bit1=1,bit0=0) → 0b0001 (bit1=0,bit0=1)
        assert_amp(sim, 0b0001, 1.0)

    def test_swap_10_gives_01(self):
        """SWAP|10⟩ = |01⟩."""
        sim = fresh(); sim._pauli_x(0); sim._swap(0, 1)
        # X on q0: state[0b0001]=1
        # SWAP(0,1): bit0↔bit1 → 0b0001 → 0b0010
        assert_amp(sim, 0b0010, 1.0)

    def test_swap_00_unchanged(self):
        """SWAP|00⟩ = |00⟩."""
        sim = fresh(); sim._swap(0, 1)
        assert_amp(sim, 0b0000, 1.0)

    def test_swap_11_unchanged(self):
        """SWAP|11⟩ = |11⟩."""
        sim = fresh(); sim._pauli_x(0); sim._pauli_x(1); sim._swap(0, 1)
        assert_amp(sim, 0b0011, 1.0)

    def test_swap_squared_is_identity(self):
        """SWAP²=I."""
        sim = fresh(); sim._pauli_x(0); sim._swap(0, 1); sim._swap(0, 1)
        assert_amp(sim, 0b0001, 1.0)

    def test_swap_preserves_norm(self):
        """SWAP is unitary."""
        sim = fresh(); sim._hadamard(0); sim._swap(0, 1)
        assert_norm(sim)

    def test_swap_all_pairs(self):
        """SWAP on each qubit pair correctly exchanges states."""
        for a in range(4):
            for b in range(4):
                if a == b:
                    continue
                sim = fresh(); sim._pauli_x(a); sim._swap(a, b)
                assert_norm(sim)
                assert abs(sim.state[1 << b] - 1.0) < Q412_TOL, \
                    f"SWAP({a},{b}) failed: expected |{1<<b:04b}⟩"

    def test_swap_via_assembly(self):
        """Assembled SWAP works correctly."""
        prog_src = "INIT\nX 0\nSWAP 0 1\nHALT"
        sim = run(prog_src)
        # X on q0 → bit0=1 → state[0b0001]; SWAP(0,1) → bit1=1 → state[0b0010]
        assert_amp(sim, 0b0010, 1.0)
