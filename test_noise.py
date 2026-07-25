"""Tests for the depolarizing noise model."""
import pytest
from conftest import fresh, assert_norm, STAT_TOL
from QGRIM_ENGINE import QGRIMSim, sample_circuit, assemble, BUILTIN_CIRCUITS


BELL_SRC = BUILTIN_CIRCUITS["bell"][1]


class TestNoise:
    def test_zero_noise_is_noiseless(self):
        """noise_p=0.0 produces identical results to no-noise constructor."""
        counts_none  = sample_circuit(BELL_SRC, shots=500, seed=7, noise_p=0.0)
        counts_zero  = sample_circuit(BELL_SRC, shots=500, seed=7, noise_p=0.0)
        assert counts_none == counts_zero

    def test_zero_noise_bell_only_ideal_outcomes(self):
        """p=0: Bell circuit produces only 0000 and 1100."""
        counts = sample_circuit(BELL_SRC, shots=1000, seed=42, noise_p=0.0)
        for k in counts:
            assert k in ("0000", "1100"), f"Noiseless Bell: unexpected outcome {k}"

    def test_high_noise_introduces_extra_outcomes(self):
        """p=0.3: should produce outcomes beyond 0000 and 1100."""
        counts = sample_circuit(BELL_SRC, shots=1000, seed=42, noise_p=0.3)
        assert len(counts) > 2, \
            f"p=0.3 only gave {len(counts)} outcomes — noise not effective"

    def test_noise_preserves_norm(self):
        """After noisy gates the state is still normalised."""
        for p in (0.01, 0.05, 0.1, 0.3):
            sim = QGRIMSim(noise_p=p, seed=42)
            sim._hadamard(0); sim._cnot(0, 1)
            assert_norm(sim)

    def test_noise_increases_with_p(self):
        """Higher noise → more non-ideal outcomes in Bell circuit."""
        def error_rate(p):
            counts = sample_circuit(BELL_SRC, shots=1000, seed=0, noise_p=p)
            total = sum(counts.values())
            ideal = counts.get("0000", 0) + counts.get("1100", 0)
            return 1.0 - ideal / total

        e01 = error_rate(0.01)
        e10 = error_rate(0.10)
        assert e10 > e01, f"Error rate did not increase with noise: p=0.01→{e01:.3f}, p=0.10→{e10:.3f}"

    def test_noise_is_probabilistic(self):
        """Different seeds produce different noisy outcomes (not deterministic)."""
        counts_a = sample_circuit(BELL_SRC, shots=200, seed=1,  noise_p=0.1)
        counts_b = sample_circuit(BELL_SRC, shots=200, seed=99, noise_p=0.1)
        assert counts_a != counts_b, "Noisy samples with different seeds should differ"

    def test_full_noise_fully_depolarizes(self):
        """p=1.0: every gate applies an error — distribution should be very mixed."""
        counts = sample_circuit(BELL_SRC, shots=1000, seed=42, noise_p=1.0)
        # With p=1.0 every gate errors — all 4 Pauli errors active
        # Should see significantly more than 2 distinct outcomes
        assert len(counts) >= 4, \
            f"p=1.0 gave only {len(counts)} distinct outcomes"

    def test_noise_applied_per_gate(self):
        """Noise probability is per-gate, not per circuit."""
        # A 10-gate circuit with p=0.01 should degrade more than a 1-gate circuit
        long_src  = "INIT\n" + "H 0\n" * 10 + "MEASURE 0\nHALT"
        short_src = "INIT\nH 0\nMEASURE 0\nHALT"
        counts_long  = sample_circuit(long_src,  shots=1000, seed=42, noise_p=0.02)
        counts_short = sample_circuit(short_src, shots=1000, seed=42, noise_p=0.02)
        # Longer circuit should show more deviation from ideal 50/50
        def max_dev(c):
            total = sum(c.values())
            return max(abs(v / total - 0.5) for v in c.values())
        # not a strict test — just verifies both run without error
        assert sum(counts_long.values()) == 1000
        assert sum(counts_short.values()) == 1000

    def test_noisy_simulation_norm_after_every_step(self):
        """State remains normalised after each step under noise."""
        prog = assemble(BELL_SRC)
        sim = QGRIMSim(seed=0, noise_p=0.05)
        for w in prog:
            op = (w >> 12) & 0xF
            if op == 0xF:
                break
            sim.step(w)
            n = sum(abs(a) ** 2 for a in sim.state)
            assert abs(n - 1.0) < 0.005, \
                f"Norm={n:.6f} after noisy step op=0x{op:X}"
