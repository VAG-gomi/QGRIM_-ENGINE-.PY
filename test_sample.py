"""Tests for sample_circuit() and render_histogram()."""
import math
import pytest
from conftest import STAT_TOL
from QGRIM_ENGINE import QGRIMSim, sample_circuit, render_histogram, BUILTIN_CIRCUITS, assemble


BELL_SRC = BUILTIN_CIRCUITS["bell"][1]
QRNG_SRC = BUILTIN_CIRCUITS["qrng"][1]


class TestSampler:
    def test_shot_count_matches(self):
        """Total counts == requested shots."""
        for n in [1, 10, 100, 1000]:
            counts = sample_circuit(BELL_SRC, shots=n, seed=42)
            assert sum(counts.values()) == n, f"shots={n}: total={sum(counts.values())}"

    def test_seeded_reproducibility(self):
        """Same seed → identical histogram."""
        c1 = sample_circuit(BELL_SRC, shots=500, seed=7)
        c2 = sample_circuit(BELL_SRC, shots=500, seed=7)
        assert c1 == c2

    def test_different_seeds_different_results(self):
        """Different seeds → different histograms (with high probability)."""
        c1 = sample_circuit(BELL_SRC, shots=200, seed=1)
        c2 = sample_circuit(BELL_SRC, shots=200, seed=99999)
        assert c1 != c2

    def test_bell_only_00_and_11_outcomes(self):
        """Bell circuit: only |0000⟩ and |1100⟩ outcomes exist."""
        counts = sample_circuit(BELL_SRC, shots=500, seed=42)
        for k in counts:
            assert k in ("0000", "1100"), f"Unexpected Bell outcome: {k}"

    def test_bell_50_50_distribution(self):
        """Bell: P(00) ≈ P(11) ≈ 0.5 over 2000 shots."""
        counts = sample_circuit(BELL_SRC, shots=2000, seed=0)
        total = sum(counts.values())
        p00 = counts.get("0000", 0) / total
        p11 = counts.get("1100", 0) / total
        assert abs(p00 - 0.5) < STAT_TOL, f"P(00)={p00:.3f}"
        assert abs(p11 - 0.5) < STAT_TOL, f"P(11)={p11:.3f}"

    def test_qrng_all_16_outcomes(self):
        """QRNG with enough shots produces all 16 4-bit outcomes."""
        counts = sample_circuit(QRNG_SRC, shots=5000, seed=42)
        assert len(counts) == 16, f"QRNG only gave {len(counts)} distinct outcomes"

    def test_qrng_uniformity_chi_square(self):
        """QRNG passes chi-square uniformity test at 10,000 shots (χ²<25 for df=15)."""
        counts = sample_circuit(QRNG_SRC, shots=10000, seed=42)
        expected = 10000 / 16
        chi2 = sum((counts.get(f"{i:04b}", 0) - expected) ** 2 / expected
                   for i in range(16))
        assert chi2 < 25.0, f"QRNG chi²={chi2:.2f} exceeds 25.0 (not uniform)"

    def test_noise_changes_distribution(self):
        """With noise_p=0.3, Bell produces non-ideal outcomes."""
        clean = sample_circuit(BELL_SRC, shots=500, seed=42, noise_p=0.0)
        noisy = sample_circuit(BELL_SRC, shots=500, seed=42, noise_p=0.3)
        # Clean: only 2 outcomes; noisy: more than 2
        assert len(noisy) > len(clean), "High noise should introduce extra outcomes"

    def test_histogram_renders_without_crash(self):
        """render_histogram returns a non-empty string."""
        counts = sample_circuit(BELL_SRC, shots=100, seed=42)
        h = render_histogram(counts, 100)
        assert isinstance(h, str) and len(h) > 10

    def test_histogram_contains_outcomes(self):
        """Histogram string includes each measured outcome."""
        counts = sample_circuit(BELL_SRC, shots=100, seed=42)
        h = render_histogram(counts, 100)
        for k in counts:
            assert k in h, f"Outcome {k} missing from histogram"

    def test_single_shot(self):
        """1 shot: exactly one outcome with count 1."""
        counts = sample_circuit(BELL_SRC, shots=1, seed=0)
        assert sum(counts.values()) == 1
        assert all(v == 1 for v in counts.values())

    def test_progress_kwarg_accepted(self):
        """sample_circuit accepts progress=False without error."""
        counts = sample_circuit(BELL_SRC, shots=10, seed=0, progress=False)
        assert sum(counts.values()) == 10
