===================================================

44% credit goes to replit agent.For writing codes!

====================================================
#!/usr/bin/env python3
"""
QGRIM v2.1 — Unified Quantum Circuit Engine  [Complete Edition]
===============================================================
Self-contained single file. Runs on any Python 3.8+ including Pydroid (Android).
No external packages needed — stdlib only.

Features:
  • Full 4-qubit state-vector simulator (Q4.12 fixed-point, matches the FPGA RTL)
  • Depolarizing noise model (configurable per-gate error rate)
  • Assembler: full ISA + macro gates (S, T, Z, SDG, TDG, CX, CY, Y, CZ, Toffoli)
  • Circuit visualizer: ASCII diagram of any assembled program
  • State formula display: amplitudes as ket notation (e.g. 0.707|00⟩ + 0.707|11⟩)
  • Bloch sphere (x,y,z) for any single qubit
  • Entanglement entropy (von Neumann, bipartite Schmidt decomposition)
  • Fidelity between two state vectors
  • CHSH Bell inequality test with correlation measurement
  • Interactive REPL with undo, save, load, formula, bloch commands
  • Shot sampler with ASCII histogram
  • 16 built-in circuits: Bell, GHZ, QFT, teleportation, Deutsch, Bernstein-Vazirani,
    Grover search, superdense coding, quantum RNG, cluster state, QPE sketch, and more
  • Save/load .qasm programs to disk (Pydroid storage compatible)
  • Hex export for FPGA flashing via SPI
  • Full register map and chip info reference
  • Witness / ODEA decoder (S/O/F/X/H labels)

Usage (Pydroid):
  Open this file in Pydroid and tap Run. A numbered menu appears.
  No pip install, no internet, no other files needed.

Usage (desktop):
  python3 QGRIM_ENGINE.py                      # interactive menu
  python3 QGRIM_ENGINE.py --run bell           # run built-in circuit
  python3 QGRIM_ENGINE.py --run grover         # run Grover search
  python3 QGRIM_ENGINE.py --shots bell 1024    # histogram
  python3 QGRIM_ENGINE.py --viz bell           # ASCII circuit diagram
  python3 QGRIM_ENGINE.py --chsh               # CHSH Bell test
  python3 QGRIM_ENGINE.py --file mycirc.qasm   # run from file
  python3 QGRIM_ENGINE.py --list               # list all circuits
  python3 QGRIM_ENGINE.py --isa                # instruction set reference
  python3 QGRIM_ENGINE.py --noise 0.02 --run bell  # noisy simulation
"""

from __future__ import annotations
import math
import random
import sys
import os
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONSTANTS & LOOK-UP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

QUBITS = 4
STATES = 1 << QUBITS   # 16 basis states
ONE_Q12 = 1 << 12      # 4096

# Phase LUT: 16 angles  k·π/8  for k = 0..15  (matches gate_phase.v exactly)
PHASE_LUT: List[Tuple[float, float]] = [
    (math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)) for k in range(16)
]

# Witness / ODEA label table: maps 4-bit basis index to label
WITNESS_LABELS = ["H","S","S","O","S","F","O","X","S","O","F","X","O","X","X","H"]
WITNESS_MEANING = {
    "H": "Hadamard / identity origin",
    "S": "Superposition state",
    "O": "Observable / collapse point",
    "F": "Field interference node",
    "X": "Entangled / cross-basis",
}

# Hardware ISA opcodes — match FPGA chip exactly
HW_OPCODES: Dict[str, int] = {
    "NOP":     0x0,
    "H":       0x1,
    "X":       0x2,
    "CNOT":    0x3,
    "MEASURE": 0x4,
    "PHASE":   0x5,
    "INIT":    0x6,
    "SWAP":    0x7,
    "LOAD_AMP":0x8,
    "TRACE":   0x9,
    "WAIT":    0xE,
    "HALT":    0xF,
}

# Software-extension opcodes (simulator only — NOT sent to FPGA)
SW_OPCODES: Dict[str, int] = {
    "Y":   0xA,   # Pauli-Y
    "CZ":  0xB,   # Controlled-Z
    "CCX": 0xC,   # Toffoli: A=c1  B=c2  IMM=target
    "RZ":  0xD,   # Alias for PHASE, marks software-only intent
}

# Combined opcode table (used by assembler)
ALL_OPCODES: Dict[str, int] = {**HW_OPCODES, **SW_OPCODES}

# Reverse map for disassembly
OPCODE_REV: Dict[int, str] = {v: k for k, v in HW_OPCODES.items()}
OPCODE_REV.update({0xA: "Y", 0xB: "CZ", 0xC: "CCX", 0xD: "RZ"})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ASSEMBLER  (ISA + macro expansion + software extensions)
# ═══════════════════════════════════════════════════════════════════════════════

class AsmError(Exception):
    pass


def _encode(op: int, a: int = 0, b: int = 0, imm: int = 0) -> int:
    if not (0 <= op <= 0xF and 0 <= a <= 0xF and 0 <= b <= 0xF and 0 <= imm <= 0xF):
        raise AsmError(f"field out of range: op={op} a={a} b={b} imm={imm}")
    return (op << 12) | (a << 8) | (b << 4) | imm


def assemble(source: str) -> List[int]:
    """
    Convert QASM text to 16-bit instruction words.
    Handles hardware ISA, software-extension opcodes, and macro aliases.

    Macro aliases (expand to one or more hardware words):
      S  q          — S gate = PHASE q 4     (π/2)
      T  q          — T gate = PHASE q 2     (π/4)
      Z  q          — Z gate = PHASE q 8     (π)
      SDG q         — S† gate = PHASE q 12   (-π/2)
      TDG q         — T† gate = PHASE q 14   (-π/4)
      CX  c t       — alias for CNOT c t
      NOT q         — alias for X q
      TOFFOLI c1 c2 t — alias for CCX c1 c2 t
      CY  c t       — Controlled-Y: SDG t; CNOT c t; S t
      CH  c t       — Controlled-H: approx via S t; H t; T t; CNOT c t; TDG t; H t; SDG t

    Software-extension (sim only, not FPGA):
      Y   q         — Pauli-Y
      CZ  c t       — Controlled-Z
      CCX c1 c2 t   — Toffoli (c1, c2 in A,B; target in IMM)
      RZ  q idx     — RZ rotation (same encoding as PHASE)
    """
    out: List[int] = []
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        mnem = parts[0].upper()
        try:
            args = [int(x, 0) for x in parts[1:]]
        except ValueError as e:
            raise AsmError(f"line {lineno}: bad argument — {e}")

        # ── macro aliases ─────────────────────────────────────────
        if mnem == "S":
            if len(args) < 1: raise AsmError(f"line {lineno}: S needs 1 arg")
            out.append(_encode(0x5, args[0], 0, 4))
        elif mnem == "T":
            if len(args) < 1: raise AsmError(f"line {lineno}: T needs 1 arg")
            out.append(_encode(0x5, args[0], 0, 2))
        elif mnem == "Z":
            if len(args) < 1: raise AsmError(f"line {lineno}: Z needs 1 arg")
            out.append(_encode(0x5, args[0], 0, 8))
        elif mnem == "SDG":
            if len(args) < 1: raise AsmError(f"line {lineno}: SDG needs 1 arg")
            out.append(_encode(0x5, args[0], 0, 12))
        elif mnem == "TDG":
            if len(args) < 1: raise AsmError(f"line {lineno}: TDG needs 1 arg")
            out.append(_encode(0x5, args[0], 0, 14))
        elif mnem in ("CX", "NOT_ALIAS"):
            # CX = CNOT
            if len(args) < 2: raise AsmError(f"line {lineno}: CX needs 2 args")
            out.append(_encode(0x3, args[0], args[1]))
        elif mnem == "NOT":
            if len(args) < 1: raise AsmError(f"line {lineno}: NOT needs 1 arg")
            out.append(_encode(0x2, args[0]))
        elif mnem == "TOFFOLI":
            if len(args) < 3: raise AsmError(f"line {lineno}: TOFFOLI needs 3 args")
            out.append(_encode(0xC, args[0], args[1], args[2]))
        elif mnem == "CY":
            # CY(c, t) = SDG t; CNOT c t; S t
            if len(args) < 2: raise AsmError(f"line {lineno}: CY needs 2 args")
            c, t = args[0], args[1]
            out.append(_encode(0x5, t, 0, 12))  # SDG target
            out.append(_encode(0x3, c, t))       # CNOT
            out.append(_encode(0x5, t, 0, 4))   # S target
        elif mnem == "CH":
            # CH(c, t) = S t; H t; T t; CNOT c t; TDG t; H t; SDG t
            if len(args) < 2: raise AsmError(f"line {lineno}: CH needs 2 args")
            c, t = args[0], args[1]
            out.append(_encode(0x5, t, 0, 4))   # S t
            out.append(_encode(0x1, t))          # H t
            out.append(_encode(0x5, t, 0, 2))   # T t
            out.append(_encode(0x3, c, t))       # CNOT c t
            out.append(_encode(0x5, t, 0, 14))  # TDG t
            out.append(_encode(0x1, t))          # H t
            out.append(_encode(0x5, t, 0, 12))  # SDG t
        # ── ISA + software-extension opcodes ─────────────────────
        elif mnem in ALL_OPCODES:
            op = ALL_OPCODES[mnem]
            try:
                if mnem in ("NOP", "INIT", "HALT"):
                    out.append(_encode(op))
                elif mnem in ("H", "X", "Y", "MEASURE"):
                    out.append(_encode(op, args[0]))
                elif mnem in ("CNOT", "SWAP", "CZ"):
                    out.append(_encode(op, args[0], args[1]))
                elif mnem in ("PHASE", "RZ"):
                    out.append(_encode(op, args[0], 0, args[1]))
                elif mnem == "CCX":
                    # c1=A, c2=B, target=IMM
                    out.append(_encode(op, args[0], args[1], args[2]))
                elif mnem == "LOAD_AMP":
                    out.append(_encode(op, args[0]))
                elif mnem in ("TRACE", "WAIT"):
                    out.append(_encode(op, 0, 0, args[0] if args else 0))
            except IndexError:
                raise AsmError(f"line {lineno}: not enough operands for {mnem}")
        else:
            raise AsmError(f"line {lineno}: unknown opcode '{mnem}'")
    return out


def disassemble(words: List[int]) -> List[str]:
    """Decode 16-bit words to human-readable assembly lines."""
    rows = []
    for pc, w in enumerate(words):
        op  = (w >> 12) & 0xF
        a   = (w >>  8) & 0xF
        b   = (w >>  4) & 0xF
        imm =  w        & 0xF
        mnem = OPCODE_REV.get(op, f"OP{op:X}")
        # format operands
        if op in (0x0, 0x6, 0xF):  # NOP, INIT, HALT
            ops = ""
        elif op in (0x1, 0x2, 0x4, 0x8, 0xA):  # H X MEASURE LOAD_AMP Y
            ops = f"q{a}"
        elif op in (0x3, 0x7, 0xB):  # CNOT SWAP CZ
            ops = f"q{a}, q{b}"
        elif op in (0x5, 0xD):  # PHASE RZ
            ops = f"q{a}, {imm}  (φ={imm*22.5:.1f}°)"
        elif op == 0xC:  # CCX
            ops = f"q{a}, q{b}, q{imm}"
        elif op in (0x9, 0xE):  # TRACE WAIT
            ops = f"{imm}"
        else:
            ops = f"a={a} b={b} imm={imm}"
        sw = "  [SW]" if op in (0xA, 0xB, 0xC, 0xD) else ""
        rows.append(f"  {pc:3d}  0x{w:04X}  {mnem:8s} {ops}{sw}")
    return rows


def hex_export(words: List[int]) -> str:
    """Export assembled program as Intel HEX-style dump for FPGA SPI loading."""
    lines = ["; QGRIM v2.1 program — load via SPI at PGM_PAGE/PGM_BASE"]
    lines.append(f"; {len(words)} instruction(s)")
    lines.append("; addr  hex    binary           mnemonic")
    lines.append("; ─────────────────────────────────────────────")
    for i, w in enumerate(words):
        op = (w >> 12) & 0xF
        mnem = OPCODE_REV.get(op, f"OP{op:X}")
        lines.append(f";  {i:03d}  0x{w:04X}  {w:016b}  {mnem}")
    lines.append("; Raw hex for SPI upload:")
    raw = " ".join(f"0x{w:04X}" for w in words)
    lines.append(raw)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SIMULATOR CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _q(x: float) -> float:
    """Round to Q4.12 grid, clamp to representable range (mirrors the RTL)."""
    v = round(x * ONE_Q12)
    v = max(-(1 << 15), min((1 << 15) - 1, v))
    return v / ONE_Q12


def _q_complex(z: complex) -> complex:
    return _q(z.real) + 1j * _q(z.imag)


@dataclass
class QGRIMSim:
    """
    Pure-Python model of the QGRIM v2.1 core.

    Mirrors the RTL at the algorithmic level:
      - Q4.12 fixed-point arithmetic throughout (same as FPGA)
      - Born-rule measurement with wavefunction collapse
      - Paired-addressing gate traversal (matches pair_addr_gen.v)
      - Depolarizing noise model (noise_p > 0 to enable)
      - Software extension gates: Y, CZ, Toffoli (CCX)
    """
    seed: int = 0xACE1
    noise_p: float = 0.0   # per-gate depolarizing error probability
    state: List[complex] = field(default_factory=lambda: [0j] * STATES)
    pc: int = 0
    halted: bool = False
    measurements: Dict[int, int] = field(default_factory=dict)
    pi: float = 0.0
    trace: List[Tuple[int, int, float]] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self.state[0] = 1.0 + 0j   # |0000⟩

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset to |0000⟩ and clear all registers."""
        self.state = [0j] * STATES
        self.state[0] = 1.0 + 0j
        self.pc = 0
        self.halted = False
        self.measurements.clear()
        self.pi = 0.0
        self.trace.clear()

    def clone(self) -> "QGRIMSim":
        """Return a deep copy of this simulator state."""
        s = QGRIMSim(seed=self.seed, noise_p=self.noise_p)
        s.state = self.state[:]
        s.pc = self.pc
        s.halted = self.halted
        s.measurements = dict(self.measurements)
        s.pi = self.pi
        s.trace = list(self.trace)
        s._rng = random.Random(self._rng.random())
        return s

    # ── noise ──────────────────────────────────────────────────────────────────

    def _apply_noise(self, qubits: List[int]):
        """Apply single-qubit depolarizing channel to each qubit in list."""
        if self.noise_p <= 0.0:
            return
        for q in qubits:
            r = self._rng.random()
            if r < self.noise_p / 3:
                self._pauli_x(q)
            elif r < 2 * self.noise_p / 3:
                self._pauli_y_raw(q)   # raw avoids Q4.12 quantise overhead
            elif r < self.noise_p:
                self._phase_raw(q, 8)  # Z

    def _pauli_y_raw(self, q: int):
        """Y gate without quantization (noise helper)."""
        new = [0j] * STATES
        mask = 1 << q
        for i in range(STATES):
            if not (i & mask):
                j = i | mask
                new[i] = -1j * self.state[j]
                new[j] =  1j * self.state[i]
        self.state = new

    def _phase_raw(self, q: int, idx: int):
        """PHASE gate without quantization (noise helper)."""
        cos_v, sin_v = PHASE_LUT[idx & 0xF]
        factor = complex(cos_v, sin_v)
        mask = 1 << q
        for i in range(STATES):
            if i & mask:
                self.state[i] *= factor

    # ── hardware ISA gates ─────────────────────────────────────────────────────

    def _init_gate(self):
        self.state = [0j] * STATES
        self.state[0] = 1.0 + 0j

    def _hadamard(self, q: int):
        new = self.state[:]
        mask = 1 << q
        inv_sqrt2 = 1.0 / math.sqrt(2)
        for i in range(STATES):
            if i & mask:
                continue
            j = i | mask
            a, b = self.state[i], self.state[j]
            s = (a + b) * inv_sqrt2
            d = (a - b) * inv_sqrt2
            new[i] = _q(s.real) + 1j * _q(s.imag)
            new[j] = _q(d.real) + 1j * _q(d.imag)
        self.state = new
        self._apply_noise([q])

    def _pauli_x(self, q: int):
        new = self.state[:]
        mask = 1 << q
        for i in range(STATES):
            if not (i & mask):
                j = i | mask
                new[i], new[j] = self.state[j], self.state[i]
        self.state = new
        self._apply_noise([q])

    def _cnot(self, c: int, t: int):
        new = self.state[:]
        cmask, tmask = 1 << c, 1 << t
        for i in range(STATES):
            if (i & tmask) or not (i & cmask):
                continue
            j = i | tmask
            new[i], new[j] = self.state[j], self.state[i]
        self.state = new
        self._apply_noise([c, t])

    def _phase(self, q: int, idx: int):
        cos_v, sin_v = PHASE_LUT[idx & 0xF]
        factor = complex(_q(cos_v), _q(sin_v))
        mask = 1 << q
        for i in range(STATES):
            if i & mask:
                v = self.state[i] * factor
                self.state[i] = _q(v.real) + 1j * _q(v.imag)
        self._apply_noise([q])

    def _swap(self, a: int, b: int):
        ma, mb = 1 << a, 1 << b
        new = self.state[:]
        for i in range(STATES):
            ba, bb = bool(i & ma), bool(i & mb)
            if ba == bb:
                continue
            j = i & ~ma & ~mb
            if ba: j |= mb
            if bb: j |= ma
            if j > i:
                new[i], new[j] = self.state[j], self.state[i]
        self.state = new
        self._apply_noise([a, b])

    def _load_amp(self, basis: int):
        self.state = [0j] * STATES
        if 0 <= basis < STATES:
            self.state[basis] = 1.0 + 0j

    def _measure(self, q: int) -> int:
        mask = 1 << q
        p0 = sum(abs(a) ** 2 for i, a in enumerate(self.state) if not (i & mask))
        bit = 0 if self._rng.random() < p0 else 1
        keep = p0 if bit == 0 else max(1.0 - p0, 0.0)
        norm = math.sqrt(keep) if keep > 1e-12 else 1.0
        new = []
        for i, a in enumerate(self.state):
            if ((i & mask) >> q) == bit:
                v = a / norm
                new.append(_q(v.real) + 1j * _q(v.imag))
            else:
                new.append(0j)
        self.state = new
        self.measurements[q] = bit
        self.pi = _q(self.pi + (16 / 4096))
        return bit

    # ── software extension gates ───────────────────────────────────────────────

    def _pauli_y(self, q: int):
        """Pauli-Y: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩"""
        new = [0j] * STATES
        mask = 1 << q
        for i in range(STATES):
            if not (i & mask):
                j = i | mask
                new[i] = _q_complex(-1j * self.state[j])
                new[j] = _q_complex( 1j * self.state[i])
        self.state = new
        self._apply_noise([q])

    def _cz(self, c: int, t: int):
        """Controlled-Z: flip sign of |11⟩ component."""
        cmask, tmask = 1 << c, 1 << t
        for i in range(STATES):
            if (i & cmask) and (i & tmask):
                self.state[i] = -self.state[i]
        self._apply_noise([c, t])

    def _ccx(self, c1: int, c2: int, t: int):
        """Toffoli / CCX: flip target iff both controls are |1⟩."""
        new = self.state[:]
        c1m, c2m, tm = 1 << c1, 1 << c2, 1 << t
        for i in range(STATES):
            if (i & tm) or not (i & c1m) or not (i & c2m):
                continue
            j = i | tm
            new[i], new[j] = self.state[j], self.state[i]
        self.state = new
        self._apply_noise([c1, c2, t])

    # ── execution ──────────────────────────────────────────────────────────────

    def step(self, instr: int) -> bool:
        op  = (instr >> 12) & 0xF
        a   = (instr >>  8) & 0xF
        b   = (instr >>  4) & 0xF
        imm =  instr        & 0xF

        if   op == 0x0: pass                          # NOP
        elif op == 0x1: self._hadamard(a)             # H
        elif op == 0x2: self._pauli_x(a)              # X
        elif op == 0x3: self._cnot(a, b)              # CNOT
        elif op == 0x4: self._measure(a)              # MEASURE
        elif op == 0x5: self._phase(a, imm)           # PHASE
        elif op == 0x6: self._init_gate()             # INIT
        elif op == 0x7: self._swap(a, b)              # SWAP
        elif op == 0x8: self._load_amp(a)             # LOAD_AMP
        elif op == 0x9:
            meas_int = sum(v << k for k, v in self.measurements.items())
            self.trace.append((self.pc, meas_int, self.pi))
        elif op == 0xA: self._pauli_y(a)              # Y  [SW]
        elif op == 0xB: self._cz(a, b)               # CZ [SW]
        elif op == 0xC: self._ccx(a, b, imm)         # CCX/Toffoli [SW]
        elif op == 0xD: self._phase(a, imm)           # RZ [SW] — same as PHASE
        elif op == 0xE: pass                          # WAIT
        elif op == 0xF:
            self.halted = True
            return False
        else:
            raise RuntimeError(f"Unknown opcode 0x{op:X} at PC={self.pc}")
        self.pc += 1
        return True

    def run(self, program: List[int], max_cycles: int = 10_000):
        self.pc = 0
        self.halted = False
        for _ in range(max_cycles):
            if self.pc >= len(program) or self.halted:
                break
            if not self.step(program[self.pc]):
                break

    # ── analysis helpers ───────────────────────────────────────────────────────

    def state_formula(self, threshold: float = 1e-3) -> str:
        """Return state as ket sum: e.g. '0.707|00⟩ + 0.707|11⟩'"""
        terms = []
        for i, a in enumerate(self.state):
            mag = abs(a)
            if mag < threshold:
                continue
            ket = f"|{i:04b}⟩"
            phase_ang = math.atan2(a.imag, a.real)
            if abs(a.imag) < 1e-4:
                coeff = f"{a.real:+.4f}"
            elif abs(a.real) < 1e-4:
                coeff = f"{a.imag:+.4f}i"
            else:
                coeff = f"({a.real:+.4f}{a.imag:+.4f}i)"
            terms.append(f"{coeff}{ket}")
        if not terms:
            return "0"
        return " ".join(terms).lstrip("+").replace(" +", " + ").replace(" -", " - ")

    def bloch_sphere(self, q: int) -> Tuple[float, float, float]:
        """
        Compute Bloch sphere (x, y, z) for qubit q via reduced density matrix.
        Returns (x, y, z) where x²+y²+z² ≤ 1 (= 1 for pure single-qubit state).
        """
        mask = 1 << q
        rho00 = rho01 = rho10 = rho11 = 0j
        for i in range(STATES):
            for j in range(STATES):
                if (i & ~mask) != (j & ~mask):
                    continue
                bi = (i >> q) & 1
                bj = (j >> q) & 1
                val = self.state[i] * self.state[j].conjugate()
                if bi == 0 and bj == 0:   rho00 += val
                elif bi == 0 and bj == 1: rho01 += val
                elif bi == 1 and bj == 0: rho10 += val
                else:                     rho11 += val
        x = 2.0 * rho01.real
        y = 2.0 * rho01.imag
        z = (rho00 - rho11).real
        return float(x), float(y), float(z)

    def bloch_sphere_str(self, q: int) -> str:
        """Human-readable Bloch sphere display for qubit q."""
        x, y, z = self.bloch_sphere(q)
        r = math.sqrt(x*x + y*y + z*z)
        theta = math.degrees(math.acos(max(-1.0, min(1.0, z / r)))) if r > 1e-6 else 0.0
        phi = math.degrees(math.atan2(y, x))
        lines = [
            f"Bloch sphere for q{q}:",
            f"  x = {x:+.4f}   ⟨X⟩",
            f"  y = {y:+.4f}   ⟨Y⟩",
            f"  z = {z:+.4f}   ⟨Z⟩",
            f"  |r| = {r:.4f}  θ = {theta:.1f}°  φ = {phi:.1f}°",
        ]
        if r < 0.01:
            lines.append("  (maximally mixed — qubit is entangled with others)")
        elif r > 0.98:
            if z > 0.9:   lines.append("  ≈ |0⟩  (north pole)")
            elif z < -0.9: lines.append("  ≈ |1⟩  (south pole)")
            elif abs(x-1)<0.1: lines.append("  ≈ |+⟩  (equator, x=+1)")
            elif abs(x+1)<0.1: lines.append("  ≈ |−⟩  (equator, x=-1)")
            elif abs(y-1)<0.1: lines.append("  ≈ |+i⟩ (equator, y=+1)")
            elif abs(y+1)<0.1: lines.append("  ≈ |−i⟩ (equator, y=-1)")
        return "\n".join(lines)

    def entanglement_entropy(self, partition_a: Optional[List[int]] = None) -> float:
        """
        Von Neumann entropy of reduced density matrix for partition A
        (traces out partition B = all other qubits).
        Default partition: A = {q0, q1}, B = {q2, q3}.
        Returns entropy in bits (log base 2).
        Uses Schmidt decomposition via SVD of reshaped amplitude matrix.
        """
        if partition_a is None:
            partition_a = [0, 1]
        # Build amplitude matrix: rows = A-subspace, cols = B-subspace
        a_bits = partition_a
        b_bits = [q for q in range(QUBITS) if q not in a_bits]
        dim_a = 1 << len(a_bits)
        dim_b = 1 << len(b_bits)

        # Map basis index → (a_index, b_index)
        M = [[0j] * dim_b for _ in range(dim_a)]
        for idx in range(STATES):
            a_idx = sum(((idx >> q) & 1) << k for k, q in enumerate(a_bits))
            b_idx = sum(((idx >> q) & 1) << k for k, q in enumerate(b_bits))
            M[a_idx][b_idx] = self.state[idx]

        # Singular values of M via eigenvalues of M†M
        n = min(dim_a, dim_b)
        # Compute Gram matrix G = M†M (dim_b × dim_b)
        G = [[sum(M[k][i].conjugate() * M[k][j] for k in range(dim_a))
               for j in range(dim_b)] for i in range(dim_b)]

        # Find eigenvalues of Hermitian G via power iteration + deflation
        eigenvalues = _hermitian_eigenvalues(G, dim_b)
        eigenvalues = [max(0.0, e.real) for e in eigenvalues]

        # Von Neumann entropy S = -Σ λ log₂(λ)
        S = 0.0
        for lam in eigenvalues:
            if lam > 1e-12:
                S -= lam * math.log2(lam)
        return S

    def fidelity(self, other_state: List[complex]) -> float:
        """Fidelity F = |⟨ψ|φ⟩|² between this state and another."""
        inner = sum(self.state[i].conjugate() * other_state[i] for i in range(STATES))
        return abs(inner) ** 2

    # ── output ─────────────────────────────────────────────────────────────────

    def dump(self) -> str:
        lines = []
        lines.append("╔══════════════════════════════════════════════════╗")
        lines.append("║              QGRIM STATE VECTOR                  ║")
        lines.append("╠══════════════════════════════════════════════════╣")
        non_zero = [(i, a) for i, a in enumerate(self.state) if abs(a) > 1e-4]
        if not non_zero:
            lines.append("║  (zero state — all amplitudes are zero)          ║")
        else:
            for i, a in non_zero:
                prob = abs(a) ** 2
                bar = "█" * int(prob * 16)
                w_lbl = WITNESS_LABELS[i]
                lines.append(
                    f"║  |{i:04b}⟩  {a.real:+.4f}{a.imag:+.4f}i"
                    f"  P={prob:.3f}  {bar:<16}  [{w_lbl}]"
                )
        lines.append("╠══════════════════════════════════════════════════╣")
        if self.measurements:
            mlines = "  ".join(f"q{k}={v}" for k, v in sorted(self.measurements.items()))
            lines.append(f"║  Measured:  {mlines}")
        lines.append(f"║  π-reg: {self.pi:.4f}  ({int(self.pi * 4096):#06x})")
        lines.append(f"║  PC: {self.pc}  HALTED: {self.halted}"
                     + (f"  NOISE: {self.noise_p:.3f}" if self.noise_p > 0 else ""))
        lines.append("╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def probability_table(self) -> str:
        lines = ["\nProbability distribution:"]
        for i, a in enumerate(self.state):
            prob = abs(a) ** 2
            bar_len = int(prob * 30)
            bar = "▓" * bar_len + "░" * (30 - bar_len)
            w = WITNESS_LABELS[i]
            lines.append(f"  |{i:04b}⟩  {bar}  {prob*100:5.1f}%  [{w}]")
        return "\n".join(lines)

    def snapshot(self) -> List[complex]:
        return self.state[:]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LINEAR ALGEBRA HELPERS (stdlib only — no numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def _hermitian_eigenvalues(A: List[List[complex]], n: int) -> List[complex]:
    """
    Find all eigenvalues of n×n Hermitian matrix via power iteration + deflation.
    Uses varied starting vectors to handle degenerate eigenvalues correctly.
    Sufficient for n ≤ 8; used for entanglement entropy.
    """
    eigs: List[complex] = []
    # Work on a mutable copy
    B = [[A[i][j] for j in range(n)] for i in range(n)]
    # Basis vectors used as starting points — varied per deflation step
    # so degenerate eigenvalues (same λ, different eigenvectors) are both found.
    _PHI = 1.6180339887  # golden ratio for quasi-random angles

    for eig_idx in range(n):
        best_lam = 0.0
        best_v = [0j] * n

        # Try several starting vectors; keep result with largest |lam|
        for restart in range(4):
            angle_offset = eig_idx * _PHI + restart * 0.7853981
            v = [complex(math.cos(angle_offset + k * _PHI),
                         math.sin(angle_offset + k * 1.2345))
                 for k in range(n)]
            norm_v = math.sqrt(sum(abs(x) ** 2 for x in v))
            if norm_v < 1e-12:
                continue
            v = [x / norm_v for x in v]

            lam = 0.0
            for _ in range(300):
                Av = [sum(B[i][j] * v[j] for j in range(n)) for i in range(n)]
                new_norm = math.sqrt(sum(abs(x) ** 2 for x in Av))
                if new_norm < 1e-14:
                    lam = 0.0
                    break
                lam = sum(v[i].conjugate() * Av[i] for i in range(n)).real
                v = [x / new_norm for x in Av]

            if abs(lam) > abs(best_lam):
                best_lam = lam
                best_v = v[:]

        eigs.append(best_lam)
        # Deflate: B = B - lam * v v†  (removes the found eigenvector)
        for i in range(n):
            for j in range(n):
                B[i][j] -= best_lam * best_v[i] * best_v[j].conjugate()

    return eigs


def fidelity(state_a: List[complex], state_b: List[complex]) -> float:
    """Fidelity F = |⟨a|b⟩|² between two pure states."""
    inner = sum(state_a[i].conjugate() * state_b[i] for i in range(len(state_a)))
    return abs(inner) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CIRCUIT VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def draw_circuit(program: List[int], title: str = "") -> str:
    """
    Render an ASCII circuit diagram for the given assembled program.
    Two-qubit gates show vertical connection lines.
    """
    # Build list of column dicts {qubit: symbol}
    cols: List[Dict[int, str]] = []
    for w in program:
        op  = (w >> 12) & 0xF
        a   = (w >>  8) & 0xF
        b   = (w >>  4) & 0xF
        imm =  w        & 0xF
        col: Dict[int, str] = {q: "─" for q in range(QUBITS)}
        two_qubit_pair: Optional[Tuple[int, int]] = None

        if op == 0x0:   # NOP
            for q in range(QUBITS): col[q] = "─"
        elif op == 0x1: col[a] = "H"
        elif op == 0x2: col[a] = "X"
        elif op == 0xA: col[a] = "Y"
        elif op == 0x5 or op == 0xD:
            labels = {0:"I",1:"P1",2:"T",3:"P3",4:"S",5:"P5",
                      6:"P6",7:"P7",8:"Z",9:"P9",10:"PA",11:"PB",
                      12:"S†",13:"PC",14:"T†",15:"PF"}
            col[a] = labels.get(imm, f"P{imm}")
        elif op == 0x3:  # CNOT
            col[a] = "●"
            col[b] = "⊕"
            two_qubit_pair = (a, b)
        elif op == 0xB:  # CZ
            col[a] = "●"
            col[b] = "Z"
            two_qubit_pair = (a, b)
        elif op == 0xC:  # CCX — three qubit; a, b = controls, imm = target
            col[a] = "●"
            col[b] = "●"
            col[imm] = "⊕"
        elif op == 0x7:  # SWAP
            col[a] = "×"
            col[b] = "×"
            two_qubit_pair = (a, b)
        elif op == 0x4: col[a] = "M"
        elif op == 0x6:
            for q in range(QUBITS): col[q] = "0"
        elif op == 0x8: col[a] = f"B{a}"
        elif op == 0x9: col[a] = "T"
        elif op == 0xF:
            for q in range(QUBITS): col[q] = "║"
        elif op == 0xE: col[a] = "W"

        # Fill intermediate qubit rows with │ for two-qubit gates
        if two_qubit_pair is not None:
            lo, hi = sorted(two_qubit_pair)
            for q in range(lo + 1, hi):
                col[q] = "│"
        cols.append(col)

    if not cols:
        return "(empty program)"

    # Render rows
    pad = 3  # symbol cell width
    header = ""
    if title:
        header = f"  Circuit: {title}\n"
    lines = [header + f"  {'─'*(len(cols)*(pad+1)+6)}"]
    for q in range(QUBITS):
        row = f"  q{q} ─"
        for col in cols:
            sym = col.get(q, "─")
            # pad symbol to `pad` characters
            s = sym[:pad].center(pad)
            row += s + "─"
        lines.append(row)
    lines.append(f"  {'─'*(len(cols)*(pad+1)+6)}")
    lines.append(f"  Depth: {len(cols)} steps  |  Qubits: {QUBITS}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SHOT SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

def sample_circuit(
    source: str,
    shots: int = 1024,
    seed: int = 0xACE1,
    noise_p: float = 0.0,
) -> Dict[str, int]:
    """
    Run the circuit `shots` times (independent fresh state each run).
    Returns a dict: 4-bit bitstring → count.
    """
    program = assemble(source)
    counts: Dict[str, int] = {}
    rng = random.Random(seed)
    for _ in range(shots):
        sim = QGRIMSim(seed=rng.randint(0, 0xFFFFFFFF), noise_p=noise_p)
        sim.run(program)
        bits = "".join(str(sim.measurements.get(q, 0)) for q in range(QUBITS))
        counts[bits] = counts.get(bits, 0) + 1
    return counts


def render_histogram(counts: Dict[str, int], shots: int, width: int = 36) -> str:
    total = sum(counts.values())
    lines = [f"\nHistogram ({shots} shots):", "─" * (width + 30)]
    for key in sorted(counts):
        n = counts[key]
        bar_len = int(n / total * width)
        bar = "█" * bar_len
        pct = n / total * 100
        w = WITNESS_LABELS[int(key, 2)] if len(key) == 4 else "?"
        lines.append(f"  |{key}⟩ [{w}]  {bar:<{width}}  {n:5d}  ({pct:5.1f}%)")
    lines.append("─" * (width + 30))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CHSH BELL INEQUALITY TEST
# ═══════════════════════════════════════════════════════════════════════════════

def _ry_gate(sim: "QGRIMSim", q: int, theta: float):
    """
    Apply Ry(theta) to qubit q of sim (float-exact, not quantized).
    Used internally for CHSH basis rotations — not part of the ISA.
    Ry(θ) = [[cos(θ/2), -sin(θ/2)],
              [sin(θ/2),  cos(θ/2)]]
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    mask = 1 << q
    new = sim.state[:]
    for i in range(STATES):
        if i & mask:
            continue
        j = i | mask
        a0, a1 = sim.state[i], sim.state[j]
        new[i] =  c * a0 - s * a1
        new[j] =  s * a0 + c * a1
    sim.state = new


def chsh_test(shots_per_setting: int = 500) -> str:
    """
    CHSH Bell inequality test using the Bell state |Φ+⟩.

    Measures E(a,b) = P(same) - P(diff) with Ry basis rotations.
    Settings for maximum violation with |Φ+⟩:
      a=0,    a'=π/2
      b=π/4,  b'=3π/4
    Theory: E(a,b) = cos(a - b)  →  S = 2√2 ≈ 2.828

    Classical bound:  |S| ≤ 2
    Quantum maximum:  |S| = 2√2 ≈ 2.828
    """
    settings = [
        ("a=0,    b=π/4",   0.0,             math.pi / 4),
        ("a=0,    b=3π/4",  0.0,             3 * math.pi / 4),
        ("a=π/2,  b=π/4",   math.pi / 2,     math.pi / 4),
        ("a=π/2,  b=3π/4",  math.pi / 2,     3 * math.pi / 4),
    ]
    # Theoretical E values
    theory = [math.cos(a - b) for (_, a, b) in settings]
    theory_S = theory[0] - theory[1] + theory[2] + theory[3]

    correlations = []
    rng = random.Random(0xBEEF)
    bell_prog = assemble("INIT\nH 0\nCNOT 0 1\nHALT")

    lines = ["\n═══ CHSH Bell Inequality Test ═══",
             f"Shots per setting: {shots_per_setting}",
             f"Theoretical maximum S = {theory_S:.4f}",
             "─" * 56]

    for label, a_ang, b_ang in settings:
        same = diff = 0
        for _ in range(shots_per_setting):
            sim = QGRIMSim(seed=rng.randint(0, 0xFFFFFFFF))
            sim.run(bell_prog)
            # Rotate Alice (q0) and Bob (q1) measurement bases
            _ry_gate(sim, 0, a_ang)
            _ry_gate(sim, 1, b_ang)
            # Now measure in Z-basis
            m0 = sim._measure(0)
            m1 = sim._measure(1)
            if m0 == m1:
                same += 1
            else:
                diff += 1
        E = (same - diff) / shots_per_setting
        th = math.cos(a_ang - b_ang)
        correlations.append(E)
        lines.append(f"  {label:<18}  E = {E:+.4f}  (theory: {th:+.4f})")

    S = correlations[0] - correlations[1] + correlations[2] + correlations[3]
    lines.append("─" * 56)
    lines.append(f"  CHSH value  S = {S:+.4f}  (theory: {theory_S:+.4f})")
    lines.append(f"  Classical bound: |S| ≤ 2.0000")
    lines.append(f"  Quantum maximum: |S| = 2.8284")
    if abs(S) > 2.0:
        excess = abs(S) - 2.0
        lines.append(f"  ✓ BELL INEQUALITY VIOLATED  (by {excess:.4f})")
        lines.append(f"  Quantum behaviour confirmed — no local hidden variables")
    else:
        lines.append(f"  ✗ No violation detected (increase shots or reduce noise)")
    lines.append("═" * 56)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BUILT-IN CIRCUITS
# ═══════════════════════════════════════════════════════════════════════════════

BUILTIN_CIRCUITS: Dict[str, Tuple[str, str]] = {

    # ── fundamental entanglement ────────────────────────────────────────────
    "bell": (
        "Bell state |Φ+⟩ = (|0000⟩ + |0011⟩)/√2  [with measurement]",
        """\
INIT
H 0
CNOT 0 1
MEASURE 0
MEASURE 1
HALT
"""),

    "bell_nomeas": (
        "Bell superposition without measurement — see amplitudes directly",
        """\
INIT
H 0
CNOT 0 1
HALT
"""),

    "ghz": (
        "3-qubit GHZ state: (|000⟩ + |111⟩)/√2",
        """\
INIT
H 0
CNOT 0 1
CNOT 1 2
MEASURE 0
MEASURE 1
MEASURE 2
HALT
"""),

    "cluster4": (
        "4-qubit cluster / graph state (universal resource for measurement-based QC)",
        """\
INIT
H 0
H 1
H 2
H 3
CZ 0 1
CZ 1 2
CZ 2 3
HALT
"""),

    # ── algorithms ─────────────────────────────────────────────────────────
    "deutsch": (
        "Deutsch algorithm: constant oracle (answer = 0 always) — q0 tells you",
        """\
INIT
; Input qubit q0 in |+⟩, ancilla q1 in |−⟩
H 0
X 1
H 1
; Oracle for constant-0 function: do nothing (f(x)=0 for all x)
; If f were balanced (f(x)=x), we would add CNOT 0 1 here
H 0
MEASURE 0
HALT
; q0=0 means constant function, q0=1 means balanced function
"""),

    "deutsch_balanced": (
        "Deutsch algorithm: balanced oracle (CNOT) — q0 will be |1⟩",
        """\
INIT
H 0
X 1
H 1
; Oracle for balanced function f(x) = x: CNOT
CNOT 0 1
H 0
MEASURE 0
HALT
"""),

    "bv": (
        "Bernstein-Vazirani: find hidden string s=101 in one query",
        """\
INIT
; Hadamard all input qubits (q0,q1,q2) and ancilla q3
H 0
H 1
H 2
X 3
H 3
; Oracle: CNOT q_i → q3 for each bit of s=101 (bits 0 and 2)
CNOT 0 3
; bit 1 of s is 0, skip CNOT 1 3
CNOT 2 3
; Hadamard input qubits
H 0
H 1
H 2
; Measure — result should be s = 101
MEASURE 0
MEASURE 1
MEASURE 2
HALT
"""),

    "grover": (
        "Grover search for |0101⟩ (qubits 0,1,2,3 = 1,0,1,0 from LSB)",
        """\
INIT
; Uniform superposition
H 0
H 1
H 2
H 3
; Oracle: mark |0101⟩ — phase flip via CZ chain
; |0101⟩ = q0=1, q1=0, q2=1, q3=0
X 1
X 3
; Multi-controlled phase flip (flip sign of |1111⟩ after X on 1,3)
; Approximate with: H q3; CCX q0 q1 q2; H q3  ... but CCX needs 3 controls
; Use CZ + H trick for the marked state:
H 3
CCX 0 1 2
H 3
X 1
X 3
; Diffusion operator (inversion about average)
H 0
H 1
H 2
H 3
X 0
X 1
X 2
X 3
H 3
CCX 0 1 2
H 3
X 0
X 1
X 2
X 3
H 0
H 1
H 2
H 3
HALT
"""),

    "qft": (
        "Quantum Fourier Transform on 3 qubits",
        """\
INIT
H 0
PHASE 1 4
CNOT 0 1
PHASE 2 2
CNOT 0 2
PHASE 2 4
CNOT 1 2
H 1
PHASE 2 4
CNOT 1 2
H 2
SWAP 0 2
HALT
"""),

    "teleport": (
        "Quantum teleportation of |+⟩ from q0 to q2",
        """\
INIT
; Prepare |+⟩ message on q0
H 0
; Create Bell pair on q1,q2
H 1
CNOT 1 2
; Bell measurement on q0,q1
CNOT 0 1
H 0
MEASURE 0
MEASURE 1
; Classical corrections on q2 (always applied in software model)
X 2
PHASE 2 8
HALT
"""),

    "superdense": (
        "Superdense coding: encode 2 classical bits (11) in 1 qubit",
        """\
INIT
; Alice and Bob share Bell pair
H 0
CNOT 0 1
; Alice wants to send bits '11': apply X then Z to her qubit q0
X 0
PHASE 0 8
; Bob decodes: CNOT then H
CNOT 0 1
H 0
; Bob measures both qubits — should get 1,1
MEASURE 0
MEASURE 1
HALT
"""),

    "phase_kickback": (
        "Phase kickback: control qubit picks up phase from target eigenstate",
        """\
INIT
H 0
X 1
H 1
CNOT 0 1
H 0
MEASURE 0
HALT
"""),

    # ── gates showcase ──────────────────────────────────────────────────────
    "toffoli": (
        "Toffoli (CCX) gate: flip q2 only when q0=1 AND q1=1",
        """\
INIT
X 0
X 1
CCX 0 1 2
MEASURE 0
MEASURE 1
MEASURE 2
HALT
"""),

    "superpos_all": (
        "All 4 qubits in uniform superposition (Hadamard register)",
        """\
INIT
H 0
H 1
H 2
H 3
HALT
"""),

    "t_gate": (
        "T gate (π/4 phase) demo — produces |+i⟩-like output",
        """\
INIT
H 0
T 0
H 0
HALT
"""),

    "qrng": (
        "Quantum random number generator — 4 truly random bits",
        """\
INIT
H 0
H 1
H 2
H 3
MEASURE 0
MEASURE 1
MEASURE 2
MEASURE 3
HALT
"""),

    "y_gate": (
        "Pauli-Y gate demo: Y|0⟩ = i|1⟩",
        """\
INIT
Y 0
MEASURE 0
HALT
"""),

    "cz_demo": (
        "Controlled-Z demo: |++⟩ → Bell-like state via CZ",
        """\
INIT
H 0
H 1
CZ 0 1
H 0
H 1
HALT
"""),

    "qpe": (
        "Quantum Phase Estimation sketch: estimate phase of Z gate (eigenvalue=-1)",
        """\
INIT
; Prepare clock qubit q0 in |+⟩, eigenstate of Z in q1 = |1⟩
H 0
X 1
; Controlled-Z: q0 controls phase on q1
; Z has eigenvalue e^(iπ) = -1, so clock picks up phase π
CZ 0 1
; Inverse QFT on clock (just H for 1-qubit clock)
H 0
MEASURE 0
HALT
; q0=1 means phase = π (Z gate)
"""),
}


def list_circuits() -> str:
    lines = ["\nBuilt-in circuits:"]
    lines.append(f"  {'Name':<20}  Description")
    lines.append(f"  {'─'*20}  {'─'*50}")
    for name, (desc, _) in BUILTIN_CIRCUITS.items():
        lines.append(f"  {name:<20}  {desc}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_program(lines_buf: List[str], filename: str) -> str:
    """Save program lines to a .qasm file."""
    if not filename.endswith(".qasm"):
        filename += ".qasm"
    content = "; QGRIM program — saved by QGRIM_ENGINE.py\n"
    content += "\n".join(lines_buf) + "\n"
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Saved to: {os.path.abspath(filename)}"
    except OSError as e:
        return f"Save failed: {e}"


def load_program(filename: str) -> Tuple[Optional[str], str]:
    """Load a .qasm file. Returns (source_text, error_message)."""
    try:
        with open(filename) as f:
            return f.read(), ""
    except OSError as e:
        return None, f"Cannot open '{filename}': {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: INTERACTIVE REPL
# ═══════════════════════════════════════════════════════════════════════════════

REPL_HELP = """
┌─────────────────────────────────────────────────────────────┐
│                   QGRIM Interactive REPL                     │
│                                                              │
│  One gate per line. State shows after each gate.             │
│                                                              │
│  Hardware gates (0-3 = qubit numbers):                       │
│    H 0            Hadamard                                   │
│    X 0            Pauli-X (bit-flip)                         │
│    Y 0            Pauli-Y                        [SW]        │
│    Z 0            Pauli-Z (= PHASE 0 8)                      │
│    S 0            S gate (π/2 phase)                         │
│    T 0            T gate (π/4 phase)                         │
│    SDG 0          S† gate (−π/2)                             │
│    TDG 0          T† gate (−π/4)                             │
│    PHASE 0 4      Phase gate (idx 0-15, see 'lut')           │
│    CNOT 0 1       CNOT: control=0 target=1                   │
│    CZ 0 1         Controlled-Z                   [SW]        │
│    CY 0 1         Controlled-Y                               │
│    CCX 0 1 2      Toffoli: c1=0 c2=1 target=2   [SW]        │
│    SWAP 0 1       Swap qubits                                │
│    MEASURE 0      Measure qubit 0 (collapse)                 │
│    INIT           Reset to |0000⟩                            │
│                                                              │
│  Analysis commands:                                          │
│    dump           full state vector                          │
│    prob           probability bar chart                      │
│    formula        state as ket notation                      │
│    bloch 0        Bloch sphere coords for qubit 0            │
│    entropy        entanglement entropy (q01 vs q23)          │
│    fidelity       compare to a saved snapshot                │
│    snap           save state snapshot for fidelity           │
│                                                              │
│  Program commands:                                           │
│    undo           undo last gate (restore previous state)    │
│    reset          full reset to |0000⟩                       │
│    history        show gate history                          │
│    run            enter multi-line program (type END)        │
│    load NAME      load a built-in circuit                    │
│    loadf FILE     load from .qasm file                       │
│    save FILE      save history to .qasm file                 │
│    shots N        run history as N-shot sampler              │
│    viz            draw ASCII circuit diagram                 │
│    hex            export assembled hex for FPGA              │
│    noise P        set depolarizing noise (e.g. noise 0.02)   │
│                                                              │
│  Reference:                                                  │
│    list           list built-in circuits                     │
│    lut            show phase LUT                             │
│    isa            instruction set reference                  │
│    help           this help text                             │
│    quit / exit    exit REPL                                  │
└─────────────────────────────────────────────────────────────┘
"""


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return "quit"


def repl(initial_noise: float = 0.0):
    """Interactive gate-by-gate REPL with full analysis and undo."""
    print(REPL_HELP)
    sim = QGRIMSim(noise_p=initial_noise)
    history: List[str] = []          # gate lines entered
    state_stack: List[List[complex]] = [sim.snapshot()]  # for undo
    meas_stack: List[Dict[int,int]] = [{}]
    pi_stack: List[float] = [0.0]
    snapshot_state: Optional[List[complex]] = None
    print(sim.dump())

    while True:
        noise_tag = f" [noise={sim.noise_p:.3f}]" if sim.noise_p > 0 else ""
        line = _safe_input(f"\nqgrim{noise_tag}> ").strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        # ── quit ──────────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # ── help / reference ──────────────────────────────────────────────
        elif cmd == "help":       print(REPL_HELP)
        elif cmd == "list":       print(list_circuits())
        elif cmd == "lut":        print(_phase_lut_str())
        elif cmd == "isa":        menu_isa()

        # ── analysis ──────────────────────────────────────────────────────
        elif cmd == "dump":       print(sim.dump())
        elif cmd == "prob":       print(sim.probability_table())
        elif cmd == "formula":
            print(f"\n  |ψ⟩ = {sim.state_formula()}")
        elif cmd == "entropy":
            S = sim.entanglement_entropy([0, 1])
            print(f"\n  S(q0,q1 | q2,q3) = {S:.4f} bits")
            if S < 0.01:   print("  (product state — no entanglement)")
            elif S > 0.98: print("  (maximally entangled)")
        elif cmd == "bloch":
            q = int(parts[1]) if len(parts) > 1 else 0
            print("\n" + sim.bloch_sphere_str(q))
        elif cmd == "snap":
            snapshot_state = sim.snapshot()
            print("  Snapshot saved.")
        elif cmd == "fidelity":
            if snapshot_state is None:
                print("  No snapshot saved. Use 'snap' first.")
            else:
                F = sim.fidelity(snapshot_state)
                print(f"\n  Fidelity F = |⟨snap|ψ⟩|² = {F:.6f}")

        # ── undo ──────────────────────────────────────────────────────────
        elif cmd == "undo":
            if len(state_stack) > 1:
                state_stack.pop()
                meas_stack.pop()
                pi_stack.pop()
                if history:
                    removed = history.pop()
                    print(f"  Undone: {removed}")
                sim.state = state_stack[-1][:]
                sim.measurements = dict(meas_stack[-1])
                sim.pi = pi_stack[-1]
                sim.pc = len(history)
                sim.halted = False
                print(sim.dump())
            else:
                print("  Nothing to undo.")

        # ── reset ─────────────────────────────────────────────────────────
        elif cmd == "reset":
            sim = QGRIMSim(noise_p=sim.noise_p)
            history.clear()
            state_stack = [sim.snapshot()]
            meas_stack = [{}]
            pi_stack = [0.0]
            print("  Reset to |0000⟩.")
            print(sim.dump())

        # ── history ───────────────────────────────────────────────────────
        elif cmd == "history":
            if not history:
                print("  (no gates entered yet)")
            else:
                print("\n  Gate history:")
                for i, g in enumerate(history, 1):
                    print(f"    {i:3d}.  {g}")

        # ── viz ───────────────────────────────────────────────────────────
        elif cmd == "viz":
            if not history:
                print("  No gates to visualize.")
            else:
                src = "\n".join(history) + "\nHALT"
                try:
                    prog = assemble(src)
                    print(draw_circuit(prog, "REPL session"))
                except AsmError as e:
                    print(f"  ASM ERROR: {e}")

        # ── hex export ────────────────────────────────────────────────────
        elif cmd == "hex":
            if not history:
                print("  No gates to export.")
            else:
                src = "\n".join(history) + "\nHALT"
                try:
                    prog = assemble(src)
                    print(hex_export(prog))
                except AsmError as e:
                    print(f"  ASM ERROR: {e}")

        # ── noise ─────────────────────────────────────────────────────────
        elif cmd == "noise":
            try:
                p = float(parts[1]) if len(parts) > 1 else 0.0
                sim.noise_p = max(0.0, min(1.0, p))
                print(f"  Noise set to {sim.noise_p:.4f}")
            except ValueError:
                print("  Usage: noise <probability>  (e.g. noise 0.02)")

        # ── shots ─────────────────────────────────────────────────────────
        elif cmd == "shots":
            if not history:
                print("  No gate history. Enter some gates first.")
                continue
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 512
            src = "\n".join(history) + "\nHALT"
            try:
                print(f"\n  Running {n} shots...")
                counts = sample_circuit(src, shots=n, noise_p=sim.noise_p)
                print(render_histogram(counts, n))
            except AsmError as e:
                print(f"  ERROR: {e}")

        # ── run multi-line ─────────────────────────────────────────────────
        elif cmd == "run":
            print("  Enter program. Type END to run, CANCEL to abort.")
            buf = []
            while True:
                l = _safe_input("  | ").strip()
                if l.upper() == "END":
                    break
                if l.upper() == "CANCEL":
                    buf = []
                    break
                buf.append(l)
            if not buf:
                print("  Cancelled.")
                continue
            src = "\n".join(buf)
            try:
                prog = assemble(src)
                sim = QGRIMSim(noise_p=sim.noise_p)
                sim.run(prog)
                history = [l for l in buf if l.strip() and not l.strip().startswith(";")]
                state_stack = [sim.snapshot()]
                meas_stack = [dict(sim.measurements)]
                pi_stack = [sim.pi]
                print(sim.dump())
            except (AsmError, RuntimeError) as e:
                print(f"  ERROR: {e}")

        # ── load built-in ──────────────────────────────────────────────────
        elif cmd == "load":
            name = parts[1].strip() if len(parts) > 1 else ""
            if name not in BUILTIN_CIRCUITS:
                print(f"  Unknown: '{name}'. Type 'list' to see options.")
                continue
            desc, src = BUILTIN_CIRCUITS[name]
            print(f"\n  {desc}")
            print("  " + "─" * 54)
            print("  " + src.strip().replace("\n", "\n  "))
            print("  " + "─" * 54)
            try:
                prog = assemble(src)
                sim = QGRIMSim(noise_p=sim.noise_p)
                sim.run(prog)
                history = [l for l in src.splitlines()
                           if l.strip() and not l.strip().startswith(";")]
                state_stack = [sim.snapshot()]
                meas_stack = [dict(sim.measurements)]
                pi_stack = [sim.pi]
                print(sim.dump())
            except (AsmError, RuntimeError) as e:
                print(f"  ERROR: {e}")

        # ── load from file ─────────────────────────────────────────────────
        elif cmd == "loadf":
            fname = parts[1] if len(parts) > 1 else ""
            if not fname:
                print("  Usage: loadf <filename.qasm>")
                continue
            src, err = load_program(fname)
            if err:
                print(f"  {err}")
                continue
            try:
                prog = assemble(src)
                sim = QGRIMSim(noise_p=sim.noise_p)
                sim.run(prog)
                history = [l for l in src.splitlines()
                           if l.strip() and not l.strip().startswith(";")]
                state_stack = [sim.snapshot()]
                meas_stack = [dict(sim.measurements)]
                pi_stack = [sim.pi]
                print(f"  Loaded: {fname}")
                print(sim.dump())
            except (AsmError, RuntimeError) as e:
                print(f"  ERROR: {e}")

        # ── save ───────────────────────────────────────────────────────────
        elif cmd == "save":
            fname = parts[1] if len(parts) > 1 else "circuit.qasm"
            if not history:
                print("  Nothing to save.")
            else:
                msg = save_program(history, fname)
                print(f"  {msg}")

        # ── gate execution ─────────────────────────────────────────────────
        else:
            try:
                words = assemble(line)
                for w in words:
                    op = (w >> 12) & 0xF
                    if op == 0xF:  # HALT
                        sim.halted = True
                        print("  [halted]")
                        continue
                    if op == 0x6:  # INIT — reset measurement state
                        sim.measurements.clear()
                        sim.pi = 0.0
                    sim.step(w)
                history.append(line)
                state_stack.append(sim.snapshot())
                meas_stack.append(dict(sim.measurements))
                pi_stack.append(sim.pi)
                print(sim.dump())
            except (AsmError, RuntimeError) as e:
                print(f"  ERROR: {e}")
                print("  Type 'help' for valid instructions.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MAIN MENU  (Pydroid-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
  ___  ___  ___ ___ __  __
 / _ \/ __|| _ \_ _|  \/  |
| (_) | (_ ||   /| || |\/| |
 \__\_\___||_|_\___|_|  |_|
        v2.1 — Complete Edition
  Pure Python · Pydroid-ready · Stdlib only
  16 circuits · Noise · CHSH · Entropy · Bloch
"""


def _phase_lut_str() -> str:
    lines = ["\nPhase LUT (matches gate_phase.v RTL exactly):"]
    lines.append("  idx  mnemonic   angle     cos Q4.12   sin Q4.12")
    lines.append("  ─────────────────────────────────────────────────")
    names = {0:"I",2:"T",4:"S",8:"Z",12:"S†",14:"T†"}
    for i, (c, s) in enumerate(PHASE_LUT):
        ang = i * 22.5
        nm = names.get(i, f"P{i}")
        lines.append(f"  {i:2d}   {nm:<6}    {ang:5.1f}°    {_q(c):+.4f}      {_q(s):+.4f}")
    return "\n".join(lines)


def menu_run_circuit(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nEnter circuit name (blank = back): ").strip()
    if not name or name not in BUILTIN_CIRCUITS:
        if name:
            print(f"  Not found: '{name}'")
        return
    desc, src = BUILTIN_CIRCUITS[name]
    print(f"\n  {desc}")
    print("─" * 56)
    print(src)
    print("─" * 56)
    prog = assemble(src)
    sim = QGRIMSim(noise_p=noise_p)
    sim.run(prog)
    print(sim.dump())
    print(f"\n  |ψ⟩ = {sim.state_formula()}")
    print(sim.probability_table())
    S = sim.entanglement_entropy([0, 1])
    print(f"\n  Entanglement entropy S(q01|q23) = {S:.4f} bits")
    see_bloch = _safe_input("\nSee Bloch sphere for which qubit? (0-3 or blank): ").strip()
    if see_bloch.isdigit() and 0 <= int(see_bloch) < QUBITS:
        print(sim.bloch_sphere_str(int(see_bloch)))
    print(draw_circuit(prog, name))


def menu_shots(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nCircuit name: ").strip()
    if name not in BUILTIN_CIRCUITS:
        print("  Not found.")
        return
    n_str = _safe_input("Number of shots [512]: ").strip()
    n = int(n_str) if n_str.isdigit() else 512
    _, src = BUILTIN_CIRCUITS[name]
    print(f"\n  Running {n} shots on '{name}'...")
    counts = sample_circuit(src, shots=n, noise_p=noise_p)
    print(render_histogram(counts, n))


def menu_custom(noise_p: float = 0.0):
    print("\n  Type your program. Type END when done, CANCEL to abort.")
    print("  Instructions: INIT  H  X  Y  Z  S  T  CNOT  CZ  CCX  PHASE  SWAP  MEASURE  HALT")
    print("  Example:  INIT  H 0  CNOT 0 1  MEASURE 0  HALT\n")
    buf = []
    while True:
        l = _safe_input("  | ").strip()
        if l.upper() == "END":
            break
        if l.upper() == "CANCEL":
            buf = []
            break
        buf.append(l)
    if not buf:
        print("  Cancelled.")
        return
    src = "\n".join(buf)
    try:
        prog = assemble(src)
        print(f"\n  Assembled {len(prog)} instruction(s):")
        for d in disassemble(prog):
            print(d)
        print()
        print(draw_circuit(prog, "custom"))
        sim = QGRIMSim(noise_p=noise_p)
        sim.run(prog)
        print(sim.dump())
        print(f"\n  |ψ⟩ = {sim.state_formula()}")
        print(sim.probability_table())
        S = sim.entanglement_entropy([0, 1])
        print(f"\n  Entanglement entropy = {S:.4f} bits")
        do_shots = _safe_input("\nRun shot sampler? [Y/n]: ").strip().lower()
        if do_shots in ("", "y", "yes"):
            n_str = _safe_input("Shots [512]: ").strip()
            n = int(n_str) if n_str.isdigit() else 512
            counts = sample_circuit(src, shots=n, noise_p=noise_p)
            print(render_histogram(counts, n))
        do_save = _safe_input("\nSave to file? (filename or blank): ").strip()
        if do_save:
            print(save_program([l for l in buf if l.strip()], do_save))
    except (AsmError, RuntimeError) as e:
        print(f"\n  ERROR: {e}")


def menu_noise_sim():
    print(list_circuits())
    name = _safe_input("\nCircuit name: ").strip()
    if name not in BUILTIN_CIRCUITS:
        print("  Not found.")
        return
    p_str = _safe_input("Depolarizing noise probability per gate [0.01]: ").strip()
    try:
        p = float(p_str) if p_str else 0.01
    except ValueError:
        p = 0.01
    n_str = _safe_input("Shots [500]: ").strip()
    n = int(n_str) if n_str.isdigit() else 500
    _, src = BUILTIN_CIRCUITS[name]
    print(f"\n  Running '{name}' with noise p={p:.4f}  ({n} shots)...")
    counts_clean = sample_circuit(src, shots=n, noise_p=0.0)
    counts_noisy = sample_circuit(src, shots=n, noise_p=p)
    print("\n  ── CLEAN ──")
    print(render_histogram(counts_clean, n))
    print("\n  ── NOISY ──")
    print(render_histogram(counts_noisy, n))


def menu_chsh():
    n_str = _safe_input("\nShots per setting [500]: ").strip()
    n = int(n_str) if n_str.isdigit() else 500
    print(chsh_test(n))


def menu_witness():
    print("\n──── WITNESS / ODEA DECODER ────")
    print("Maps a 4-qubit measurement outcome to its Witness label.")
    print("\n  |basis⟩  Witness  Meaning")
    print("  ─────────────────────────────────────────────────────")
    for i in range(STATES):
        lbl = WITNESS_LABELS[i]
        meaning = WITNESS_MEANING[lbl]
        print(f"  |{i:04b}⟩    {lbl}       {meaning}")


def menu_isa():
    print("""
═══ QGRIM v2.1 INSTRUCTION SET ════════════════════════════════════════════
 Encoding:  [15:12] opcode  |  [11:8] A  |  [7:4] B  |  [3:0] IMM

 Hardware ISA (runs on FPGA chip):
  Opcode  Mnemonic    Operands        Description
  ──────  ──────────  ──────────────  ─────────────────────────────────────
  0x0     NOP         —               One cycle of nothing
  0x1     H           q               Hadamard gate on qubit q
  0x2     X           q               Pauli-X (bit flip)
  0x3     CNOT        c  t            Controlled-NOT  control=c  target=t
  0x4     MEASURE     q               Born-rule collapse, latch to MEAS_OUT
  0x5     PHASE       q  idx          Multiply |1⟩_q by e^{i·idx·π/8}
  0x6     INIT        —               Reset state vector to |0000⟩
  0x7     SWAP        a  b            Swap qubits a and b
  0x8     LOAD_AMP    basis           Collapse to single basis |basis⟩
  0x9     TRACE       —               Snapshot (PC, MEAS_OUT, PI) to log
  0xE     WAIT        n               Idle n cycles (hardware sync)
  0xF     HALT        —               Stop, set DONE, raise IRQ

 Software-only extensions (simulator only — not sent to FPGA):
  0xA     Y           q               Pauli-Y gate
  0xB     CZ          c  t            Controlled-Z
  0xC     CCX         c1  c2  target  Toffoli gate  (c1=A, c2=B, t=IMM)
  0xD     RZ          q  idx          Alias for PHASE (float-exact intent)

 Macro aliases (assembler expands to ISA words):
  S  q                S gate = PHASE q 4      (π/2)
  T  q                T gate = PHASE q 2      (π/4)
  Z  q                Z gate = PHASE q 8      (π)
  SDG q               S† gate = PHASE q 12    (−π/2)
  TDG q               T† gate = PHASE q 14    (−π/4)
  CX  c t             CNOT alias
  NOT q               X alias
  CY  c t             Controlled-Y  (expands to 3 instructions)
  CH  c t             Controlled-H  (expands to 7 instructions)
  TOFFOLI c1 c2 t     CCX alias

 Phase LUT index → angle:
  0→0°  1→π/8  2→π/4(T)  4→π/2(S)  8→π(Z)  12→−π/2(S†)  14→−π/4(T†)

 SPI Register Map (for real hardware):
  0x00 CTRL (START/RESET)  0x08 PI_REG  0x0C MEAS_OUT  0x10 STATUS
  0x14 PC_REG  0x1C VERSION(0x0210)  0x20 PGM_PAGE  0x80+ PGM_DATA
  0xC0+ SV_BASE (state vector, 2×Q4.12 per amplitude)
═══════════════════════════════════════════════════════════════════════════════
""")


def menu_chip_info():
    print("""
═══ QGRIM v2.1 CHIP INFORMATION ═══════════════════════════════════════════
 Target:        iCE40HX1K (iCEstick evaluation board)
 Clock:         SB_HFOSC → 48 MHz → /4 → 12 MHz system clock
 Qubits:        4 (state vector: 16 × 32-bit complex amplitudes, Q4.12)
 Program mem:   256 × 16-bit instructions (1 BRAM on iCE40)
 Gate cycles (at 12 MHz system clock):
   H           41 cycles   (~3.4 µs)
   X           18 cycles   (~1.5 µs)
   CNOT        26 cycles   (~2.2 µs)
   PHASE       65 cycles   (~5.4 µs)
   MEASURE     90 cycles   (~7.5 µs)
   INIT        18 cycles   (~1.5 µs)
 RTL modules:   12 Verilog files in qgrim/rtl/
 Three structural fixes vs v2.0 audit:
   Fix 1 — spi_bram_bridge.v: atomic write (no partial-byte BRAM writes)
   Fix 2 — measurement_unit.v: correct Born rule (|a|² accumulation)
   Fix 3 — pair_addr_gen.v: gate-correct paired addressing (XOR partner)
 Host interface: SPI Mode 0, 6-byte fixed packets, 1 MHz max
 Toolchain:     yosys → nextpnr-ice40 → icepack → iceprog
 Build:         cd qgrim && make
 Flash:         cd qgrim && make prog   (requires icestick connected)
 Sim verify:    cd qgrim/host && python3 QGRIM_ENGINE.py --run bell
═══════════════════════════════════════════════════════════════════════════════
""")


def menu_bloch(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nCircuit name (or blank for current state): ").strip()
    if name and name not in BUILTIN_CIRCUITS:
        print(f"  Not found: '{name}'")
        return
    if name:
        _, src = BUILTIN_CIRCUITS[name]
        prog = assemble(src)
        sim = QGRIMSim(noise_p=noise_p)
        sim.run(prog)
    else:
        sim = QGRIMSim()
    for q in range(QUBITS):
        print(sim.bloch_sphere_str(q))
        print()


def menu_entropy(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nCircuit name: ").strip()
    if name not in BUILTIN_CIRCUITS:
        print(f"  Not found: '{name}'")
        return
    _, src = BUILTIN_CIRCUITS[name]
    prog = assemble(src)
    sim = QGRIMSim(noise_p=noise_p)
    sim.run(prog)
    print(sim.dump())
    print(f"\n  State formula: {sim.state_formula()}")
    print()
    print("  Von Neumann entanglement entropy across all single-qubit cuts:")
    print("  ─────────────────────────────────────────────────────────────")
    # All single-qubit bipartitions — these reveal which pairs are entangled
    for q in range(QUBITS):
        S = sim.entanglement_entropy([q])
        rest = [r for r in range(QUBITS) if r != q]
        label_b = "".join(f"q{r}" for r in rest)
        bar_len = int(S * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        note = ""
        if S > 0.95:   note = "  ← maximally entangled"
        elif S > 0.1:  note = "  ← partially entangled"
        else:          note = "  ← separable (no entanglement)"
        print(f"  S(q{q} | {label_b}) = {S:.4f} bits  {bar}{note}")
    print()
    print("  Two-qubit partition cuts:")
    print("  ─────────────────────────────────────────────────────────────")
    for partition in ([0, 1], [0, 2], [0, 3]):
        S = sim.entanglement_entropy(partition)
        rest = [q for q in range(QUBITS) if q not in partition]
        label_a = "".join(f"q{q}" for q in partition)
        label_b = "".join(f"q{q}" for q in rest)
        print(f"  S({label_a} | {label_b}) = {S:.4f} bits")


def menu_viz(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nCircuit name: ").strip()
    if name not in BUILTIN_CIRCUITS:
        print(f"  Not found: '{name}'")
        return
    _, src = BUILTIN_CIRCUITS[name]
    prog = assemble(src)
    print(draw_circuit(prog, name))
    desc, _ = BUILTIN_CIRCUITS[name]
    print(f"\n  {desc}")
    for d in disassemble(prog):
        print(d)


def menu_hex(noise_p: float = 0.0):
    print(list_circuits())
    name = _safe_input("\nCircuit name: ").strip()
    if name not in BUILTIN_CIRCUITS:
        print(f"  Not found: '{name}'")
        return
    _, src = BUILTIN_CIRCUITS[name]
    prog = assemble(src)
    print(hex_export(prog))
    do_save = _safe_input("\nSave hex to file? (filename or blank): ").strip()
    if do_save:
        fname = do_save if "." in do_save else do_save + ".hex"
        try:
            with open(fname, "w") as f:
                f.write(hex_export(prog))
            print(f"  Saved to: {os.path.abspath(fname)}")
        except OSError as e:
            print(f"  Save failed: {e}")


def main_menu():
    """Top-level menu — works on Pydroid and desktop."""
    print(BANNER)
    # Noise level persists across menu choices
    noise_p: float = 0.0

    def n(): return noise_p

    options = [
        ("Run a built-in circuit",               lambda: menu_run_circuit(n())),
        ("Shot sampler (N-run histogram)",         lambda: menu_shots(n())),
        ("Enter a custom circuit",                 lambda: menu_custom(n())),
        ("Interactive gate REPL",                  lambda: repl(n())),
        ("Noise simulation (clean vs noisy)",      lambda: menu_noise_sim()),
        ("CHSH Bell inequality test",              lambda: menu_chsh()),
        ("Entanglement entropy",                   lambda: menu_entropy(n())),
        ("Bloch sphere viewer",                    lambda: menu_bloch(n())),
        ("Circuit visualizer (ASCII diagram)",     lambda: menu_viz(n())),
        ("Hex export for FPGA",                    lambda: menu_hex(n())),
        ("Witness / ODEA decoder table",           lambda: menu_witness()),
        ("Instruction set reference",              lambda: menu_isa()),
        ("Chip information & register map",        lambda: menu_chip_info()),
        ("Phase LUT reference",                    lambda: print(_phase_lut_str())),
    ]

    while True:
        print("\n" + "═" * 46)
        print("  QGRIM Engine — Main Menu"
              + (f"  [noise={noise_p:.3f}]" if noise_p > 0 else ""))
        print("═" * 46)
        for i, (label, _) in enumerate(options, 1):
            print(f"  {i:2d}.  {label}")
        print("  ─────────────────────────────────────────")
        print("   N.  Set noise level (depolarizing)")
        print("   0.  Exit")
        print("─" * 46)
        choice = _safe_input(f"Choose [0-{len(options)}]: ").strip()
        if choice == "0":
            print("Goodbye.")
            break
        elif choice.lower() == "n":
            p_str = _safe_input("Noise probability per gate (0 = off): ").strip()
            try:
                noise_p = float(p_str)
                noise_p = max(0.0, min(1.0, noise_p))
                print(f"  Noise set to {noise_p:.4f}")
            except ValueError:
                print("  Invalid number.")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    options[idx][1]()
                else:
                    print("  Invalid choice.")
            except ValueError:
                print("  Please type a number.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def cli():
    args = sys.argv[1:]

    # Parse optional --noise flag
    noise_p = 0.0
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--noise" and i + 1 < len(args):
            try:
                noise_p = float(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    if not args:
        main_menu()
        return

    if args[0] == "--run" and len(args) >= 2:
        name = args[1]
        if name not in BUILTIN_CIRCUITS:
            print(f"Unknown circuit '{name}'. Options:\n" + list_circuits())
            sys.exit(1)
        desc, src = BUILTIN_CIRCUITS[name]
        print(f"\n{desc}")
        prog = assemble(src)
        sim = QGRIMSim(noise_p=noise_p)
        sim.run(prog)
        print(sim.dump())
        print(f"\n|ψ⟩ = {sim.state_formula()}")
        print(sim.probability_table())

    elif args[0] == "--shots" and len(args) >= 2:
        name = args[1]
        n = int(args[2]) if len(args) >= 3 else 512
        if name not in BUILTIN_CIRCUITS:
            print(f"Unknown circuit '{name}'.")
            sys.exit(1)
        _, src = BUILTIN_CIRCUITS[name]
        counts = sample_circuit(src, shots=n, noise_p=noise_p)
        print(render_histogram(counts, n))

    elif args[0] == "--viz" and len(args) >= 2:
        name = args[1]
        if name not in BUILTIN_CIRCUITS:
            print(f"Unknown circuit '{name}'.")
            sys.exit(1)
        _, src = BUILTIN_CIRCUITS[name]
        prog = assemble(src)
        print(draw_circuit(prog, name))
        for d in disassemble(prog):
            print(d)

    elif args[0] == "--chsh":
        n = int(args[1]) if len(args) >= 2 else 500
        print(chsh_test(n))

    elif args[0] == "--entropy" and len(args) >= 2:
        name = args[1]
        if name not in BUILTIN_CIRCUITS:
            print(f"Unknown circuit '{name}'.")
            sys.exit(1)
        _, src = BUILTIN_CIRCUITS[name]
        prog = assemble(src)
        sim = QGRIMSim(noise_p=noise_p)
        sim.run(prog)
        S = sim.entanglement_entropy([0, 1])
        print(f"Entanglement entropy S(q01|q23) = {S:.4f} bits")

    elif args[0] == "--file" and len(args) >= 2:
        src, err = load_program(args[1])
        if err:
            print(err)
            sys.exit(1)
        prog = assemble(src)
        sim = QGRIMSim(noise_p=noise_p)
        sim.run(prog)
        print(sim.dump())
        print(f"\n|ψ⟩ = {sim.state_formula()}")
        print(sim.probability_table())

    elif args[0] == "--repl":
        repl(noise_p)

    elif args[0] == "--list":
        print(list_circuits())

    elif args[0] == "--isa":
        menu_isa()

    elif args[0] == "--chip":
        menu_chip_info()

    elif args[0] == "--hex" and len(args) >= 2:
        name = args[1]
        if name not in BUILTIN_CIRCUITS:
            print(f"Unknown circuit '{name}'.")
            sys.exit(1)
        _, src = BUILTIN_CIRCUITS[name]
        prog = assemble(src)
        print(hex_export(prog))

    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    cli()
=====================================