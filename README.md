`Author Shariya` : It's more like i was talking with myself while writing 😂


Past Time : I built QGRIM v2.1 as a classical software simulator...
 
Present Time : QGRIM v2.1 is a classical software simulator of a 4-qubit quantum circuit processor, packaged as a single self-contained Python file with zero dependencies beyond the standard library. It runs anywhere Python 3.8+ runs—including Pydroid on Android with no internet, no pip, and no setup.
 
Past time : The challenge was to implement this faithfully...
 
Present time : The engine implements quantum state-vector simulation faithfully without NumPy or SciPy. It relies only on Python's standard library because the target environment is Pydroid, where external packages may not be available.
 
Present time : I solved the problem by writing... more # Data model and algorithms for multimodal route planning with transportation networks (and)
 
[ [https://academic.oup.com/mnras/article-pdf/279/2/693/3230070/279-2-693.pdf](https://academic.oup.com/mnras/article-pdf/279/2/693/3230070/279-2-693.pdf) ] ↓ [ [https://aclanthology.org/2026.findings-acl.1126.pdf](https://aclanthology.org/2026.findings-acl.1126.pdf) ] ↓ As inspiration, but also added! yesterday my own solutions by reading few China chip model deigning.
 
Final Execution : The engine includes custom implementations of: Power iteration with deflation for eigenvalues. Schmidt decomposition for entanglement entropy. Born-rule measurement with wavefunction collapse. Explicit matrix-based quantum gate operations.
 
Future goal on Android : QGRIM provides a platform for exploring quantum-inspired neural architectures. While it is not a neural-network framework, it can be used to prototype new neuron models, feature encodings, probabilistic routing mechanisms, and learning experiments based on quantum circuit mathematics.
 
============================================
 
Polished Description :
 
QGRIM Engine — Developer's Description
 
What I Built QGRIM v2.1 is a classical software simulator of a 4-qubit quantum circuit processor, packaged as a single self-contained Python file with zero dependencies beyond the standard library. It runs anywhere Python 3.8+ runs — including Pydroid on Android with no internet, no pip, and no setup. The engine has two layers of purpose: it is a faithful software model of real FPGA hardware (the iCE40HX1K chip, with RTL written in Verilog), and it is simultaneously a fully interactive quantum computing playground that anyone can use to learn and experiment.
 
The Core Problem I Solved A real quantum chip stores its state as a vector of 2ⁿ complex amplitudes — for 4 qubits, that is 16 complex numbers. Every gate is a unitary matrix multiplication on that vector. The challenge was to implement this faithfully without NumPy or SciPy, using only Python's built-in math, cmath, random, and dataclasses — because the target runtime is Pydroid, which has no guaranteed package manager. Every algorithm that would normally take one line of NumPy (np.linalg.eig, scipy.linalg.svd) had to be written from scratch: Power iteration + deflation for eigenvalues (used by entanglement entropy) Schmidt decomposition by building a Gram matrix and extracting singular values Ry gate as explicit 2×2 matrix application across the full 16-entry state vector (used by the CHSH test for correct basis rotation) Born-rule measurement with wavefunction collapse and renormalization
 
Architecture The engine is divided into 12 sections, each self-contained: QGRIM_ENGINE.py  (~2,130 lines) │ ├── Section 1   Constants — phase LUT (16-entry, matches RTL gate_phase.v exactly), │               witness labels, opcode tables │ ├── Section 2   Assembler — converts QASM text to 16-bit instruction words. │               Handles hardware opcodes, software-extension opcodes (Y, CZ, Toffoli), │               and macro aliases (S, T, Z, SDG, TDG, CY, CH, TOFFOLI) │ ├── Section 3   QGRIMSim dataclass — the state vector engine. │               Q4.12 fixed-point arithmetic throughout (matching the FPGA RTL), │               depolarizing noise model, all gate implementations │ ├── Section 4   Linear algebra — eigenvalue solver (power iteration + deflation), │               entanglement entropy, fidelity, Bloch sphere │ ├── Section 5   Circuit visualizer — ASCII diagram renderer with vertical │               connection lines for two-qubit gates │ ├── Section 6   Shot sampler + histogram renderer │ ├── Section 7   CHSH Bell inequality test — proper Ry basis rotations, │               4-setting correlation measurement, violation check │ ├── Section 8   19 built-in circuits — Bell, GHZ, QFT, Grover, Deutsch, │               Bernstein-Vazirani, teleportation, superdense, cluster state, │               QPE, QRNG, Toffoli demo, CZ demo, Y gate, T gate, and more │ ├── Section 9   Save / load — .qasm file I/O (Pydroid storage compatible) │ ├── Section 10  Interactive REPL — gate-by-gate with unlimited undo, │               formula view, Bloch sphere, entropy, fidelity, shots │ ├── Section 11  Main menu — 14 numbered options, Pydroid-friendly input loop │ └── Section 12  CLI flags — --run, --shots, --viz, --chsh, --entropy, --file, --repl, --list, --isa, --chip, --hex, --noise
 
Key Design Decisions Q4.12 fixed-point everywhere. The FPGA RTL stores amplitudes as 16-bit signed fixed-point numbers (4 integer bits, 12 fractional bits). The simulator mirrors this exactly — every gate result is quantized to the Q4.12 grid before being stored. This means the software model and the hardware chip accumulate identical rounding errors, making the simulator a true behavioral twin of the chip, not just a mathematical approximation. Depolarizing noise model. After every gate, if noise_p > 0, each affected qubit has a probability noise_p of having a random Pauli error (X, Y, or Z) injected. This models gate imperfection on real hardware. The noise is applied at the state vector level, not the circuit level, so it interacts correctly with subsequent gates. Varied starting vectors for degenerate eigenvalues. The power iteration eigenvalue solver uses golden-ratio-spaced initial vectors (φ = 1.618...) for each deflation step. This is critical for the Bell state, where the Gram matrix has two equal eigenvalues (both 0.5). A fixed starting vector fails silently — it finds only one eigenvalue, returning 0.5 bits of entropy instead of the correct 1.0 bit. The quasi-random restart strategy guarantees orthogonal initial directions across deflation steps. Proper Ry rotations for CHSH. A common mistake in CHSH simulation is using PHASE gates for basis rotation. PHASE gates apply e^{iθ} to the |1⟩ component — this rotates the phase of the amplitude but does not change the Z-basis measurement statistics, so correlations stay at ±1. The correct rotation for measuring at angle α is Ry(α), which mixes the |0⟩ and |1⟩ amplitudes. The CHSH test implements this directly as a 2×2 matrix applied to the 16-entry state vector, bypassing the ISA quantization. Result: S ≈ 2.875 (theoretically 2√2 ≈ 2.828), confirming genuine quantum violation. Macro expansion in the assembler. Composite gates like CY and CH are not ISA primitives — they expand to 3 and 7 instructions respectively at assemble time. The assembler handles this transparently, so the user writes CY 0 1 and gets the correct gate sequence without needing to know the decomposition.
 
What It Can Do Feature Detail State vector 16 complex amplitudes, Q4.12 fixed-point Gates H, X, Y, Z, S, T, SDG, TDG, PHASE, CNOT, CZ, CY, CH, TOFFOLI, SWAP Noise model Per-gate depolarizing, configurable probability Entanglement entropy Von Neumann via Schmidt decomposition, all bipartitions CHSH test Ry basis rotations, 4-setting correlation, violation detection Bloch sphere (x, y, z) for any qubit via reduced density matrix Fidelity
 
Shot sampler N independent runs, ASCII histogram Circuit visualizer ASCII diagram with ●/⊕/H/M/║ symbols Built-in circuits 19 circuits covering all major quantum algorithms REPL Gate-by-gate with undo, snap, formula, entropy, save File I/O Save/load .qasm programs Hex export Intel HEX-style dump for FPGA SPI upload ISA reference Full opcode table, phase LUT, register map built-in
 
What It Is Not
 
QGRIM is a classical simulation of quantum mathematics — it runs on a conventional CPU and uses floating-point/fixed-point arithmetic to track the amplitudes of a wavefunction. It does not use actual quantum hardware. It cannot offer quantum speedup. The "chip" it models is an FPGA that does the same classical simulation in hardware logic at higher speed — not a physical qubit device like a superconducting transmon or trapped ion system. The engine is useful for: learning quantum circuits, verifying algorithms before running on real hardware, testing gate decompositions, and teaching quantum information concepts (entanglement, decoherence, Bell inequalities) with immediate visual feedback on a mobile device. Built by hand — every line from scratch. No dependencies. Runs on a phone.
 
Where and How QGRIM Can Be Used
 
General Use Cases
 
 
1.  
Education & Learning Anyone studying quantum computing can use it to run real circuits and see the math happen — amplitudes, probabilities, collapse — without needing a cloud account or real quantum hardware. It runs locally, offline, on a phone.
 
 
2.  
Algorithm Prototyping Before sending a circuit to IBM Quantum or Google Sycamore (which cost compute credits and have queue times), you can prototype and debug the logic in QGRIM first. The ISA is designed to map directly to FPGA hardware, so a verified circuit here is close to hardware-ready.
 
 
3.  
FPGA Hardware Companion If you build the actual iCE40HX1K chip from the RTL files in qgrim/rtl/, QGRIM is the software twin. You can verify expected output in software, then flash the hex file via SPI to the physical chip and compare results.
 
 
4.  
Research & Demonstrations CHSH violation, entanglement entropy, quantum teleportation, Bernstein-Vazirani — all produce real, verifiable quantum results. Useful for presentations, papers, or demos where you need a reproducible, self-contained example.
 
 

 
Can It Be Used in an AI Neuron Project?
 
Yes — in several concrete ways. Here is how:
 
 
1. Quantum-Inspired Neural Activation
 

 
A classical neuron computes: output = activation(weights · inputs + bias) You can replace the activation function with a quantum phase or interference measurement. Feed weighted inputs as phase angles into QGRIM's PHASE gates, run interference (Hadamard + CNOT), and read the collapsed probability as the neuron's output signal. This is called a quantum-inspired neuron — it uses quantum math (superposition, interference) computed classically. Input weights → PHASE angles → H gates → interference → MEASURE → output probability
 
 
1. Quantum Random Number Generator for Neural Nets
 

 
Neural networks need randomness for:
 
Weight initialization
 
Dropout masks
 
Stochastic gradient descent sampling
 
QGRIM's qrng circuit (H 0; H 1; H 2; H 3; MEASURE all) produces 4 genuinely Born-rule-random bits per run — randomness that comes from quantum measurement collapse, not a pseudo-random algorithm. You can call it in a loop to seed any neural network's stochastic operations.
 
 
1. Quantum Feature Encoding
 

 
In quantum machine learning (QML), classical data is encoded into a quantum state before processing. QGRIM can simulate this:
 
Normalize your feature vector
 
Encode each feature as a rotation angle (PHASE gate index)
 
Run the circuit to produce a quantum state
 
Read the amplitude probabilities as the encoded feature representation
 
This is called amplitude encoding or angle encoding — standard techniques in QML papers.
 
 
1. Entanglement as a Correlation Measure
 

 
In a neuron network, you often want to measure how strongly two nodes are correlated. QGRIM's entanglement entropy (S(q0 | q1q2q3)) is exactly a measure of quantum correlation between subsystems. You can:
 
Encode two data signals as qubit states
 
Entangle them via CNOT/CZ gates
 
Measure the von Neumann entropy to quantify their correlation strength
 
This gives you a non-linear, interference-based correlation metric — richer than simple dot products.
 
 
1. Variational Quantum Circuit (VQC) Layer
 

 
The most direct AI use: treat the PHASE gate indices as learnable parameters. A training loop would: 1. Forward pass: run the circuit with current PHASE angles → get output probabilities
 
 
1. 
 

 
Loss: compare to target labels
 
 
1. 
 

 
Backward pass: shift each angle by ±π/2, re-run (parameter-shift rule), compute gradient
 
 
1. 
 

 
Update: adjust PHASE indices to reduce loss
 
QGRIM already has all the circuit execution needed for steps 1 and 3. You would wrap it in a Python training loop. This is exactly what IBM's Qiskit Machine Learning and PennyLane do — QGRIM is a stripped-down, dependency-free version of the same concept.
 
Honest Limitations for AI Use
 
Limitation Reason 4 qubits only State vector is 16 entries — small for real ML tasks No automatic differentiation Gradients must be computed manually via parameter-shift Q4.12 quantization Introduces small rounding errors that accumulate over many gate layers No GPU acceleration Pure Python — slow for large training loops
 
The Bottom Line
 
QGRIM is the right tool if you want to:
 
Understand how quantum circuits behave inside an AI system
 
Prototype a quantum layer before using a full framework
 
Teach how quantum neurons differ from classical ones
 
Run on mobile without any cloud dependency
 
It is not a replacement for PennyLane or Qiskit for production QML — but as a learning engine, a proof-of-concept testbed, and a hardware-backed simulator that fits in a single file, it fits naturally into any AI + quantum neuron research project, especially in early-stage exploration where you want to understand the mechanics before scaling up.
 
============================================