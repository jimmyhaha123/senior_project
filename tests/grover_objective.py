# grover4_objective.py
# Objective for 4-qubit Grover: minimize (1 - fidelity_est + λ * makespan_proxy)
# Interface matches your runner: returns (space, objective_callable) via make_grover4_objective()
# Sampling-based fidelity estimate using counts (hardware-like).
#
# Decision variables (Regime-1; Dd=10, Dc=10)
# ----------------------------------------------------------------------
# Discrete (10 binary bits total):
#   b0..b7 : 8 bits (2 bits per logical qubit q0..q3) -> preferred physical {0,1,2,3}
#            duplicates resolved greedily to a valid permutation (initial_layout).
#   b8..b9 : 2 bits routing method 00->'sabre', 01->'stochastic', 10->'basic', 11->'sabre'
#
# Continuous (10 reals in [0,1], linearly mapped to physical ranges):
#   c0 : scale_1q          ∈ [0.7, 1.3]   -> 1q gate duration multiplier
#   c1 : scale_2q          ∈ [0.7, 1.3]   -> 2q gate duration multiplier
#   c2 : p_1q              ∈ [5e-4, 3e-3] -> 1q depolarizing prob
#   c3 : p_2q              ∈ [5e-3, 3e-2] -> 2q depolarizing base prob
#   c4 : readout_p01       ∈ [0.005,0.05] -> 0→1 flip prob at measurement
#   c5 : readout_p10       ∈ [0.005,0.05] -> 1→0 flip prob at measurement
#   c6 : T1_scale          ∈ [0.5, 1.5]   -> base T1 (100μs) multiplier
#   c7 : T2_scale          ∈ [0.5, 1.5]   -> base T2 (100μs) multiplier
#   c8 : extra_idle_buffer ∈ [0, 80ns]    -> per-depth guard, used in makespan proxy only
#   c9 : two_qubit_drift   ∈ [0.8, 1.2]   -> multiplier on 2q depolarizing (p_2q_eff)
#
# Objective:
#   f = (1 - p_marked) + λ * MakespanProxy
#   where p_marked is the sampled probability of the marked bitstring,
#   MakespanProxy = depth * max(dur_2q, dur_1q) + depth * idle_buffer
#
# Requirements: qiskit >= 0.45, qiskit-aer
# ----------------------------------------------------------------------

from math import floor, pi, sqrt
from typing import Tuple
from functools import lru_cache

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

# ---- Your MIVABO types (import from your repo) ----
from MIVABO import MixedSpace, DiscreteSpec, ContinuousSpec, LinearConstraint
import torch as _t


# ---------------- Grover circuit helpers ----------------

def _oracle_marked_state(n_qubits: int, marked: str) -> QuantumCircuit:
    assert len(marked) == n_qubits and set(marked) <= {"0", "1"}
    qc = QuantumCircuit(n_qubits, name=f"Oracle|{marked}>")
    # map |marked> -> |11..1| with X on zero bits (LSB at qubit 0)
    for i, b in enumerate(reversed(marked)):
        if b == "0":
            qc.x(i)
    # apply multi-controlled Z as H + MCX + H on last qubit
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), [*range(n_qubits - 1), n_qubits - 1])
    qc.h(n_qubits - 1)
    # undo the Xs
    for i, b in enumerate(reversed(marked)):
        if b == "0":
            qc.x(i)
    return qc


def _diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), [*range(n_qubits - 1), n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def _build_grover4(marked: str, iterations: int | None = None) -> QuantumCircuit:
    n = 4
    if iterations is None:
        iterations = floor(pi/4 * sqrt(2**n))  # 3 for n=4
    qc = QuantumCircuit(n, name=f"Grover4|{marked}>")
    qc.h(range(n))
    O, D = _oracle_marked_state(n, marked), _diffuser(n)
    for _ in range(iterations):
        qc.append(O.to_gate(), range(n))
        qc.append(D.to_gate(), range(n))
    return qc


def _build_grover4_decomposed(marked: str, iterations: int | None = None) -> QuantumCircuit:
    """Pre-decompose into a tight basis ahead of transpilation to make repeated calls cheap."""
    qc = _build_grover4(marked, iterations)
    # Push MCX and composite gates into {cx, rz, sx, x, id}
    return qc.decompose(reps=3)


# ---------------- Regime-1 decoding (Dd=10, Dc=10) ----------------

def _decode_layout_from_8bits(bits8):
    """
    bits8: length-8 iterable (2 bits per logical qubit) -> initial_layout (permutation [0,1,2,3])
    Map each logical qi to preferred physical vi in {0,1,2,3} via its 2-bit value,
    then greedily resolve duplicates.
    """
    assert len(bits8) == 8
    prefs = []
    for i in range(4):
        v = (int(bits8[2 * i]) & 1) | ((int(bits8[2 * i + 1]) & 1) << 1)  # 0..3
        prefs.append(v)
    remaining = set([0, 1, 2, 3])
    mapping = [-1] * 4
    for i, v in enumerate(prefs):
        if v in remaining:
            mapping[i] = v
            remaining.remove(v)
    for i in range(4):
        if mapping[i] < 0:
            mapping[i] = remaining.pop()
    return mapping


def _decode_routing_from_2bits(b0, b1):
    """2-bit categorical -> routing method string. 00->'sabre', 01->'stochastic', 10->'basic', 11->'sabre'."""
    code = (int(b0) & 1) | ((int(b1) & 1) << 1)
    return {0: 'sabre', 1: 'stochastic', 2: 'basic', 3: 'sabre'}[code]


def _affine(u, lo, hi):
    """Map u in [0,1] to [lo, hi]."""
    return lo + (hi - lo) * float(u)


# ---------------- Global config & caches to avoid stalls ----------------

# A small, realistic coupling map (line across 4 nodes)
_COUPLING = CouplingMap(couplinglist=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
# Tight basis so transpiler doesn't explore large spaces
_BASIS = ["cx", "rz", "sx", "x", "id"]
# One backend reused across calls
_BACKEND = AerSimulator()  # qasm-style sampling simulator
# Deterministic transpilation (helps cache hits)
_SEED_TR = 1234


@lru_cache(maxsize=128)
def _compile_structure_only(marked: str, layout_key: tuple, routing: str | None):
    """Compile a *structure-only* Grover circuit once per (marked, layout, routing), without backend.

    We do NOT pass a backend to avoid invalidating target properties; we supply our own noise model anyway.
    """
    qc = _build_grover4_decomposed(marked)
    kwargs = dict(
        initial_layout=list(layout_key),
        coupling_map=_COUPLING,   # our small map
        basis_gates=_BASIS,       # tight basis
        optimization_level=1,
        seed_transpiler=_SEED_TR,
    )
    try:
        if routing in {"sabre", "stochastic", "basic"}:
            kwargs["routing_method"] = routing
        compiled = transpile(qc, **kwargs)   # no backend here
    except TranspilerError:
        kwargs.pop("routing_method", None)
        compiled = transpile(qc, **kwargs)
    return compiled


# ---------------- Noise model (with readout error, since we measure) ----------------

def _make_noise_model_from_continuous(xc, n_qubits=4):
    """
    xc: length-10 vector in [0,1].
      c0 -> scale_1q in [0.7, 1.3]
      c1 -> scale_2q in [0.7, 1.3]
      c2 -> p_1q in [5e-4, 3e-3]
      c3 -> p_2q in [5e-3, 3e-2]
      c4 -> readout_p01 in [0.005, 0.05]
      c5 -> readout_p10 in [0.005, 0.05]
      c6 -> T1_scale in [0.5, 1.5]  (base T1 = 100 µs)
      c7 -> T2_scale in [0.5, 1.5]  (base T2 = 100 µs)
      c8 -> extra_idle_buffer per depth layer in [0, 80 ns] (makespan proxy)
      c9 -> 2q depolarizing multiplier in [0.8, 1.2]
    Returns: (noise_model, dur_1q, dur_2q, idle_buf, p2q_eff)
    """
    scale_1q = _affine(xc[0], 0.7, 1.3)
    scale_2q = _affine(xc[1], 0.7, 1.3)
    p_1q     = _affine(xc[2], 5e-4, 3e-3)
    p_2q     = _affine(xc[3], 5e-3, 3e-2)
    ro01     = _affine(xc[4], 0.005, 0.05)
    ro10     = _affine(xc[5], 0.005, 0.05)
    T1       = 100e-6 * _affine(xc[6], 0.5, 1.5)
    T2       = 100e-6 * _affine(xc[7], 0.5, 1.5)
    idle_buf = _affine(xc[8], 0.0, 80e-9)
    mult2q   = _affine(xc[9], 0.8, 1.2)
    p_2q_eff = p_2q * mult2q

    # Enforce physical constraint T2 <= 2*T1 explicitly to avoid simulator errors
    try:
        from MIVABO import InfeasibleError as _Infeas
    except Exception:
        class _Infeas(Exception):
            pass
    if T2 > 2.0 * T1 + 1e-18:
        raise _Infeas("constraint violated: T2 must be <= 2*T1")

    base_1q = 50e-9
    base_2q = 300e-9
    dur_1q = base_1q * scale_1q
    dur_2q = base_2q * scale_2q

    noise = NoiseModel()

    # 1q errors (attach to basis gates; 'h' is not needed since we decomposed)
    for g in ["sx", "x", "rz", "id"]:
        trel = thermal_relaxation_error(T1, T2, dur_1q)
        dep  = depolarizing_error(p_1q, 1)
        err1 = trel.compose(dep)
        for q in range(n_qubits):
            noise.add_quantum_error(err1, g, [q])

    # 2q errors (applied to 'cx')
    trel2 = thermal_relaxation_error(T1, T2, dur_2q).tensor(
            thermal_relaxation_error(T1, T2, dur_2q))
    dep2  = depolarizing_error(p_2q_eff, 2)
    err2  = trel2.compose(dep2)
    noise.add_all_qubit_quantum_error(err2, "cx")

    # Readout error (now relevant)
    ro = ReadoutError([[1 - ro01, ro01], [ro10, 1 - ro10]])
    for q in range(n_qubits):
        noise.add_readout_error(ro, [q])

    return noise, dur_1q, dur_2q, idle_buf, p_2q_eff


# ---------------- Objective factory (matches your interface) ----------------

def make_grover4_objective(marked: str = "1110", lam: float = 0.0, shots: int = 4000) -> Tuple[MixedSpace, callable]:
    """
    Returns (space, f) with:
      - space: MixedSpace(DiscreteSpec(Dd=10), ContinuousSpec(Dc=10))
      - f(xd_bits, xc_unit) -> float: objective = 1 - p(marked) + lam * makespan_proxy
    Sampling-based fidelity estimate with `shots` (hardware-like).
    """
    assert isinstance(shots, int) and shots > 0, "shots must be a positive integer"
    Dd, Dc = 10, 10
    # Add a mixed linear constraint to ensure physical validity of thermal relaxation:
    #   T2 <= 2*T1  where T1=100µs*(0.5+u6), T2=100µs*(0.5+u7)  => (0.5+u7) <= 2*(0.5+u6)
    #   which simplifies to: u7 - 2*u6 <= 0.5  (linear in the unit box variables)
    A = _t.zeros(1, Dd + Dc, dtype=_t.double)
    A[0, Dd + 7] = 1.0    # coefficient for u7
    A[0, Dd + 6] = -2.0   # coefficient for u6
    b = _t.tensor([0.5], dtype=_t.double)
    lin_cons = LinearConstraint(A=A, b=b)
    space = MixedSpace(DiscreteSpec(Dd=Dd), ContinuousSpec(Dc=Dc), lin_cons=lin_cons)

    def f(xd_bits, xc_unit) -> float:
        # --- validate shapes
        if xd_bits.numel() != space.disc.Dd:
            raise ValueError(f"xd_bits must have {space.disc.Dd} elements, got {xd_bits.numel()}")
        if xc_unit.numel() != space.cont.Dc:
            raise ValueError(f"xc_unit must have {space.cont.Dc} elements, got {xc_unit.numel()}")

        # --- decode discrete (binary -> layout, routing)
        b = [int(bool(x)) for x in xd_bits.tolist()]
        layout = _decode_layout_from_8bits(b[:8])           # 8 bits -> permutation of [0,1,2,3]
        routing = _decode_routing_from_2bits(b[8], b[9])    # 2 bits -> routing method

        # --- quick feasibility checks: continuous in [0,1]
        try:
            from MIVABO import InfeasibleError
        except Exception:
            InfeasibleError = Exception
        if any((float(u) < 0.0 or float(u) > 1.0) for u in xc_unit.tolist()):
            raise InfeasibleError("continuous variables must be in [0,1]")

        # --- enforce mixed linear constraint here as well (A @ [xd; xc] <= b)
        # This ensures any infeasible evaluation raises inside the objective
        try:
            if getattr(space, 'lin_cons', None) is not None:
                A = space.lin_cons.A
                b_lin = space.lin_cons.b
                import torch as _tt
                xd_vec = xd_bits.to(dtype=_tt.double)
                xc_vec = xc_unit.to(dtype=_tt.double)
                vec = _tt.cat([xd_vec, xc_vec])
                if (_tt.matmul(A, vec) > b_lin + 1e-12).any():
                    raise InfeasibleError('mixed linear constraint violated')
        except Exception as e:
            if isinstance(e, InfeasibleError):
                raise
            # bubble up unexpected issues
            raise

        # --- get cached structure-only compiled circuit (no backend to avoid warnings)
        qc_struct = _compile_structure_only(marked, tuple(layout), routing)

        # --- decode continuous -> noise model & durations
        xc = [float(u) for u in xc_unit.tolist()]
        noise_model, dur_1q, dur_2q, idle_buf, _ = _make_noise_model_from_continuous(xc)

        # --- simulate with noise (qasm-style, sampling)
        backend = _BACKEND
        backend.set_options(noise_model=noise_model, seed_simulator=777)

        qc_meas = qc_struct.copy()           # don't mutate cached object
        qc_meas.measure_all()

        result = backend.run(qc_meas, shots=shots).result()
        counts = result.get_counts()
        # Probability estimate of the marked bitstring
        p_marked = counts.get(marked, 0) / float(shots)

        # --- makespan proxy based on depth and dominant durations + idle buffer
        depth = qc_struct.depth()
        makespan_proxy = depth * max(dur_2q, dur_1q) + depth * idle_buf

        return float((1.0 - p_marked) + lam * makespan_proxy)

    return space, f
