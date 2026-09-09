"""
THC PREPARE: signed-amplitude check for SuperpositionTHC + alias_sampling_thc.

Replaces the probability-only version. Two differences that matter:

  1. The target weights the two-body block with |zeta| / 2, not |zeta|. The
     symmetrization step at the end of alias_sampling_thc splits every
     off-diagonal entry 50/50 between |mu,nu> and |nu,mu>, and _build_thc_pairs
     already halves the diagonal, so EVERY two-body cell ends up with
     |zeta[mu,nu]| / 2. The sentinel column (nu = M) is excluded from the
     symmetrization and keeps the full |t_ell|.

  2. It reads qp.state() instead of qp.probs() and recovers the SIGN of each
     amplitude. This is legitimate here because the reduced density matrix over
     mu_wires + nu_wires is exactly diagonal, and all ancilla branches belonging
     to one index share a single common phase (0 or pi). The script asserts that
     second property instead of assuming it.

The overall sign is gauge (SuperpositionTHC deliberately leaves work_wires[0]
uncleaned, and there is a GlobalPhase(pi) in its decomposition), so it is pinned
on the heaviest basis state before comparing.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # comment out if you want an interactive window
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane.labs.templates import SuperpositionTHC, alias_sampling_thc

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
M = 5          # THC rank
N = 4          # number of spin orbitals (requires N // 2 <= M + 1)
ALEPH = 5      # bits of the keep register; discretization error ~ 2 ** -ALEPH
SEED = 3
DEVICE = "sparse.qubit"   # or "default.qubit" for small cases
OUTFILE = "thc_signed_amplitudes.png"

# ----------------------------------------------------------------------------
# Wire layout
# ----------------------------------------------------------------------------
# n is FIXED by M: alias_sampling_thc requires exactly ceil(log2(M + 1)) wires
# per index register, so that the sentinel value M fits and nothing more.
n = int(np.ceil(np.log2(M + 1)))
d = N // 2 + M * (M + 1) // 2
n_d = int(np.ceil(np.log2(d))) + 1

if N // 2 > M + 1:
    raise ValueError("N // 2 must be <= M + 1")

mu_wires = list(range(n))
nu_wires = list(range(n, 2 * n))

# SuperpositionTHC needs 3n + 5 work wires. It returns all of them to |0>
# except indices 0 (dirty), 3 (nu == M flag) and 6 (success flag), so the rest
# are recycled as alias-sampling scratch.
sup_work = list(range(2 * n, 2 * n + 3 * n + 5))
edge_flag = sup_work[3]
clean = [w for i, w in enumerate(sup_work) if i not in (0, 3, 6)]

num_work = n_d + 2 * n + 3 * ALEPH + 4
fresh = list(range(sup_work[-1] + 1, sup_work[-1] + 1 + max(0, num_work - len(clean))))
work_wires = (clean + fresh)[:num_work]

total_wires = max(mu_wires + nu_wires + sup_work + work_wires) + 1
index_wires = mu_wires + nu_wires

# The amplitude amplification inside SuperpositionTHC only reaches success
# probability 1 when the valid pairs are at least a quarter of all index
# combinations. Otherwise there is leftover weight in the garbage subspace and
# you must post-select on sup_work[6] == 1 instead of tracing over it.
exact_superposition = d >= 2 ** (2 * n) / 4

print(f"M={M} N={N} n={n} aleph={ALEPH}  d={d}  n_d={n_d}")
print(f"total wires = {total_wires}   (index wires {index_wires})")
print(f"SuperpositionTHC success prob == 1 : {exact_superposition}"
      f"   (d={d} vs 2^(2n)/4={2 ** (2 * n) / 4:g})")
if not exact_superposition:
    print("  WARNING: post-select on sup_work[6] == 1, the trace-out below is not valid")

# ----------------------------------------------------------------------------
# Coefficients
# ----------------------------------------------------------------------------
np.random.seed(SEED)
zeta = np.random.randn(M, M)
zeta = (zeta + zeta.T) / 2      # alias_sampling_thc only reads the upper triangle
t_ell = np.random.randn(N // 2)


# ----------------------------------------------------------------------------
# Signed grouping over the index register
# ----------------------------------------------------------------------------
def group_signed_amplitudes(state, total_wires, index_wires, tol=1e-10):
    """Collapse a full state onto index_wires, keeping the sign.

    Accepts either a dense complex array of length 2 ** total_wires, or a dict
    {full_bitstring: amplitude} such as sparse.qubit's SparseState.coefs_dic.

    Returns (amps, probs, phase_spread). phase_spread is the largest phase
    disagreement found inside a single index; it must be ~0 for the signed
    amplitude to mean anything. Do NOT sum the raw amplitudes: different ancilla
    branches are orthogonal, so summing them is not a physical operation.
    """
    k = len(index_wires)
    nidx = 2 ** k

    if isinstance(state, dict):
        buckets = [[] for _ in range(nidx)]
        for bitstring, amp in state.items():
            if abs(amp) <= tol:
                continue
            if len(bitstring) <= max(index_wires):
                raise ValueError(
                    f"bitstring length {len(bitstring)} does not cover index wire "
                    f"{max(index_wires)}; the device is not labelling wires 0..n-1"
                )
            key = "".join(bitstring[w] for w in index_wires)
            buckets[int(key, 2)].append(complex(amp))
        rows = [np.array(b) for b in buckets]
    else:
        psi = np.asarray(state).reshape([2] * total_wires)
        rest = [w for w in range(total_wires) if w not in set(index_wires)]
        flat = np.transpose(psi, list(index_wires) + rest).reshape(nidx, -1)
        rows = [flat[l][np.abs(flat[l]) > tol] for l in range(nidx)]

    amps = np.zeros(nidx)
    probs = np.zeros(nidx)
    spread = 0.0
    for l, nz in enumerate(rows):
        if len(nz) == 0:
            continue
        probs[l] = float((np.abs(nz) ** 2).sum())
        ph = np.angle(nz)
        spread = max(spread, float(np.abs(np.angle(np.exp(1j * (ph - ph[0])))).max()))
        amps[l] = np.sqrt(probs[l]) * np.sign(np.cos(ph[0]))
    return amps, probs, spread


def signed_target(M, N, zeta, t_ell, n):
    """sign(w) * sqrt(|w| / sum|w|) over the |mu>|nu> basis."""
    size = 2 ** n
    A = np.zeros((size, size))
    S = np.ones((size, size))
    for mu in range(M):
        for nu in range(M):
            A[mu, nu] = abs(zeta[mu, nu]) / 2.0     # /2 from the symmetrization
            S[mu, nu] = np.sign(zeta[mu, nu]) or 1.0
    for ell in range(N // 2):
        A[ell, M] = abs(t_ell[ell])                 # sentinel column, no /2
        S[ell, M] = np.sign(t_ell[ell]) or 1.0
    A = A / A.sum()
    return (S * np.sqrt(A)).reshape(-1), A.reshape(-1)


def fix_global_sign(amps, target_amps, target_probs):
    """The circuit's overall sign is unobservable; pin it on the heaviest state."""
    ref = int(np.argmax(target_probs))
    return -amps if amps[ref] * target_amps[ref] < 0 else amps


# ----------------------------------------------------------------------------
# Circuit
# ----------------------------------------------------------------------------
dev = qml.device(DEVICE, wires=total_wires)


@qml.qnode(dev)
def circuit():
    SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
    alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, ALEPH)
    return qml.state()


raw = circuit()
state = raw.coefs_dic if hasattr(raw, "coefs_dic") else raw

amps, probs, spread = group_signed_amplitudes(state, total_wires, index_wires)
target_amps, target_probs = signed_target(M, N, zeta, t_ell, n)
amps = fix_global_sign(amps, target_amps, target_probs)

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
support = target_probs > 1e-12
print(f"\nphase spread inside each index = {spread:.3e}   (must be ~0)")
if spread > 1e-6:
    print("  WARNING: phases disagree inside an index. The signed amplitude is NOT")
    print("  well defined; check that no ancilla was truncated or left dirty.")
print(f"probs.sum() = {probs.sum():.6f}   target.sum() = {target_probs.sum():.6f}")
print(f"signs all match : {bool(np.all(np.sign(amps[support]) == np.sign(target_amps[support])))}")
print(f"max|amp - target_amp|  = {np.max(np.abs(amps - target_amps)):.3e}"
      f"   (discretization bound 2^-aleph = {2.0 ** -ALEPH:.2e})")
print(f"max|prob - target_prob| = {np.max(np.abs(probs - target_probs)):.3e}")

size = 2 ** n
labels = [f"|{a}{b}>" for a in range(size) for b in range(size)]
print(f"\n{'state':8s} {'amp(out)':>10s} {'amp(tgt)':>10s} {'p(out)':>10s} {'p(tgt)':>10s}")
for i in range(len(amps)):
    if abs(amps[i]) > 1e-9 or abs(target_amps[i]) > 1e-9:
        print(f"{labels[i]:8s} {amps[i]:10.4f} {target_amps[i]:10.4f}"
              f" {probs[i]:10.4f} {target_probs[i]:10.4f}")

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
x = np.arange(len(amps))
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax0.axhline(0, color="0.6", lw=1)
ax0.plot(x, target_amps, "o-", label="target signed amplitude", lw=2)
ax0.plot(x, amps, "s--", label=f"circuit signed amplitude (aleph={ALEPH})", lw=2)
ax0.set_ylabel("amplitude (signed)")
ax0.set_title(f"THC PREPARE  (M={M}, N={N}, n={n}, aleph={ALEPH})")
ax0.legend()
ax0.grid(alpha=0.3)

ax1.plot(x, target_probs, "o-", label="target probability", lw=2)
ax1.plot(x, probs, "s--", label="circuit probability", lw=2)
ax1.set_ylabel("probability")
ax1.set_xlabel(r"basis state $|\mu\rangle|\nu\rangle$")
ax1.legend()
ax1.grid(alpha=0.3)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(OUTFILE, dpi=140)
print(f"\nplot written to {OUTFILE}")
