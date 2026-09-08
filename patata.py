import numpy as np
import pennylane as qml
from pennylane.labs.templates import SuperpositionTHC, alias_sampling_thc
import matplotlib
import matplotlib.pyplot as plt

"""Extract SIGNED amplitudes over mu_wires + nu_wires from the THC PREPARE circuit."""
import numpy as np


def group_signed_amplitudes(state, total_wires, index_wires, tol=1e-10):
    """Collapse a full statevector onto `index_wires`, keeping the sign.

    Works because the reduced density matrix over the index register is exactly
    diagonal (verified numerically), and every ancilla branch belonging to a given
    index shares one common phase. So each index has a well-defined real amplitude
    up to one arbitrary GLOBAL sign.

    Args:
        state: dense complex array of length 2**total_wires, OR a dict
               {full_bitstring: amplitude} (as returned by sparse.qubit's
               SparseState.coefs_dic).
        total_wires (int): number of wires in the device.
        index_wires (list[int]): mu_wires + nu_wires, in that order.

    Returns:
        (amps, probs, phase_spread)
        amps: signed real amplitudes, length 2**len(index_wires)
        probs: |amps|**2
        phase_spread: max phase disagreement inside a single index (should be ~0;
                      if it is not, the grouping is meaningless and you must
                      inspect the ancillas instead)
    """
    k = len(index_wires)
    nidx = 2 ** k

    if isinstance(state, dict):
        # sparse path: bucket the sparse amplitudes by index bitstring
        buckets = [[] for _ in range(nidx)]
        for bitstring, amp in state.items():
            if abs(amp) <= tol:
                continue
            key = "".join(bitstring[w] for w in index_wires)
            buckets[int(key, 2)].append(complex(amp))
        rows = [np.array(b) for b in buckets]
    else:
        psi = np.asarray(state).reshape([2] * total_wires)
        rest = [w for w in range(total_wires) if w not in set(index_wires)]
        Tm = np.transpose(psi, list(index_wires) + rest).reshape(nidx, -1)
        rows = [Tm[l][np.abs(Tm[l]) > tol] for l in range(nidx)]

    amps = np.zeros(nidx)
    probs = np.zeros(nidx)
    spread = 0.0
    for l, nz in enumerate(rows):
        if len(nz) == 0:
            continue
        probs[l] = float((np.abs(nz) ** 2).sum())
        ph = np.angle(nz)
        # phases inside one index must agree modulo 2*pi
        spread = max(spread, float(np.abs(np.angle(np.exp(1j * (ph - ph[0])))).max()))
        amps[l] = np.sqrt(probs[l]) * np.sign(np.cos(ph[0]))
    return amps, probs, spread


def signed_target(M, N, zeta, t_ell, n):
    """Signed amplitude target: sign(weight) * sqrt(normalized |weight|)."""
    size = 2 ** n
    A = np.zeros((size, size))
    S = np.ones((size, size))
    for mu in range(M):
        for nu in range(M):
            A[mu, nu] = abs(zeta[mu, nu]) / 2.0          # /2 from the symmetrization
            S[mu, nu] = np.sign(zeta[mu, nu]) or 1.0
    for ell in range(N // 2):
        A[ell, M] = abs(t_ell[ell])                      # sentinel column, no /2
        S[ell, M] = np.sign(t_ell[ell]) or 1.0
    A = A / A.sum()
    return (S * np.sqrt(A)).reshape(-1), A.reshape(-1)


def fix_global_sign(amps, target_amps, target_probs):
    """The circuit's overall sign is gauge. Pin it on the heaviest state."""
    ref = int(np.argmax(target_probs))
    if amps[ref] * target_amps[ref] < 0:
        return -amps
    return amps


M, N, n, aleph = 3, 2, 2, 6
np.random.seed(129)
zeta = np.random.randn(M, M)
zeta = (zeta + zeta.T) / 2
t_ell = np.random.randn(N // 2)

print("zeta", zeta)
print("t_ell", t_ell)

mu_wires = list(range(n))
nu_wires = list(range(n, 2 * n))

# SuperpositionTHC prepares the uniform superposition and the one-body flag.
sup_work = list(range(2 * n, 2 * n + 3 * n + 5))
edge_flag = sup_work[3]  # nu register in state |M>

# SuperpositionTHC returns every work wire to |0> except its flags at
# indices 0, 3 and 6, so the rest are reused as alias-sampling scratch.
clean = [w for i, w in enumerate(sup_work) if i not in (0, 3, 6)]

n_d = int(np.ceil(np.log2(N // 2 + M * (M + 1) / 2))) + 1
num_work = n_d + 2 * n + 3 * aleph + 4
fresh = list(range(sup_work[-1] + 1, sup_work[-1] + 1 + max(0, num_work - len(clean))))
work_wires = (clean + fresh)[:num_work]

def dict_to_vector(d):
    if not d:
        return np.array([])

    # Obtener n (longitud del bitstring) del primer elemento
    sample_key = next(iter(d))
    n = len(sample_key)

    # Crear un vector de ceros de tamaño 2**n
    vector = np.zeros(2 ** n, dtype=np.float64)

    # Asignar cada valor en el índice decimal correspondiente
    for bitstring, val in d.items():
        idx = int(bitstring, 2)  # Convierte p. ej. '0110' a 6
        vector[idx] = val
    return vector

dev = qml.device("sparse.qubit", wires=max(mu_wires + nu_wires + sup_work + work_wires) + 1)
print(dev.wires)
@qml.qnode(dev)
def circuit():
    SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
    alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
    #return qml.state()
    return qml.probs(wires=mu_wires + nu_wires)

output = circuit()
print(output)
output = dict_to_vector(output.coefs_dic)
output = np.asarray(output)
print(output)
# ---- target: physical symmetric THC distribution over |mu>|nu> ----
# Two-body weight |zeta_{mu,nu}| symmetric in (mu,nu); one-body |t_ell| on sentinel col (ell, M).
size = 2 ** n
P = np.zeros((size, size))
for mu in range(M):
    for nu in range(M):
        P[mu, nu] += abs(zeta[mu, nu]) / 2.0
for ell in range(N // 2):
    P[ell, M] += abs(t_ell[ell])
P = P / P.sum()
target = P.reshape(-1)  # flatten to match probs(wires=mu_wires+nu_wires) ordering

print("output.shape:", output.shape, "target.shape:", target.shape)
print("output.sum():", float(output.sum()), "target.sum():", float(target.sum()))
print("max|output-target|:", float(np.max(np.abs(output - target))))
labels = [f"|{a}{b}>" for a in range(size) for b in range(size)]
for i,(o,t) in enumerate(zip(output, target)):
    if o > 1e-9 or t > 1e-9:
        print(f"  {labels[i]}: out={o:.4f} target={t:.4f}")

# ---- plot ----
print("error", np.linalg.norm(target - output))
x = np.arange(len(output))
plt.figure(figsize=(10, 5))
plt.plot(x, target, "o-", label="target (physical THC dist.)", linewidth=2)
plt.plot(x, output, "s--", label=f"alias_sampling_thc output (aleph={aleph})", linewidth=2)
plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
plt.xlabel(r"basis state $|\mu\rangle|\nu\rangle$")
plt.ylabel("probability")
plt.title(f"THC PREPARE: target vs prepared  (M={M}, N={N}, n={n}, aleph={aleph})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()