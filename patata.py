import numpy as np
import pennylane as qml
from pennylane.labs.templates import SuperpositionTHC, alias_sampling_thc
import matplotlib
import matplotlib.pyplot as plt

M, N, n, aleph = 3, 2, 2, 2
np.random.seed(3)
zeta = np.random.randn(M, M)
zeta = (zeta + zeta.T) / 2
t_ell = np.random.randn(N // 2)

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

dev = qml.device("lightning.qubit", wires=max(mu_wires + nu_wires + sup_work + work_wires) + 1)
print(dev.wires)
@qml.qnode(dev)
def circuit():
    SuperpositionTHC(M, N, mu_wires, nu_wires, sup_work)
    alias_sampling_thc(M, N, zeta, t_ell, mu_wires, nu_wires, edge_flag, work_wires, aleph)
    return qml.probs(wires=mu_wires + nu_wires)

output = np.asarray(circuit())

# ---- target: physical symmetric THC distribution over |mu>|nu> ----
# Two-body weight |zeta_{mu,nu}| symmetric in (mu,nu); one-body |t_ell| on sentinel col (ell, M).
size = 2 ** n
P = np.zeros((size, size))
for mu in range(M):
    for nu in range(M):
        P[mu, nu] += abs(zeta[mu, nu])
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
plt.savefig("/tmp/thc_target_vs_output.png", dpi=150)
plt.show()
