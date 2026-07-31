import numpy as np
import pennylane as qml
from pennylane.labs.templates import SuperpositionTHC, alias_sampling_thc
import matplotlib
import matplotlib.pyplot as plt

M,N,n,aleph=6,6,3,2
np.random.seed(3)
zeta=np.random.randn(M,M); zeta=(zeta+zeta.T)/2
t_ell=np.random.randn(N//2)

mu_wires=list(range(n)); nu_wires=list(range(n,2*n))
sup_work=list(range(2*n,2*n+3*n+5)); edge_flag=sup_work[3]
clean=[w for i,w in enumerate(sup_work) if i not in (0,3,6)]
n_d=int(np.ceil(np.log2(N//2+M*(M+1)/2)))+1
num_work=n_d+2*n+3*aleph+4
fresh=list(range(sup_work[-1]+1,sup_work[-1]+1+max(0,num_work-len(clean))))
work_wires=(clean+fresh)[:num_work]
dev=qml.device("lightning.qubit",wires=max(mu_wires+nu_wires+sup_work+work_wires)+1)

@qml.qnode(dev)
def circuit():
    SuperpositionTHC(M,N,mu_wires,nu_wires,sup_work)
    alias_sampling_thc(M,N,zeta,t_ell,mu_wires,nu_wires,edge_flag,work_wires,aleph)
    return qml.probs(wires=mu_wires+nu_wires)

output=np.asarray(circuit())

# CORRECT target: the distribution the alias-sampling routine encodes.
# The valid-pair table halves the two-body diagonal and normalizes by
# total_w = sum over unordered pairs (diag halved) + one-body 1-norm.
# Symmetrization then splits each two-body pair across (mu,nu) and (nu,mu);
# the one-body sentinel column is excluded from the swap (keeps full weight).
size=2**n
total_w=0.0
for nu in range(M):
    for mu in range(nu+1):
        w=abs(zeta[mu,nu]); w=w/2 if mu==nu else w
        total_w+=w
for ell in range(N//2):
    total_w+=abs(t_ell[ell])

target=np.zeros((size,size))
for mu in range(M):
    for nu in range(M):
        target[mu,nu]=abs(zeta[mu,nu])/(2*total_w)   # split across both orderings
for ell in range(N//2):
    target[ell,M]=abs(t_ell[ell])/total_w            # one-body: full weight
target=target.reshape(-1)

print("output.sum",float(output.sum()),"target.sum",float(target.sum()))
print("max|output-target|",float(np.max(np.abs(output-target))))

labels=[f"|{a}{b}>" for a in range(size) for b in range(size)]
x=np.arange(len(output))
plt.figure(figsize=(10,5))
plt.plot(x,target,"o-",label="target (distribution encoded by the routine)",linewidth=2)
plt.plot(x,output,"s--",label=f"alias_sampling_thc output (aleph={aleph})",linewidth=2)
plt.xticks(x,labels,rotation=45,ha="right",fontsize=8)
plt.xlabel(r"basis state $|\mu\rangle|\nu\rangle$"); plt.ylabel("probability")
plt.title(f"THC PREPARE: corrected target vs prepared  (M={M}, N={N}, n={n}, aleph={aleph})")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()
