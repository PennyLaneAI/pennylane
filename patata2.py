import numpy as np
import pennylane as qp
from pennylane.labs.templates.alias_sampling import uniform_prep_ops, alias_sampling, alias_sampling_wires

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

L=5
mu = 4
rng = np.random.default_rng(L * 13 + 1)
w = rng.random(L) + 0.05

w = w/ np.sum(w)
print(w)
print(np.sum(w))
req = alias_sampling_wires(L, mu)
n = req["target_wires"] + req["temp_wires"] + req["work_wires"]
wires, temp, work = np.split(
    np.arange(n), np.cumsum([req["target_wires"], req["temp_wires"]])
)

dev = qp.device("sparse.qubit", wires=n)
print("wires", n)

@qp.qnode(dev)
def circuit():
    alias_sampling(w, mu, wires, temp, work)
    return qp.probs(wires=wires)

output = circuit()
output = dict_to_vector(output.coefs_dic)

print(output)
print(np.sum(output))