import numpy as np
import pennylane as qml

from pennylane.labs.templates import alias_sampling, alias_sampling_wires


L = 5
mu = 3

w = np.array([0.1, 0.3, 0.2, 0.2, 0.2])

req = alias_sampling_wires(L, mu)
n = req["target_wires"] + req["temp_wires"] + req["work_wires"]
wires, temp, work = np.split(
    np.arange(n), np.cumsum([req["target_wires"], req["temp_wires"]])
)

dev = qml.device("lightning.qubit", wires=n)

@qml.qnode(dev)
def circuit():
    alias_sampling(w, mu, wires, temp, work)
    return qml.probs(wires=wires)

print(circuit())
"""
#output = np.asarray(circuit())
output = circuit()

output = dict_to_vector(output.coefs_dic)
print(output)
print(w)

print(np.linalg.norm(output, ord=1))
plt.plot(output)
plt.plot(w/np.linalg.norm(w, ord=1))

plt.show()
"""