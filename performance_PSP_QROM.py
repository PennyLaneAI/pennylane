import pennylane as qml
import numpy as np
from time import time
import matplotlib.pyplot as plt


n = 4 # size of the embedded vector 2 ** n

vector_square = np.random.rand(2 ** n)
vector_square = vector_square / np.linalg.norm(vector_square, ord = 1)

limit_time = 6 # max time we run each circuit

dev_names = ["default.qubit", "lightning.qubit", "sparse.qubit"]
dic_performance = {name: []  for name in dev_names}


for name in dev_names:
    print()
    print(name, ":")
    print("-------------")

    stop_criteria = False

    for precision_wires in range(2,30):

        if stop_criteria:
            break

        print("precision wires: ", precision_wires)

        time1 = time()

        wires = qml.registers({"embedding_wires": n, "precision_wires": precision_wires})

        dev = qml.device(name, wires = n + precision_wires)

        @qml.qnode(dev)
        def circuit():

            qml.StatePrepQROM(vector = np.sqrt(vector_square),
                              embedding_wires = wires["embedding_wires"],
                              precision_wires = wires["precision_wires"])
        
            return qml.probs(wires = wires["embedding_wires"])

        circuit()

        value_time = time() - time1
        dic_performance[name].append(value_time)

        if value_time > limit_time:
            stop_criteria = True


for name in dev_names:
    plt.plot(dic_performance[name][1:], label = name)

plt.xlabel("precision wires")
plt.ylabel("time (s)")
plt.legend()
plt.show()
