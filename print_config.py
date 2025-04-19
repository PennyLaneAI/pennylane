"""Example script to print ExecutionConfig output for visual inspection."""

import pennylane as qml
from pennylane.workflow import construct_execution_config


@qml.qnode(qml.device("default.qubit", wires=1))
def circuit(x):
    qml.RX(x, 0)
    return qml.expval(qml.Z(0))


config = construct_execution_config(circuit, resolve=False)(1)

print(config)
