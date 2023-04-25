import pennylane as qml
import jax
import jax.numpy as jnp

dev = qml.device("default.qubit", wires=2)

px = qml.PauliX(1)

def raw_circuit(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(0.133, wires=1)
    return qml.expval(qml.PauliZ(wires=[0]))

with qml.queuing.AnnotatedQueue() as queue:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(0.133, wires=1)
    qml.expval(qml.PauliZ(wires=[0]))

with qml.queuing.AnnotatedQueue() as queue2:
    qml.RX(0.432, wires=0)
    qml.RY(0.543, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(0.133, wires=1)
    qml.expval(qml.PauliZ(wires=[0]))

tp=qml.tape.QuantumScript.from_queue(queue)
tp2=qml.tape.QuantumScript.from_queue(queue2)

dev.execute(tp)
dev.execute(tp2)

l, s = jax.tree_util.tree_flatten(tp)
l2, s2 = jax.tree_util.tree_flatten(tp2)

e1 = qml.expval(qml.PauliZ(wires=[0]))
e2 = qml.expval(qml.PauliZ(wires=[0]))

circuit = qml.qnode(device=dev, interface="jax", cache=False)(raw_circuit)
circuit = qml.qnode(device=dev, cache=False)(raw_circuit)

x = jnp.array(0.432)
y = jnp.array(0.543)

jcircuit = jax.jit(circuit)

circuit(x, y)
jcircuit(x, y)
