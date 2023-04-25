from jax.config import config

config.update("jax_enable_x64", True)
del config


import pennylane as qml
import jax
import jax.numpy as jnp

dev = qml.device("lightning.qubit", wires=2)

px = qml.PauliX(1)

def raw_circuit(x, y):
    qml.RX(x, wires=0)
    qml.RY(y, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(0.133, wires=1)
    return qml.expval(qml.PauliZ(wires=[0]))

#circuit = qml.qnode(device=dev, interface="jax", cache=False)(raw_circuit)
vv = []
circuit = qml.qnode(device=dev, cache=False)(raw_circuit)
def rcircuit(x,y):
    vv.append(x)
    vv.append(y)
    return circuit(x,y)

x = jnp.array(0.432)
y = jnp.array(0.543)

fun = rcircuit

gfun = jax.grad(rcircuit)

print("=====")
fun(x, y)
print("ko")
print("=====")
#gfun(x,y)