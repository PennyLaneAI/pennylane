import pennylane as qml
import jax

print("input", type(jax.numpy.array([1,0,0])))
op = qml.BasisState(jax.numpy.array([1,0,0]), wires=range(3))
print("output", type(op.state_vector()))


