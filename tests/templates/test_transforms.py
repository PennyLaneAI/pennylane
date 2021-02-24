import numpy as np
from pennylane.templates.transforms import adjoint
import pennylane as qml

def test_adjoint_sanity_check():

	dev = qml.device("default.qubit", wires=1)

	def my_op():
		qml.RX(0.123, wires=0)
		qml.RY(2.32, wires=0)
		qml.RZ(1.95, wires=0)

	@qml.qnode(dev)
	def my_circuit():
		qml.PauliX(wires=0)
		my_op()
		adjoint(my_op)()
		return qml.state()

	np.testing.assert_allclose(my_circuit(), np.array([0.0, 1.0]), atol=1e-6, rtol=1e-6)

def test_adjoint_directly_on_op():

	dev = qml.device("default.qubit", wires=1)
	@qml.qnode(dev)
	def my_circuit():
		qml.RX(0.123, wires=0)
		adjoint(qml.RX)(0.123, wires=0)
		return qml.state()

	np.testing.assert_allclose(my_circuit(), np.array([1.0, 0.0]))