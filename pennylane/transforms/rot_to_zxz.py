"""
A transform for decomposing "Rot" gates into [RZ, RX, RZ] sequence.
"""
import pennylane as qml
from pennylane.numpy import pi


@qml.qfunc_transform
def rot_to_zxz(tape):
    r"""Quantum function transform to decompose :class:`~.Rot`
    gates into the sequence ``[RZ(a), RX(b), RZ(c)]``.

    "Rot" gates implement the sequence [RZ(theta), RY(phi), RZ(omega)]. Using
    the equality RZ(pi/2)RX(alpha)RZ(-pi/2)=RY(alpha) valid for every alpha,
    the "Rot" gates is replaced by [RZ(theta+pi/2), RX(phi), RZ(omega-pi/2)].

    Args:
        tape (function): a quantum function

    **Example**

    Suppose that we would like to decompose all the "Rot" gates in the following
    circuit:

    .. code-block:: python3

        def circuit(angles0, angles1):
            phi0, theta0, omega0 = [a for a in angles0]
            phi1, theta1, omega1 = [a for a in angles1]
            qml.Rot(phi0, theta0, omega0,wires=0)
            qml.Rot(phi1, theta1, omega1,wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    The ``rot_to_zxz`` performs this decomposition by also preserving the
    differentiability.

    For angles0=[0.01, 0.2, 1.5], angles1=[1.2, 0.9, 0.7] the original circuit
    is:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> qnode = qml.QNode(circuit, dev)
    >>> print(qml.draw(qnode)([0.01, 0.2, 1.5], [1.2, 0.9, 0.7]))
     0: ──Rot(0.01, 0.2, 1.5)──╭C──┤ ⟨Z⟩
     1: ──Rot(1.2, 0.9, 0.7)───╰X──┤

    The transform returns the following circuit:

    >>> transformed_circuit = rot_to_zxz(circuit)
    >>> transformed_qnode = qml.QNode(transformed_circuit, dev)
    >>> print(qml.draw(transformed_qnode)([0.01, 0.2, 1.5], [1.2, 0.9, 0.7]))
     0: ──RZ(1.58)──RX(0.2)──RZ(-0.0708)──╭C──┤ ⟨Z⟩
     1: ──RZ(2.77)──RX(0.9)──RZ(-0.871)───╰X──┤

    This decomposition is fully differentiable. We can differentiate
    with respect to input QNode parameters when they are being used in the
    "Rot" gates being decomposed. For example, for the circuit above we
    have:

    >>> grad0 = qml.grad(transformed_qnode, argnum=[0])
    >>> grad1 = qml.grad(transformed_qnode, argnum=[1])
    >>> print("grad0:", grad0([0.01, 0.2, 1.5], [1.2, 0.9, 0.7]))
    >>> print("grad1:", grad1([0.01, 0.2, 1.5], [1.2, 0.9, 0.7]))
    grad0: ([array(-1.66533454e-16), array(-0.19866933), array(-4.16337e-17)],)
    grad1: ([array(-2.77555756e-17), array(0.), array(-5.55111512e-17)],)

    We can also differentiate when the input parameters are not used in the
    "Rot":


    >>> def circuit(angles):
    ...     qml.RX(angles[0],wires=0)
    ...     qml.RX(angles[1],wires=1)
    ...     qml.Rot(0.01, 0.2, 1.5,wires=0)
    ...     qml.Rot(1.2, 0.9, 0.7,wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> dev = qml.device('default.qubit', wires=2)
    >>> transformed_qnode = qml.QNode(rot_to_zxz(circuit), dev)
    >>> grad = qml.grad(transformed_qnode, argnum=0)
    >>> print("grad:", grad([3.1, 2.7]))
    grad: [array(-0.04273676), array(-8.32667268e-17)]
    """
    for op in tape.operations + tape.measurements:
        if op.name == "Rot":
            wire = op.wires
            angles = op.single_qubit_rot_angles()
            qml.RZ(angles[0] + pi / 2, wires=wire)
            qml.RX(angles[1], wires=wire)
            qml.RZ(angles[2] - pi / 2, wires=wire)
        else:
            qml.apply(op)
