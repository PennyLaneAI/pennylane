import pennylane as qml
from pennylane.operation import Tensor
import numpy as np

A = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

H1 = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.Hermitian(A, [0, 1]) @ qml.PauliX(2)])
H2 = qml.Hamiltonian([1, 1], [qml.PauliZ(0) @ qml.Identity(1), qml.Hermitian(A, [0, 1]) @ qml.PauliX(2)])

H = qml.Hamiltonian([1, 1], [qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(0) @ qml.Identity(1)])

H3 = qml.Hamiltonian([1], [qml.PauliZ(0)])


def equals(H1, H2):

    H1 = zip(*H1.terms)
    H2 = zip(*H2.terms)

    H1_terms = set()
    H2_terms = set()

    for co, op in H1:
        obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
        tensor = []
        for ob in obs:
            parameters = tuple(param.tostring() for param in ob.parameters)
            tensor.append((ob.name, ob.wires, parameters))
        H1_terms.add((co, frozenset(tensor)))

    for co, op in H2:
        obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
        tensor = []
        for ob in obs:
            parameters = tuple(param.tostring() for param in ob.parameters)
            tensor.append((ob.name, ob.wires, parameters))
        H2_terms.add((co, frozenset(tensor)))
    
    return H1_terms == H2_terms

print(3 * Tensor(qml.PauliX(0), qml.PauliZ(1)) + Tensor(qml.PauliZ(1), qml.PauliX(0)))
print((H1 + H2).terms)
print(H3 == Tensor(qml.PauliX(0)))

print(3 * qml.PauliX(0) + Tensor(qml.PauliZ(1)))

print(type(Tensor(qml.PauliZ(0))))
print(type(Tensor(qml.PauliZ(0)).prune()))

'''
- Take into account identities placed in tensor products
- Take into account the fact that addition commutes
- Take into account the fact that tensor products commute
- Take into account the fact that Hermitians can be used as terms, or as part of tensor products (have the same "name")
- Take into account that a single obs = obs tensor one or multiple identities
'''

'''
op_attributes = []

        for op in ops:
            name = op.name if isinstance(op.name, list) else [op.name]

            if "Hermitian" not in name:
                op_attributes.append(set(zip(name, op.wires)))

        new_coeffs = []
        new_ops = []

        for coeff, op in zip(H.coeffs, H.ops):
            name = op.name if isinstance(op.name, list) else [op.name]

            if "Hermitian" in name:
                new_coeffs.append(coeff)
                new_ops.append(op)

            else:
                attr = set(zip(name, op.wires))

                if attr in op_attributes:
                    coeffs[op_attributes.index(attr)] += coeff
                else:
                    coeffs.append(coeff)
                    ops.append(op)
                    op_attributes.append(attr)

        for i, c in enumerate(coeffs):
            if not np.allclose([c], [0]):
                new_coeffs.append(c)
                new_ops.append(ops[i])

        return qml.Hamiltonian(new_coeffs, new_ops)
'''
