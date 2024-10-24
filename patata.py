import pennylane as qml
from tests.ops.functions.test_map_wires import mapped_op

x_wires = [0, 1, 2]
y_wires = [3, 4, 5]
input_registers = [x_wires, y_wires]

output_wires = [6, 7, 8]
work_wires = [9,10]


def f(x, y):
    return x ** 2 + y

@qml.qnode(qml.device("default.qubit", shots = 1))
def circuit():
    # loading values for x and y
    qml.BasisEmbedding(3, wires=x_wires)
    qml.BasisEmbedding(2, wires=y_wires)

    # applying the polynomial
    qml.OutPoly(
        f,
        input_registers,
        output_wires,
        mod = 7,
        work_wires=work_wires
    )

    return qml.sample(wires=output_wires)

print(circuit())