import matplotlib.pyplot as plt

import pennylane as qml


def example():
    tape = qml.tape.QuantumTape([qml.Hadamard(0), qml.CNOT([0, 1]), qml.CNOT([0, 2])], [qml.probs(wires=[0, 1, 2])])

    tapes, processing_fn = qml.drawer.plot_visualize(tape, qml.drawer.ProbHistogram)
    # tapes, processing_fn = qml.drawer.plot_visualize(tape, qml.drawer.WignerFunction)
    # tapes, processing_fn = qml.drawer.plot_animate(tape, qml.drawer.ProbHistogram)
    # tapes, processing_fn = qml.drawer.plot_animate(tape, qml.drawer.WignerFunction)

    dev = qml.devices.experimental.DefaultQubit2()
    res = dev.execute(tapes)
    res = processing_fn(res)

    plt.show()


def main():
    example()


if __name__ == '__main__':
    main()
