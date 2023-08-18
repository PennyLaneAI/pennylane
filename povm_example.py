import numpy as np
import pennylane as qml

qml.operation.enable_new_opmath()


P = qml.povm.POVM([
    (qml.Identity(wires=0) + qml.PauliZ(wires=0)) / 4,
    (qml.Identity(wires=0) - qml.PauliZ(wires=0)) / 4,
    (qml.Identity(wires=0) + qml.PauliX(wires=0)) / 4,
    (qml.Identity(wires=0) - qml.PauliX(wires=0)) / 4
], validate=True)


tape = qml.tape.QuantumTape([], [qml.probs(op=P)])
# tape = qml.tape.QuantumTape([], [qml.sample(op=P)], shots=1000)
tapes, processing_fn = qml.povm.povm_dilate(tape)

dev = qml.devices.experimental.DefaultQubit2()


res = qml.execute(tapes, dev, gradient_fn=None)
res = processing_fn(res)
print(res)


