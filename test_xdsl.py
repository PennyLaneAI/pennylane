import pennylane as qml
from pennylane.ftqc import RotXZX
from pennylane.compiler.python_compiler.transforms import (
    decompose_graph_state_pass,
    convert_to_mbqc_formalism_pass,
    measurements_from_samples_pass,
)

import catalyst
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from catalyst.ftqc.pipelines import mbqc_pipeline

dev = qml.device("null.qubit", wires=2)

qml.capture.enable()

@qml.for_loop(0, 2, 1)
def loop_func(i):
    qml.H(i)
    qml.S(i)
    RotXZX(0.1, 0.2, 0.3, wires=[i])
    qml.RZ(phi=0.1, wires=[i])


@catalyst.qjit(
    target="mlir",
    pass_plugins=[getXDSLPluginAbsolutePath()],
    pipelines=mbqc_pipeline(),
    autograph=True,
    keep_intermediate=True,
)
@decompose_graph_state_pass
@convert_to_mbqc_formalism_pass
@measurements_from_samples_pass
@qml.set_shots(1000)
@qml.qnode(dev)
def circuit():
    qml.S(0)
    loop_func()
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(wires=0))

result = circuit()
print(result)
#print(circuit.mlir)