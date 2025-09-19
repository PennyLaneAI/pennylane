import pennylane as qml

import xdsl # = pytest.importorskip("xdsl")
import catalyst #= pytest.importorskip("catalyst")

from catalyst.ftqc import mbqc_pipeline
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    ConvertToMBQCFormalismPass,
    convert_to_mbqc_formalism_pass,
    decompose_graph_state_pass,
    measurements_from_samples_pass,
)

dev = qml.device("null.qubit", wires=2)
qml.capture.enable()

@qml.qjit(
    target="mlir",
    pass_plugins=[getXDSLPluginAbsolutePath()],
    pipelines=mbqc_pipeline(),
    autograph=True,
    keep_intermediate=True,
)
@decompose_graph_state_pass
@convert_to_mbqc_formalism_pass
@measurements_from_samples_pass
@qml.set_shots(2)
@qml.qnode(dev)
def circuit():
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(wires=0))

res = circuit()
print(res)