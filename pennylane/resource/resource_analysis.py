# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Alias transform for Catalyst's ``resource-analysis`` compilation pass."""

from pennylane.core.transforms import transform


def resource_analysis_setup_inputs(output_fname: str, output_json: bool = True):
    r"""Performs resource analysis on the MLIR representation and outputs the results to a json file.

    See :func:`~.analyze` for the fully integrated feature. This pass is an internal tool.

    Args:
        qnode (QNode): A quantum circuit.
        output_fname (str): Filename to write the JSON resource report to when ``output_json`` is
            ``True``.
        output_json (bool): If ``True``, write resource counts as JSON. Defaults to ``True``.


    **Example**

    .. code-block:: python

        @qp.qjit(capture=True, target="mlir")
        @qp.resource.resource_analysis("my_file.json")
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.expval(qp.Z(0))

    Compiling ``circuit`` runs the ``resource-analysis`` pass and places the output in ``"my_file.json"``.

    Passing ``output_fname=""`` will write the output to llvm's stdout, and it can be viewed by setting
    the ``verbose=True`` option on :func:`~.qjit`.


    """
    if not isinstance(output_json, bool):
        raise ValueError(f"output_json must be of type bool. Got {output_json!r}.")
    if not isinstance(output_fname, str):
        raise ValueError(f"output_fname must be of type str. Got {output_fname!r}.")
    return (), {"output_json": output_json, "output_fname": output_fname}


resource_analysis = transform(
    pass_name="resource-analysis", setup_inputs=resource_analysis_setup_inputs
)
