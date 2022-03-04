# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains logic for the text based circuit drawer through the ``tape_text`` function.
"""

from multimethod import multimethod
from functools import wraps

import pennylane as qml
from pennylane.operation import Expectation, Probability, Sample, Variance, State

from .drawable_layers import drawable_layers
from .utils import convert_wire_order


def _add_grouping_symbols(op, layer_str, wire_map):
    """Adds symbols indicating the extent of a given object."""
    if len(op.wires) > 1:
        mapped_wires = [wire_map[w] for w in op.wires]
        min_w, max_w = min(mapped_wires), max(mapped_wires)
        layer_str[min_w] = "╭"
        layer_str[max_w] = "╰"

        for w in range(min_w + 1, max_w):
            layer_str[w] = "├" if w in mapped_wires else "│"

    return layer_str


def _add_op(op, layer_str, wire_map, decimals, cache):
    """Updates ``layer_str`` with ``op`` operation."""
    layer_str = _add_grouping_symbols(op, layer_str, wire_map)

    control_wires = op.control_wires
    for w in control_wires:
        layer_str[wire_map[w]] += "C"

    label = op.label(decimals=decimals, cache=cache).replace("\n", "")
    for w in op.wires:
        if w not in control_wires:
            layer_str[wire_map[w]] += label

    return layer_str


measurement_label_map = {
    Expectation: lambda label: f"<{label}>",
    Probability: lambda label: f"Probs[{label}]" if label else "Probs",
    Sample: lambda label: f"Sample[{label}]" if label else "Sample",
    Variance: lambda label: f"Var[{label}]",
    State: lambda label: "State",
}


def _add_measurement(m, layer_str, wire_map, decimals, cache):
    """Updates ``layer_str`` with the ``m`` measurement."""
    layer_str = _add_grouping_symbols(m, layer_str, wire_map)

    if m.obs is None:
        obs_label = None
    else:
        obs_label = m.obs.label(decimals=decimals, cache=cache).replace("\n", "")
    meas_label = measurement_label_map[m.return_type](obs_label)

    if len(m.wires) == 0:  # state or probability across all wires
        for i, s in enumerate(layer_str):
            layer_str[i] = s + meas_label

    for w in m.wires:
        layer_str[wire_map[w]] += meas_label
    return layer_str


# pylint: disable=too-many-arguments
@multimethod
def draw(
    circuit: qml.tape.QuantumTape,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    max_length=100,
    show_matrices=False,
    cache=None,
    expansion_strategy=None
):
    cache = cache or {}
    cache.setdefault("tape_offset", 0)
    cache.setdefault("matrices", [])
    tape_cache = []

    wire_map = convert_wire_order(
        circuit.operations + circuit.measurements, wire_order=wire_order, show_all_wires=show_all_wires
    )
    n_wires = len(wire_map)
    if n_wires == 0:
        return ""

    totals = [f"{wire}: " for wire in wire_map]
    line_length = max(len(s) for s in totals)
    totals = [s.rjust(line_length, " ") for s in totals]

    # Used to store lines that are hitting the maximum length
    finished_lines = []

    layers_list = [
        drawable_layers(circuit.operations, wire_map=wire_map),
        drawable_layers(circuit.measurements, wire_map=wire_map),
    ]
    add_list = [_add_op, _add_measurement]
    fillers = ["─", " "]
    enders = [True, False]  # add "─┤" after all operations

    for layers, add, filler, ender in zip(layers_list, add_list, fillers, enders):

        for layer in layers:
            layer_str = [filler] * n_wires

            for op in layer:
                if isinstance(op, qml.tape.QuantumTape):
                    layer_str = _add_grouping_symbols(op, layer_str, wire_map)
                    label = f"Tape:{cache['tape_offset']+len(tape_cache)}"
                    for w in op.wires:
                        layer_str[wire_map[w]] += label
                    tape_cache.append(op)
                else:
                    layer_str = add(op, layer_str, wire_map, decimals, cache)

            max_label_len = max(len(s) for s in layer_str)
            layer_str = [s.ljust(max_label_len, filler) for s in layer_str]

            line_length += max_label_len + 1  # one for the filler character
            if line_length > max_length:
                # move totals into finished_lines and reset totals
                finished_lines += totals
                finished_lines[-1] += "\n"
                totals = [filler] * n_wires
                line_length = 2 + max_label_len

            totals = [filler.join([t, s]) for t, s in zip(totals, layer_str)]
        if ender:
            totals = [s + "─┤" for s in totals]

    # Recursively handle nested tapes #
    tape_totals = "\n".join(finished_lines + totals)
    current_tape_offset = cache["tape_offset"]
    cache["tape_offset"] += len(tape_cache)
    for i, nested_tape in enumerate(tape_cache):
        label = f"\nTape:{i+current_tape_offset}"
        tape_str = draw(
            nested_tape,
            wire_order,
            show_all_wires,
            decimals,
            max_length,
            show_matrices=False,
            cache=cache,
        )
        tape_totals = "\n".join([tape_totals, label, tape_str])

    if show_matrices:
        mat_str = ""
        for i, mat in enumerate(cache["matrices"]):
            mat_str += f"\nM{i} = \n{mat}"
        return tape_totals + mat_str

    return tape_totals

@multimethod
def draw(
    circuit: qml.QNode,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    max_length=100,
    show_matrices=False,
    expansion_strategy=None,
    cache=None
):
    """Create a function that draws the given qnode.

     Args:
         circuit (qml.QNode/ qml.tape.QuantumTape): the input QNode or Quantum Tape that is to be drawn
         wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
         show_all_wires (bool): If True, all wires, including empty wires, are printed.
         decimals (int): How many decimal points to include when formatting operation parameters.
             ``None`` will omit parameters from operation labels.
         max_length (int): Maximum string width (columns) when printing the circuit
         show_matrices=False (bool): show matrix valued parameters below all circuit diagrams
        cache (dict): QuantumTape input only. Used to store information between recursive calls. Keys are ``'tape_offset'``
            and ``'matrices'``.
         expansion_strategy (str): QNode input only. The strategy to use when circuit expansions or decompositions
             are required.

             - ``gradient``: The QNode will attempt to decompose
               the internal circuit such that all circuit operations are supported by the gradient
               method.

             - ``device``: The QNode will attempt to decompose the internal circuit
               such that all circuit operations are natively supported by the device.


     Returns:
         A function that has the same argument signature as ``qnode``. When called,
         the function will draw the QNode.

     **Example**

     .. code-block:: python3

         @qml.qnode(qml.device('lightning.qubit', wires=2))
         def circuit(a, w):
             qml.Hadamard(0)
             qml.CRX(a, wires=[0, 1])
             qml.Rot(*w, wires=[1])
             qml.CRX(-a, wires=[0, 1])
             return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

     >>> print(qml.draw(circuit)(a=2.3, w=[1.2, 3.2, 0.7]))
     1: ────╭RX(2.30)──Rot(1.20,3.20,0.70)─╭RX(-2.30)─┤ ╭<Z@Z>
     0: ──H─╰C─────────────────────────────╰C─────────┤ ╰<Z@Z>

     .. UsageDetails::


     By specifying the ``decimals`` keyword, parameters are displayed to the specified precision.

     >>> print(qml.draw(circuit, decimals=4)(a=2.3, w=[1.2, 3.2, 0.7]))
     0: ──H─╭C─────────────────────────────────────╭C───────────┤ ╭<Z@Z>
     1: ────╰RX(2.3000)──Rot(1.2000,3.2000,0.7000)─╰RX(-2.3000)─┤ ╰<Z@Z>

     Parameters can be omitted by requesting ``decimals=None``:

     >>> print(qml.draw(circuit, decimals=None)(a=2.3, w=[1.2, 3.2, 0.7]))
     0: ──H─╭C───────╭C──┤ ╭<Z@Z>
     1: ────╰RX──Rot─╰RX─┤ ╰<Z@Z>

     If the parameters are not acted upon by classical processing like ``-a``, then
     ``qml.draw`` can handle string-valued parameters as well:

     >>> @qml.qnode(qml.device('lightning.qubit', wires=1))
     ... def circuit2(x):
     ...     qml.RX(x, wires=0)
     ...     return qml.expval(qml.PauliZ(0))
     >>> print(qml.draw(circuit2)("x"))
     0: ──RX(x)─┤  <Z>

     When requested with ``show_matrices=True``, matrix valued parameters are printed below the
     circuit:

     .. code-block:: python

     >>> @qml.qnode(qml.device('default.qubit', wires=2))
     ... def circuit():
     ...     qml.QubitUnitary(np.eye(2), wires=0)
     ...     qml.QubitUnitary(-np.eye(4), wires=(0,1))
     ...     return qml.expval(qml.Hermitian(np.eye(2), wires=1))
     >>> print(qml.draw(circuit, show_matrices=True)())
     0: ──U(M0)─╭U(M1)─┤
     1: ────────╰U(M1)─┤  <𝓗(M0)>
     M0 =
     [[1. 0.]
     [0. 1.]]
     M1 =
     [[-1. -0. -0. -0.]
     [-0. -1. -0. -0.]
     [-0. -0. -1. -0.]
     [-0. -0. -0. -1.]]

     The ``max_length`` keyword warps long circuits:

     .. code-block:: python

         rng = np.random.default_rng(seed=42)
         shape = qml.StronglyEntanglingLayers.shape(n_wires=3, n_layers=3)
         params = rng.random(shape)

         @qml.qnode(qml.device('lightning.qubit', wires=3))
         def longer_circuit(params):
             qml.StronglyEntanglingLayers(params, wires=range(3))
             return [qml.expval(qml.PauliZ(i)) for i in range(3)]

         print(qml.draw(longer_circuit, max_length=60)(params))

     .. code-block:: none

         0: ──Rot(0.77,0.44,0.86)─╭C────╭X──Rot(0.45,0.37,0.93)─╭C─╭X
         1: ──Rot(0.70,0.09,0.98)─╰X─╭C─│───Rot(0.64,0.82,0.44)─│──╰C
         2: ──Rot(0.76,0.79,0.13)────╰X─╰C──Rot(0.23,0.55,0.06)─╰X───

         ───Rot(0.83,0.63,0.76)──────────────────────╭C────╭X─┤  <Z>
         ──╭X────────────────────Rot(0.35,0.97,0.89)─╰X─╭C─│──┤  <Z>
         ──╰C────────────────────Rot(0.78,0.19,0.47)────╰X─╰C─┤  <Z>

     The ``wire_order`` keyword specifies the order of the wires from
     top to bottom:

     >>> print(qml.draw(circuit, wire_order=[1,0])(a=2.3, w=[1.2, 3.2, 0.7]))
     1: ────╭RX(2.30)──Rot(1.20,3.20,0.70)─╭RX(-2.30)─┤ ╭<Z@Z>
     0: ──H─╰C─────────────────────────────╰C─────────┤ ╰<Z@Z>

     If the device or ``wire_order`` has wires not used by operations, those wires are omitted
     unless requested with ``show_all_wires=True``

     >>> empty_qfunc = lambda : qml.expval(qml.PauliZ(0))
     >>> empty_circuit = qml.QNode(empty_qfunc, qml.device('lightning.qubit', wires=3))
     >>> print(qml.draw(empty_circuit, show_all_wires=True)())
     0: ───┤  <Z>
     1: ───┤
     2: ───┤

     Drawing also works on batch transformed circuits:

     .. code-block:: python

         @qml.gradients.param_shift(shift=0.1)
         @qml.qnode(qml.device('lightning.qubit', wires=1))
         def transformed_circuit(x):
             qml.RX(x, wires=0)
             return qml.expval(qml.PauliZ(0))

         print(qml.draw(transformed_circuit)(np.array(1.0)))

     .. code-block:: none

         0: ──RX(1.10)─┤  <Z>

         0: ──RX(0.90)─┤  <Z>

     """
    @wraps(circuit)
    def wrapper(*args, **kwargs):

        original_expansion_strategy = getattr(circuit, "expansion_strategy", None)

        try:
            circuit.expansion_strategy = expansion_strategy or original_expansion_strategy
            tapes = circuit.construct(args, kwargs)
        finally:
            circuit.expansion_strategy = original_expansion_strategy

        _wire_order = wire_order or circuit.device.wires

        if tapes is not None:
            cache = {"tape_offset": 0, "matrices": []}
            res = [
                draw(
                    t,
                    wire_order=_wire_order,
                    show_all_wires=show_all_wires,
                    decimals=decimals,
                    show_matrices=False,
                    max_length=max_length,
                    cache=cache,
                )
                for t in tapes[0]
            ]
            if show_matrices:
                mat_str = "\n"
                for i, mat in enumerate(cache["matrices"]):
                    mat_str += f"\nM{i} = \n{mat}"
                return "\n\n".join(res) + mat_str
            return "\n\n".join(res)

        return draw(
            circuit.tape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            show_matrices=show_matrices,
            max_length=max_length,
        )

    return wrapper