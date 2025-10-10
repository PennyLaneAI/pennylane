# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
This module contains the `_add_obj` function and its related utilities for adding objects to the text drawer.

The `_add_obj` function is a generic function that dispatches to specific implementations based on the type of the object being added. These implementations handle various types of quantum operations, measurements, and other constructs, ensuring they are properly represented in the text-based quantum circuit visualization.

Key Features:
- Handles conditional operators, controlled operations, and mid-measurement processes.
- Supports grouping symbols to visually indicate the extent of multi-wire operations.
- Provides specialized handling for mid-circuit measurement statistics.

Usage:
The `_add_obj` function is automatically invoked by the text drawer when rendering a quantum circuit. Users typically do not need to call it directly.
"""


from functools import singledispatch

from pennylane.measurements import (
    CountsMP,
    DensityMatrixMP,
    ExpectationMP,
    MeasurementProcess,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    StateMP,
    VarianceMP,
)
from pennylane.operation import Operator
from pennylane.ops import Adjoint, Conditional, Controlled, GlobalPhase, Identity
from pennylane.tape import QuantumScript
from pennylane.templates.subroutines import TemporaryAND


def _add_cond_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for conditional
    operators"""
    n_wires = len(config.wire_map)

    mapped_wires = [config.wire_map[w] for w in op.wires]
    mapped_bits = [config.bit_map[m] for m in op.meas_val.measurements]
    max_w = max(mapped_wires)
    max_b = max(mapped_bits)

    ctrl_symbol = "╩"
    if any(config.cur_layer == stretch[-1] for stretch in config.cwire_layers[max_b]):
        ctrl_symbol = "╝"
    layer_str[max_b + n_wires] = f"═{ctrl_symbol}"

    for w in range(max_w + 1, max(config.wire_map.values()) + 1):
        layer_str[w] = "─║"

    for b in range(max_b):
        if b in mapped_bits:
            intersection = "╬"
            if any(config.cur_layer == stretch[-1] for stretch in config.cwire_layers[b]):
                intersection = "╣"
            layer_str[b + n_wires] = f"═{intersection}"
        else:
            filler = " " if layer_str[b + n_wires][-1] == " " else "═"
            layer_str[b + n_wires] = f"{filler}║"

    return layer_str


def _add_grouping_symbols(op_wires, layer_str, config, closing=False):
    """Adds symbols indicating the extent of a given sequence of wires.
    Does nothing if the sequence has length 0 or 1."""

    if len(op_wires) <= 1:
        return layer_str

    mapped_wires = [config.wire_map[w] for w in op_wires]
    min_w, max_w = min(mapped_wires), max(mapped_wires)

    if closing:
        layer_str[min_w] += "╮"
        layer_str[max_w] += "╯"

        for w in range(min_w + 1, max_w):
            layer_str[w] += "┤" if w in mapped_wires else "│"
    else:
        layer_str[min_w] = "╭"
        layer_str[max_w] = "╰"

        for w in range(min_w + 1, max_w):
            layer_str[w] = "├" if w in mapped_wires else "│"

    return layer_str


def _add_mid_measure_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-measure
    operators"""
    if op not in config.bit_map:
        return layer_str

    n_wires = len(config.wire_map)
    mapped_wire = config.wire_map[op.wires[0]]
    bit = config.bit_map[op] + n_wires
    layer_str[bit] += " ╚"

    for w in range(mapped_wire + 1, n_wires):
        layer_str[w] += "─║"

    for b in range(n_wires, bit):
        filler = " " if layer_str[b][-1] == " " else "═"
        layer_str[b] += f"{filler}║"

    return layer_str


@singledispatch
def _add_obj(
    obj, layer_str: list[str], config, tape_cache=None, skip_grouping_symbols=False
) -> list[str]:
    raise NotImplementedError(f"unable to draw object {obj}")


@_add_obj.register
def _add_cond(obj: Conditional, layer_str, config, tape_cache=None, skip_grouping_symbols=False):
    layer_str = _add_cond_grouping_symbols(obj, layer_str, config)
    return _add_obj(obj.base, layer_str, config)


@_add_obj.register
def _add_controlled(
    obj: Controlled, layer_str, config, tape_cache=None, skip_grouping_symbols=False
):
    if isinstance(obj.base, (GlobalPhase, Identity)):
        return _add_controlled_global_op(obj, layer_str, config)

    layer_str = _add_grouping_symbols(obj.wires, layer_str, config)
    for w, val in zip(obj.control_wires, obj.control_values):
        layer_str[config.wire_map[w]] += "●" if val else "○"
    return _add_obj(obj.base, layer_str, config, skip_grouping_symbols=True)


def _add_controlled_global_op(obj, layer_str, config):
    """This is not another dispatch managed by @_add_obj.register,
    but a manually managed dispatch."""
    layer_str = _add_grouping_symbols(list(config.wire_map.keys()), layer_str, config)

    for w, val in zip(obj.control_wires, obj.control_values):
        layer_str[config.wire_map[w]] += "●" if val else "○"

    label = obj.base.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    for w, val in config.wire_map.items():
        if w not in obj.control_wires:
            layer_str[val] += label

    return layer_str


def _add_elbow_core(obj, layer_str, config):
    cvals = obj.hyperparameters["control_values"]
    mapped_wires = [config.wire_map[w] for w in obj.wires]
    layer_str[mapped_wires[0]] += "●" if cvals[0] else "○"
    layer_str[mapped_wires[1]] += "●" if cvals[1] else "○"
    layer_str[mapped_wires[2]] += "⊕"
    return layer_str, mapped_wires


@_add_obj.register
def _add_left_elbow(
    obj: TemporaryAND, layer_str, config, tape_cache=None, skip_grouping_symbols=False
):
    """Updates ``layer_str`` with ``op`` operation of type ``TemporaryAND``,
    also known as left elbow."""
    if not skip_grouping_symbols:
        layer_str = _add_grouping_symbols(obj.wires, layer_str, config)
    layer_str, _ = _add_elbow_core(obj, layer_str, config)
    return layer_str


def _add_right_elbow(obj: TemporaryAND, layer_str, config):
    """Updates ``layer_str`` with ``op`` operation of type ``Adjoint(TemporaryAND)``,
    also known as right elbow."""
    layer_str, mapped_wires = _add_elbow_core(obj, layer_str, config)
    # Fill with "─" on intermediate wires the elbow does not act on, to shift "|" correctly
    for w in range(min(mapped_wires) + 1, max(mapped_wires)):
        if w not in mapped_wires:
            layer_str[w] += "─"
    return _add_grouping_symbols(obj.wires, layer_str, config, closing=True)


@_add_obj.register
def _add_adjoint(obj: Adjoint, layer_str, config, tape_cache=None, skip_grouping_symbols=False):
    """Updates ``layer_str`` with ``op`` operation of type Adjoint. Currently
    only differs from ``_add_op`` if the base of the adjoint op is a ``TemporaryAND``,
    making the overall object a right elbow."""
    if isinstance(obj.base, TemporaryAND):
        return _add_right_elbow(obj.base, layer_str, config)
    return _add_op(obj, layer_str, config, tape_cache, skip_grouping_symbols)


@_add_obj.register
def _add_op(obj: Operator, layer_str, config, tape_cache=None, skip_grouping_symbols=False):
    """Updates ``layer_str`` with ``op`` operation."""
    if not skip_grouping_symbols:
        layer_str = _add_grouping_symbols(obj.wires, layer_str, config)

    label = obj.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    if len(obj.wires) == 0:  # operation (e.g. barrier, snapshot) across all wires
        n_wires = len(config.wire_map)
        for i, s in enumerate(layer_str[:n_wires]):
            layer_str[i] = s + label
    else:
        for w in obj.wires:
            layer_str[config.wire_map[w]] += label

    return layer_str


@_add_obj.register(Identity)
@_add_obj.register(GlobalPhase)
def _add_global_op(
    obj: GlobalPhase | Identity,
    layer_str,
    config,
    tape_cache=None,
    skip_grouping_symbols=False,
):
    n_wires = len(config.wire_map)
    if not skip_grouping_symbols:
        layer_str = _add_grouping_symbols(list(config.wire_map.keys()), layer_str, config)

    label = obj.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    for i, s in enumerate(layer_str[:n_wires]):
        layer_str[i] = s + label

    return layer_str


@_add_obj.register
def _add_mid_measure_op(
    op: MidMeasureMP, layer_str, config, tape_cache=None, skip_grouping_symbols=False
):
    """Updates ``layer_str`` with ``op`` operation when ``op`` is a
    ``qml.measurements.MidMeasureMP``."""
    layer_str = _add_mid_measure_grouping_symbols(op, layer_str, config)
    label = op.label(decimals=config.decimals, cache=config.cache).replace("\n", "")

    for w in op.wires:
        layer_str[config.wire_map[w]] += label

    return layer_str


@_add_obj.register
def _add_tape(obj: QuantumScript, layer_str, config, tape_cache, skip_grouping_symbols=False):
    layer_str = _add_grouping_symbols(obj.wires, layer_str, config)
    label = f"Tape:{config.cache['tape_offset']+len(tape_cache)}"
    for w in obj.wires:
        layer_str[config.wire_map[w]] += label
    tape_cache.append(obj)
    return layer_str


measurement_label_map = {
    ExpectationMP: lambda label: f"<{label}>",
    ProbabilityMP: lambda label: f"Probs[{label}]" if label else "Probs",
    SampleMP: lambda label: f"Sample[{label}]" if label else "Sample",
    CountsMP: lambda label: f"Counts[{label}]" if label else "Counts",
    VarianceMP: lambda label: f"Var[{label}]",
    StateMP: lambda label: "State",
    DensityMatrixMP: lambda label: "DensityMatrix",
}


def _add_cwire_measurement_grouping_symbols(mcms, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-circuit measurement
    statistics."""
    if len(mcms) > 1:
        n_wires = len(config.wire_map)
        mapped_bits = [config.bit_map[m] for m in mcms]
        min_b, max_b = min(mapped_bits) + n_wires, max(mapped_bits) + n_wires

        layer_str[min_b] = "╭"
        layer_str[max_b] = "╰"

        for b in range(min_b + 1, max_b):
            layer_str[b] = "├" if b - n_wires in mapped_bits else "│"

    return layer_str


def _add_cwire_measurement(m, layer_str, config):
    """Updates ``layer_str`` with the ``m`` measurement when it is used
    for collecting mid-circuit measurement statistics."""
    mcms = [v.measurements[0] for v in m.mv] if isinstance(m.mv, list) else m.mv.measurements
    layer_str = _add_cwire_measurement_grouping_symbols(mcms, layer_str, config)

    mv_label = "MCM"
    meas_label = measurement_label_map[type(m)](mv_label)

    n_wires = len(config.wire_map)
    for mcm in mcms:
        ind = config.bit_map[mcm] + n_wires
        layer_str[ind] += meas_label

    return layer_str


@_add_obj.register
def _add_measurement(
    m: MeasurementProcess,
    layer_str,
    config,
    tape_cache=None,
    skip_grouping_symbols=False,
):
    """Updates ``layer_str`` with the ``m`` measurement."""
    if m.mv is not None:
        return _add_cwire_measurement(m, layer_str, config)

    layer_str = _add_grouping_symbols(m.wires, layer_str, config)

    if m.obs is None:
        obs_label = None
    else:
        obs_label = m.obs.label(decimals=config.decimals, cache=config.cache).replace("\n", "")
    if type(m) in measurement_label_map:
        meas_label = measurement_label_map[type(m)](obs_label)
    else:
        meas_label = str(m)

    if len(m.wires) == 0:  # state or probability across all wires
        n_wires = len(config.wire_map)
        for i, s in enumerate(layer_str[:n_wires]):
            layer_str[i] = s + meas_label

    for w in m.wires:
        layer_str[config.wire_map[w]] += meas_label
    return layer_str
