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
This module contains the @op_transform decorator.
"""
import functools
import os
import warnings

import pennylane as qml


def _make_tape(obj, wire_order, *args, **kwargs):
    if isinstance(obj, qml.QNode):
        # user passed a QNode, get the tape
        obj.construct(args, kwargs)
        tape = obj.qtape

        # if no wire ordering is specified, take wire list from the device
        wire_order = obj.device.wires if wire_order is None else qml.wires.Wires(wire_order)

    elif isinstance(obj, qml.tape.QuantumTape):
        # user passed a tape
        tape = obj
        # if no wire ordering is specified, take wire list from tape
        wire_order = tape.wires if wire_order is None else qml.wires.Wires(wire_order)

    elif issubclass(obj, qml.operation.Operator):
        tape = obj(*args, **kwargs)
        wire_order = tape.wires if wire_order is None else qml.wires.Wires(wire_order)

    elif callable(obj):
        # user passed something that is callable but not a tape or QNode.
        tape = qml.transforms.make_tape(obj)(*args, **kwargs)

        # raise exception if it is not a quantum function
        if len(tape.operations) == 0:
            raise ValueError("Function contains no quantum operation")

        # if no wire ordering is specified, take wire list from tape
        wire_order = tape.wires if wire_order is None else qml.wires.Wires(wire_order)

    else:
        raise ValueError("Input is not an Operator, tape, QNode, or quantum function")

    # check that all wire labels in the circuit are contained in wire_order
    if not set(tape.wires).issubset(wire_order):
        raise ValueError("Wires in circuit are inconsistent with those in wire_order")

    return tape, wire_order


class op_transform:

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if os.environ.get("SPHINX_BUILD") == "1":
            # If called during a Sphinx documentation build,
            # simply return the original function rather than
            # instantiating the object. This allows the signature to
            # be correctly displayed in the documentation.

            warnings.warn(
                "Operator transformations have been disabled, as a Sphinx "
                "build has been detected via SPHINX_BUILD='1'. If this is not the "
                "case, please set the environment variable SPHINX_BUILD='0'.",
                UserWarning,
            )

            args[0].custom_qnode_wrapper = lambda x: x
            return args[0]

        return super().__new__(cls)

    def __init__(self, fn):
        if not callable(fn):
            raise ValueError(
                f"The operator function to register, {fn}, "
                "does not appear to be a valid Python function or callable."
            )

        self.fn = fn
        self._tape_fn = None
        functools.update_wrapper(self, fn)

    def __call__(self, *targs, **tkwargs):
        obj = None

        if targs:
            obj, *targs = targs

        if isinstance(obj, (qml.operation.Operator, qml.tape.QuantumTape)) or callable(obj):
            return self._create_wrapper(obj, *targs, **tkwargs)

        # Input is not an operator nor a QNode nor a quantum tape nor a qfunc.
        # Assume Python decorator syntax:
        #
        # result = op_func(*transform_args)(obj)(*obj_args)
        #
        # or
        #
        # @op_func(*transform_args)
        # @qml.qnode(dev)
        # def circuit(...):
        #     ...
        # result = circuit(*qnode_args)

        # Prepend the input to the transform args,
        # and create a wrapper function.
        if obj is not None:
            targs = (obj,) + tuple(targs)

        def wrapper(obj):
            return self._create_wrapper(obj, *targs, **tkwargs)

        return wrapper

    @property
    def tape_fn(self):
        if self._tape_fn is None:
            raise ValueError("This transform does not support tapes or QNodes with multiple operations.")

        return self._tape_fn

    def tape_transform(self, fn):
        self._tape_fn = fn
        return self

    def _create_wrapper(self, obj, *targs, wire_order=None, **tkwargs):

        if isinstance(obj, qml.operation.Operator):
            # Input is a single operation.
            # op_transform(obj, *transform_args)
            if wire_order is not None:
                tkwargs["wire_order"] = wire_order

            wrapper = self.fn(obj, *targs, **tkwargs)

        elif isinstance(obj, qml.tape.QuantumTape):
            # Input is a quantum tape. Get the quantum tape.
            tape, new_wire_order = _make_tape(obj, wire_order)

            if wire_order is not None:
                tkwargs["wire_order"] = new_wire_order

            wrapper = self.tape_fn(tape, *targs, **tkwargs)

        elif callable(obj):
            # Input is a QNode, or qfunc. Get the quantum tape.
            def wrapper(*args, **kwargs):
                nonlocal wire_order
                tape, new_wire_order = _make_tape(obj, wire_order, *args, **kwargs)

                if wire_order is not None:
                    tkwargs["wire_order"] = new_wire_order

                if isinstance(tape, qml.operation.Operator):
                    return self.fn(tape, *targs, **tkwargs)

                return self.tape_fn(tape, *targs, **tkwargs)

        else:
            raise ValueError("Input is not an Operator, tape, QNode, or quantum function")

        return wrapper
