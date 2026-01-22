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
Contains the abstractions for subroutines.
"""
import copy
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache, update_wrapper
from importlib.util import find_spec
from inspect import BoundArguments, Signature, signature
from typing import Any, ParamSpec

import numpy as np

from pennylane import capture, queuing
from pennylane.capture import subroutine as capture_subroutine
from pennylane.operation import Operation
from pennylane.pytrees import flatten
from pennylane.wires import Wires

has_jax = find_spec("jax") is not None


def _default_setup_inputs(*args, **kwargs):
    return args, kwargs


@lru_cache
def _get_array_types():
    if has_jax:
        import jax  # pylint: disable=import-outside-toplevel

        return (jax.numpy.ndarray, np.ndarray)
    return (np.ndarray,)


@lru_cache
def _get_iterable_wires_types():
    return (
        list,
        tuple,
        Wires,
        range,
        capture.autograph.ag_primitives.PRange,
        set,
        *_get_array_types(),
    )


def _setup_wires(wires):
    if isinstance(wires, _get_array_types()) and wires.shape == ():
        return (wires,)
    if isinstance(wires, _get_iterable_wires_types()):
        return tuple(wires)
    return (wires,)


class SubroutineOp(Operation):
    """An operator constructed from a :class:`~.Subroutine` together with its bound arguments.
    This class should not be created directly, but is the byproduct of calling a ``Subroutine``.

    Args:
        subroutine (Subroutine): the definition of a subroutine from a quantum function
        bound_args (inspect.BoundArguments): the inputs to the subroutine bound to the subroutine's signature
        decomposition (list[Operator]): the decomposition of the subroutine with the given ``bound_args``.
        output (Any): Any output from the subroutine.

    """

    _primitive = None

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        raise ValueError(
            "SubroutineOp's should never be directly captured. That should occur in Subroutine instead."
        )

    def __deepcopy__(self, memo) -> "SubroutineOp":
        bound_args = copy.deepcopy(self._bound_args, memo)
        # create new decomp and output to keep inputs, decomp, and outputs consistent with each other
        with queuing.AnnotatedQueue() as decomposition:
            output = self.subroutine.definition(**bound_args.arguments)
        return SubroutineOp(self.subroutine, bound_args, decomposition.queue, output)

    grad_method = None

    def _flatten(self):
        dynamic_args = {
            arg: self._bound_args.arguments[arg] for arg in self.subroutine.dynamic_argnames
        }
        static_args = {
            arg: self._bound_args.arguments[arg] for arg in self.subroutine.static_argnames
        }
        for wire_argname in self.subroutine.wire_argnames:
            static_args[wire_argname] = self._bound_args.arguments[wire_argname]
        static_args = tuple((key, value) for key, value in static_args.items())
        return (dynamic_args,), (self.subroutine, static_args)

    @classmethod
    def _unflatten(cls, data, metadata):
        subroutine = metadata[0]
        with queuing.AnnotatedQueue() as decomposition:
            output = subroutine.definition(**data[0], **dict(metadata[1]))
        bound_args = subroutine.signature.bind(**data[0], **dict(metadata[1]))
        return SubroutineOp(subroutine, bound_args, decomposition.queue, output)

    def __repr__(self):
        inputs = ", ".join(f"{key}={value}" for key, value in self._bound_args.arguments.items())
        return f"<{self.name}({inputs})>"

    def __init__(
        self,
        subroutine: "Subroutine",
        bound_args: BoundArguments,
        decomposition: list[Operation],
        output: Any = None,
    ):
        self._subroutine = subroutine
        self._bound_args = bound_args
        self._decomp = decomposition
        self._output = output

        wires = []
        for wire_argname in self._subroutine.wire_argnames:
            reg_wires = self._bound_args.arguments[wire_argname]
            # allow same wires to exist in multiple registers
            reg_wires = [w for w in reg_wires if w not in wires]
            wires.extend(reg_wires)
        super().__init__(wires=wires)
        self.name = subroutine.name

        dynamic_args = [self._bound_args.arguments[arg] for arg in self.subroutine.dynamic_argnames]
        self.data = tuple(flatten(dynamic_args)[0])

    @property
    def bound_args(self) -> BoundArguments:
        """The inputs to the Subroutine."""
        return self._bound_args

    @property
    def output(self):
        """Test output of the subroutine."""
        return self._output

    @property
    def subroutine(self) -> "Subroutine":
        """The subroutine definition used with this operator."""
        return self._subroutine

    def map_wires(self, wire_map):
        new_args = deepcopy(self._bound_args)
        for wire_argname in self._subroutine.wire_argnames:
            new_wires = tuple(wire_map.get(w, w) for w in self._bound_args.arguments[wire_argname])
            new_args.arguments[wire_argname] = new_wires

        with queuing.AnnotatedQueue() as decomposition:
            output = self.subroutine.definition(*new_args.args, **new_args.kwargs)
        return SubroutineOp(self.subroutine, new_args, decomposition.queue, output)

    def decomposition(self):
        if queuing.QueuingManager.recording():
            _ = [queuing.apply(op) for op in self._decomp]
        return self._decomp

    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache: dict | None = None
    ) -> str:
        return super().label(decimals, base_label=self.name, cache=cache)


P = ParamSpec("P")


# pylint: disable=too-many-arguments
class Subroutine:
    """The definition of a Subroutine, compatible both with program capture and backwards
    compatible with operators.

    Args:
        definition (Callable): a quantum function that can contain both quantum and classical processing.
            The definition can return purely classical values or the outputs from mid circuit measurements, but
            it cannot return terminal statistics.
        setup_inputs (Callable): An function that can run preprocessing on the inputs before hitting
            definition.  This can be used to make static arguments hashable for compatibility with program capture.
        static_argnames (str | tuple[str]): The name of arguments that are treated as static (trace- and compile-time constant).
        wire_argnames (str | tuple[str]): The name of arguments that represent wire registers.  While the users can
            be more permissive in what they provide to wire arguments, the definition should treat all wire
            arguments as 1D arrays.
        compute_resources (None | Callable): A function for computing resources used by the function.
            It should only calculate the resources from the static arguments, the length of the wire registers,
            and the shape and dtype of the dynamic arguments. In the case of the specific resources
            depending on the specifics of a dynamic argument, a worse case scenario can be used.

    For simple cases, a ``Subroutine`` can simply be created from a single quantum function, like:

    .. code-block:: python

        from pennylane.templates import Subroutine

        @Subroutine
        def MyTemplate(x, y, wires):
            qml.RX(x, wires[0])
            qml.RY(y, wires[0])

        @qml.qnode(qml.device('default.qubit'))
        def c():
            MyTemplate(0.1, 0.2, 0)
            return qml.state()

        c()

    >>> print(qml.draw(c)())
    0: ──MyTemplate─┤  State
    >>> print(qml.draw(c, level="device")())
    0: ──RX(0.10)──RY(0.20)─┤  State
    >>> print(qml.specs(c)().resources)
    Total wire allocations: 1
    Total gates: 1
    Circuit depth: 1
    <BLANKLINE>
    Gate types:
      MyTemplate: 1
    <BLANKLINE>
    Measurements:
      state(all wires): 1

    For multiple wire register inputs or use of a different name than ``"wires"``, the
    ``wire_argnames`` can be provided:

    .. code-block:: python

        from functools import partial

        @partial(Subroutine, wire_argnames=("register1", "register2"))
        def MultiRegisterTemplate(register1, register2):
            for wire in register1:
                qml.X(wire)
            for wire in register2:
                qml.Z(wire)

    >>> print(qml.draw(MultiRegisterTemplate)(0, [1,2]))
    0: ─╭MultiRegisterTemplate─┤
    1: ─├MultiRegisterTemplate─┤
    2: ─╰MultiRegisterTemplate─┤

    Static arguments are treated as compile-time constant with ``qml.qjit``, and must
    be hashable. These are any inputs that are not numerical data or Operators. In the below
    example, the ``pauli_word`` argument is a string that is a static argument.

    .. code-block:: python

        @partial(Subroutine, static_argnames="pauli_word")
        def WithStaticArg(x, wires, pauli_word: str):
            qml.PauliRot(x, pauli_word, wires)

    Sometimes we want to allow the user to be able to provide a static input in a
    non-hashable format. For example, the user might provide an input as a ``list``
    instead of a ``tuple``.  This can be done by providing the ``setup_inputs`` function.
    This function should have the same call signature as the template and return
    a tuple of position arguments and a dictionary of keyword arguments.

    .. code-block:: python

        def setup_inputs(x, wires, pauli_words):
            return (x, wires, tuple(pauli_words)), {}

        @partial(Subroutine, static_argnames="pauli_words", setup_inputs=setup_inputs)
        def WithSetup(x, wires, pauli_words: Sequence[str]):
            for word in pauli_words:
                qml.PauliRot(x, word, wires)


    >>> print(qml.draw(WithSetup)(0.5, [0, 1], ["XX", "XY", "XZ"]))
    0: ─╭WithSetup─┤
    1: ─╰WithSetup─┤


    """

    def __repr__(self):
        return f"<Subroutine: {self.name}>"

    def __instancecheck__(self, instance) -> bool:
        return isinstance(instance, SubroutineOp) and instance.subroutine is self

    def __init__(
        self,
        definition: Callable[P, Any],
        *,
        setup_inputs: Callable[P, tuple[tuple, dict]] = _default_setup_inputs,
        static_argnames: str | tuple[str, ...] = tuple(),
        wire_argnames: str | tuple[str, ...] = ("wires",),
        compute_resources: None | Callable[P, dict] = None,
    ):
        self._definition = definition
        self._setup_inputs = setup_inputs
        self._compute_resources = compute_resources
        self._signature = signature(definition)
        update_wrapper(self, definition)
        if isinstance(static_argnames, str):
            static_argnames = (static_argnames,)
        if isinstance(wire_argnames, str):
            wire_argnames = (wire_argnames,)
        # need to use tuple for static argnames and wire argnames to preserve ordering
        # otherwise things can get shuffled in SubroutineOp
        self._static_argnames = tuple(static_argnames)
        self._wire_argnames = tuple(wire_argnames)

        self._capture_subroutine = capture_subroutine(definition, static_argnames=static_argnames)

    @property
    def name(self) -> str:
        """A string representation to label the Subroutine."""
        return getattr(self._definition, "__name__", str(self._definition))

    @property
    def signature(self) -> Signature:
        """ "The signature for the definition. Used to preprocess the user inputs."""
        return self._signature

    def compute_resources(self, *args, **kwargs) -> dict:
        """Calculate a condensed representation for the resources required for the Subroutine."""
        if self._compute_resources is None:
            raise NotImplementedError(f"{self} does not have a defined compute_resources function.")
        return self._compute_resources(*args, **kwargs)

    def definition(self, *args, **kwargs):
        """The quantum function definition of the subroutine."""
        return self._definition(*args, **kwargs)

    def setup_inputs(self, *args, **kwargs) -> tuple[tuple, dict]:
        """Perform and initial setup of the arguments."""
        return self._setup_inputs(*args, **kwargs)

    @property
    def static_argnames(self) -> tuple[str, ...]:
        """The names of arguments that are compile time constant."""
        return self._static_argnames

    @property
    def wire_argnames(self) -> tuple[str, ...]:
        """The names for the arguments that represent a register of wires."""
        return self._wire_argnames

    @property
    def dynamic_argnames(self) -> tuple[str, ...]:
        """The names of the function arguments that are pytrees of numerical data. These are the arguments
        that are not static or wires."""

        def is_static(name):
            return name in self.static_argnames or name in self.wire_argnames

        return tuple(name for name in self._signature.parameters if not is_static(name))

    def __call__(self, *args, **kwargs):
        args, kwargs = self.setup_inputs(*args, **kwargs)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for wire_argname in self.wire_argnames:
            register = _setup_wires(bound_args.arguments[wire_argname])
            if capture.enabled():
                import jax  # pylint: disable=import-outside-toplevel

                if len(register) > 0:
                    bound_args.arguments[wire_argname] = jax.numpy.stack(register)
            else:
                bound_args.arguments[wire_argname] = Wires(register)

        if capture.enabled():
            return self._capture_subroutine(*bound_args.args, **bound_args.kwargs)

        with queuing.AnnotatedQueue() as decomposition:
            output = self.definition(*bound_args.args, **bound_args.kwargs)
        op = SubroutineOp(self, bound_args, decomposition.queue, output)
        return op if op.output is None else op.output
