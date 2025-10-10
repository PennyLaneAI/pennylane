# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This subpackage contains PennyLane transforms and their building blocks.

.. warning::

    The transforms ``add_noise``, ``insert``, ``mitigate_with_zne``, ``fold_global``, ``poly_extrapolate``, ``richardson_extrapolate``,
    ``exponential_extrapolate`` have been moved to the :mod:`pennylane.noise` module.
    Accessing these transforms from the :mod:`pennylane.transforms` module is deprecated
    and will be removed in v0.44.

.. currentmodule:: pennylane

.. _transforms:

Custom transforms
-----------------

:func:`qml.transform <pennylane.transform>` can be used to define custom transformations
that work with PennyLane QNodes; such transformations can map a circuit
to one or many new circuits alongside associated classical post-processing.

.. autosummary::
    :toctree: api

    ~transforms.core.transform

Transforms library
------------------
A range of ready-to-use transforms are available in PennyLane.

Transforms for circuit compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A set of transforms to perform basic circuit compilation tasks.

.. autosummary::
    :toctree: api

    ~compile
    ~transforms.cancel_inverses
    ~transforms.combine_global_phases
    ~transforms.commute_controlled
    ~transforms.decompose
    ~transforms.merge_amplitude_embedding
    ~transforms.merge_rotations
    ~transforms.pattern_matching_optimization
    ~transforms.remove_barrier
    ~transforms.match_relative_phase_toffoli
    ~transforms.match_controlled_iX_gate
    ~transforms.single_qubit_fusion
    ~transforms.transpile
    ~transforms.undo_swaps
    ~transforms.unitary_to_rot
    ~transforms.rz_phase_gradient
    ~transforms.rowcol

Compilation transforms using ZX calculus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a set of transforms that use ZX calculus to optimize circuits.

.. currentmodule:: pennylane.transforms
.. autosummary::
    :toctree: api

    zx.optimize_t_count
    zx.push_hadamards
    zx.reduce_non_clifford
    zx.todd

The following utility functions assist when working explicitly with ZX diagrams,
for example when writing custom ZX compilation passes. Also see the section
on intermediate representations below.

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    ~transforms.to_zx
    ~transforms.from_zx

Other compilation utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are additional utility functions and decompositions available that assist with
both transforms and decompositions within the larger PennyLane codebase.

.. autosummary::
    :toctree: api

    ~transforms.set_decomposition
    ~transforms.pattern_matching

There are also utility functions that take a circuit and return a DAG.

.. autosummary::
    :toctree: api

    ~transforms.commutation_dag
    ~transforms.CommutationDAG
    ~transforms.CommutationDAGNode

Transform for Clifford+T decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This transform accepts quantum circuits and decomposes them to the Clifford+T basis.

.. autosummary::
    :toctree: api

    ~clifford_t_decomposition

Other transforms
~~~~~~~~~~~~~~~~

These transforms use the :func:`pennylane.transform` function/decorator and can be used on
:class:`pennylane.tape.QuantumTape`, :class:`pennylane.QNode`. They fulfill multiple purposes like circuit
preprocessing, getting information from a circuit, and more.

.. autosummary::
    :toctree: api

    ~batch_params
    ~batch_input
    ~defer_measurements
    ~transforms.diagonalize_measurements
    ~transforms.split_non_commuting
    ~transforms.split_to_single_terms
    ~transforms.broadcast_expand
    ~transforms.sign_expand
    ~transforms.convert_to_numpy_parameters
    ~apply_controlled_Q
    ~quantum_monte_carlo
    ~transforms.resolve_dynamic_wires

Transforms for intermediate representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intermediate representations (IRs) are alternative representations of quantum circuits, typically
offering a more efficient classical description for special classes of circuits.
The following functions produce intermediate representations of quantum circuits:

.. autosummary::
    :toctree: api

    ~transforms.parity_matrix
    ~transforms.phase_polynomial
    ~transforms.rowcol

In addition, there are the following utility functions to traverse a graph:

.. currentmodule:: pennylane.transforms
.. autosummary::
    :toctree: api

    intermediate_reps.postorder_traverse
    intermediate_reps.preorder_traverse

Transforms that act only on QNodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These transforms only accept QNodes, and return new transformed functions
that compute the desired quantity.

.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    ~batch_partial
    ~draw
    ~draw_mpl

Transforms developer functions
------------------------------

:class:`~.TransformContainer`, :class:`~.TransformDispatcher`, and  :class:`~.TransformProgram` are
developer-facing objects that allow the
creation, dispatching, and composability of transforms. If you would like to make a custom transform, refer
instead to the documentation of :func:`qml.transform <pennylane.transform>`.

.. autosummary::
    :toctree: api

    ~transforms.core.transform_dispatcher
    ~transforms.core.transform_program

Transforming circuits
---------------------

A quantum transform is a crucial concept in PennyLane, and refers to mapping a quantum
circuit to one or more circuits, alongside a classical post-processing function.
Once a transform is registered with PennyLane, the transformed circuits will be executed,
and the classical post-processing function automatically applied to the outputs.
This becomes particularly valuable when a transform generates multiple circuits,
requiring a method to aggregate or reduce the results (e.g.,
applying the parameter-shift rule or computing the expectation value of a Hamiltonian
term-by-term).

.. note::

    For examples of built-in transforms that come with PennyLane, see the
    :doc:`/introduction/compiling_circuits` documentation.

Creating your own transform
---------------------------

To streamline the creation of transforms and ensure their versatility across
various circuit abstractions in PennyLane, the
:func:`pennylane.transform` is available.

This decorator registers transforms that accept a :class:`~.QuantumTape`
as its primary input and returns a sequence of :class:`~.QuantumTape`
and an associated processing function.

To illustrate the process of creating a quantum transform, let's consider an example. Suppose we want
a transform that removes all :class:`~.RX` operations from a given circuit. In this case, we merely need to filter the
original :class:`~.QuantumTape` and return a new one without the filtered operations. As we don't require a specific processing
function in this scenario, we include a function that simply returns the first and only result.

.. code-block:: python

    from pennylane.tape import QuantumScript, QuantumScriptBatch
    from pennylane.typing import PostprocessingFn

    def remove_rx(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:

        operations = filter(lambda op: op.name != "RX", tape.operations)
        new_tape = tape.copy(operations=operations)

        def null_postprocessing(results):
            return results[0]

        return [new_tape], null_postprocessing

To make your transform applicable to both :class:`~.QNode` and quantum functions, you can use the :func:`pennylane.transform` decorator.

.. code-block:: python

    dispatched_transform = qml.transform(remove_rx)

For a more advanced example, let's consider a transform that sums a circuit with its adjoint. We define the adjoint
of the tape operations, create a new tape with these new operations, and return both tapes.
The processing function then sums the results of the original and the adjoint tape.
In this example, we use ``qml.transform`` in the form of a decorator in order to turn the custom
function into a quantum transform.

.. code-block:: python

    from pennylane.tape import QuantumScript, QuantumScriptBatch
    from pennylane.typing import PostprocessingFn

    @qml.transform
    def sum_circuit_and_adjoint(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:

        operations = [qml.adjoint(op) for op in tape.operation]
        new_tape = tape.copy(operations=operations)

        def sum_postprocessing(results):
            return qml.sum(results)

        return [tape, new_tape], sum_postprocessing

Composability of transforms
---------------------------

Transforms are inherently composable on a :class:`~.QNode`, meaning that transforms with compatible post-processing
functions can be successively applied to QNodes. For example, this allows for the application of multiple compilation
passes on a QNode to maximize gate reduction before execution.

.. code-block:: python

        dev = qml.device("default.qubit", wires=1)

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(device=dev)
        def circuit(x, y):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(y, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.Z(0))

In this example, inverses are canceled, leading to the removal of two Hadamard gates. Subsequently, rotations are
merged into a single :class:`qml.Rot` gate. Consequently, two transforms are successfully applied to the circuit.


Passing arguments to transforms
-------------------------------

We can decorate a QNode with ``@partial(transform_fn, **transform_kwargs)`` to provide additional keyword arguments to a transform function.
In the following example, we pass the keyword argument ``grouping_strategy="wires"`` to the :func:`~.split_non_commuting` quantum transform,
which splits a circuit into tapes measuring groups of commuting observables.

.. code-block:: python

        from functools import partial

        dev = qml.device("default.qubit", wires=2)

        @partial(qml.transforms.split_non_commuting, grouping_strategy="wires")
        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RZ(params[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

Additional information
----------------------

Explore practical examples of transforms focused on compiling circuits in the :doc:`compiling circuits documentation </introduction/compiling_circuits>`.
For gradient transforms, refer to the examples in the :doc:`gradients documentation <../code/qml_gradients>`. Finally,
for a comprehensive overview of transforms and core functionalities, consult the :doc:`summary above <../code/qml_transforms>`.
"""

# Leave as alias for backwards-compatibility
from pennylane.tape import make_qscript as make_tape
from pennylane.exceptions import TransformError

# Import the decorators first to prevent circular imports when used in other transforms
from .core import transform
from .batch_params import batch_params
from .batch_input import batch_input
from .batch_partial import batch_partial
from .convert_to_numpy_parameters import convert_to_numpy_parameters
from .compile import compile

from .decompositions import clifford_t_decomposition
from .defer_measurements import defer_measurements
from .diagonalize_measurements import diagonalize_measurements
from .dynamic_one_shot import dynamic_one_shot, is_mcm
from .sign_expand import sign_expand
from .split_non_commuting import split_non_commuting
from .split_to_single_terms import split_to_single_terms
from .combine_global_phases import combine_global_phases
from .resolve_dynamic_wires import resolve_dynamic_wires

from .optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    single_qubit_fusion,
    merge_amplitude_embedding,
    remove_barrier,
    undo_swaps,
    pattern_matching,
    pattern_matching_optimization,
    match_controlled_iX_gate,
    match_relative_phase_toffoli,
)
from .qmc import apply_controlled_Q, quantum_monte_carlo
from .unitary_to_rot import unitary_to_rot
from .commutation_dag import (
    commutation_dag,
    CommutationDAG,
    CommutationDAGNode,
)
from .tape_expand import (
    expand_invalid_trainable,
    expand_multipar,
    expand_nonunitary_gen,
    expand_trainable_multipar,
    create_expand_fn,
    create_decomp_expand_fn,
    create_expand_trainable_multipar,
    set_decomposition,
)
from .transpile import transpile
from .zx import (
    to_zx,
    from_zx,
)
from .broadcast_expand import broadcast_expand
from .decompose import decompose
from .intermediate_reps import (
    parity_matrix,
    phase_polynomial,
    rowcol,
)
from .rz_phase_gradient import rz_phase_gradient


def __getattr__(name):
    if name in {
        "add_noise",
        "insert",
        "mitigate_with_zne",
        "fold_global",
        "poly_extrapolate",
        "richardson_extrapolate",
        "exponential_extrapolate",
    }:

        # pylint: disable=import-outside-toplevel
        import warnings
        from pennylane import exceptions
        from pennylane import noise

        warnings.warn(
            f"pennylane.{name} is no longer accessible from the transforms module and must "
            "be imported as pennylane.noise.{name}. Support for access through this module "
            "will be removed in v0.44.",
            exceptions.PennyLaneDeprecationWarning,
        )

        return getattr(noise, name)

    raise AttributeError(f"module 'pennylane' has no attribute '{name}'")  # pragma: no cover
