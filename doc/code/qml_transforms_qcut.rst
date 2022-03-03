:orphan:

qml.transforms.qcut
===================

This module provides support for quantum circuit cutting, allowing larger circuits to be split into
smaller circuits that are compatible with devices that have a restricted number of qubits. The main
entrypoint to circuit cutting is through the :func:`qml.cut_circuit() <pennylane.cut_circuit>`
batch transform.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    cut_circuit

Advanced users also have the option to work directly with a :class:`~.QuantumTape` and manipulate
the tape to perform circuit cutting using the following functionality:

.. currentmodule:: pennylane.transforms.qcut

.. autosummary::
    :toctree: api

    tape_to_graph
    replace_wire_cut_nodes
    fragment_graph
    graph_to_tape
    remap_tape_wires
    expand_fragment_tapes
    qcut_processing_fn
    CutStrategy
