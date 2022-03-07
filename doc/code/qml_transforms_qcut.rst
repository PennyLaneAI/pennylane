:orphan:

qml.transforms.qcut
===================

This module provides support for quantum circuit cutting, allowing larger circuits to be split into
smaller circuits that are compatible with devices that have a restricted number of qubits.

The suggested entrypoint into circuit cutting is through the
:func:`qml.cut_circuit() <pennylane.cut_circuit>`
batch transform, which performs cutting according to the manual placement of :class:`~.WireCut`
operations in the circuit.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    cut_circuit
    WireCut

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
    expand_fragment_tape
    qcut_processing_fn
    CutStrategy

Building blocks for circuit cutting
-----------------------------------

The following discusses the elementary steps taken by the
:func:`qml.cut_circuit() <pennylane.cut_circuit>` transform. Consider the circuit below:

.. code-block:: python

    with qml.tape.QuantumTape() as tape:
        qml.RX(0.531, wires=0)
        qml.RY(0.9, wires=1)
        qml.RX(0.3, wires=2)

        qml.CZ(wires=[0, 1])
        qml.RY(-0.4, wires=0)

        qml.WireCut(wires=1)

        qml.CZ(wires=[1, 2])

        qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

>>> print(tape.draw())
 0: ──RX(0.531)──╭C──RY(-0.4)──────╭┤ ⟨Z ⊗ Z ⊗ Z⟩
 1: ──RY(0.9)────╰Z──//────────╭C──├┤ ⟨Z ⊗ Z ⊗ Z⟩
 2: ──RX(0.3)──────────────────╰Z──╰┤ ⟨Z ⊗ Z ⊗ Z⟩

To cut the circuit, we first convert convert it to its graph representation:

>>> graph = qml.transforms.qcut.tape_to_graph(tape)

.. figure:: ../../_static/qcut_graph.svg
    :align: center
    :width: 60%
    :target: javascript:void(0);

Our next step is to remove the :class:`~.WireCut` nodes in the graph and replace with
:class:`~.MeasureNode` and :class:`~.PrepareNode` pairs.

>>> qml.transforms.qcut.replace_wire_cut_nodes(graph)

The :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs are placeholder operations that
allow us to cut the circuit graph and then iterate over measurement and preparation
configurations at cut locations. First, the :func:`~.fragment_graph` function pulls apart
the graph into disconnected components as well as returning the ``communication_graph``
detailing the connectivity between the components.

>>> fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)

We now convert the ``fragments`` back to :class:`~.QuantumTape` objects

>>> fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

The circuit fragments can now be visualized:

>>> print(fragment_tapes[0].draw())
 0: ──RX(0.531)──╭C──RY(-0.4)─────┤ ⟨Z⟩
 1: ──RY(0.9)────╰Z──MeasureNode──┤

>>> print(fragment_tapes[1].draw())
 2: ──RX(0.3)──────╭Z──╭┤ ⟨Z ⊗ Z⟩
 1: ──PrepareNode──╰C──╰┤ ⟨Z ⊗ Z⟩

Additionally, we must remap the tape wires to match those available on our device.

>>> dev = qml.device("default.qubit", wires=2)
>>> fragment_tapes = [
...     qml.transforms.qcut.remap_tape_wires(t, dev.wires) for t in fragment_tapes
... ]

Next, each circuit fragment is expanded over :class:`~.MeasureNode` and
:class:`~.PrepareNode` configurations and a flat list of tapes is created:

.. code-block::

    expanded = [qml.transforms.qcut.expand_fragment_tape(t) for t in fragment_tapes]

    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    tapes = tuple(tape for c in configurations for tape in c)

Each configuration is drawn below:

>>> for t in tapes:
...     print(t.draw())

.. code-block::

     0: ──RX(0.531)──╭C──RY(-0.4)──╭┤ ⟨Z ⊗ I⟩ ╭┤ ⟨Z ⊗ Z⟩
     1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ I⟩ ╰┤ ⟨Z ⊗ Z⟩

     0: ──RX(0.531)──╭C──RY(-0.4)──╭┤ ⟨Z ⊗ X⟩
     1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ X⟩

     0: ──RX(0.531)──╭C──RY(-0.4)──╭┤ ⟨Z ⊗ Y⟩
     1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ Y⟩

     0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
     1: ──I────────╰C──╰┤ ⟨Z ⊗ Z⟩

     0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
     1: ──X────────╰C──╰┤ ⟨Z ⊗ Z⟩

     0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
     1: ──H────────╰C──╰┤ ⟨Z ⊗ Z⟩

     0: ──RX(0.3)─────╭Z──╭┤ ⟨Z ⊗ Z⟩
     1: ──H────────S──╰C──╰┤ ⟨Z ⊗ Z⟩

The last step is to execute the tapes and postprocess the results using
:func:`~.qcut_processing_fn`, which converts the results into a tensor network contraction.

>>> results = qml.execute(tapes, dev, gradient_fn=None)
>>> qml.transforms.qcut.qcut_processing_fn(
...     results,
...     communication_graph,
...     prepare_nodes,
...     measure_nodes,
... )
0.47165198882111165
