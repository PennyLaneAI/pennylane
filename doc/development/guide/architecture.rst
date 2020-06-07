.. role:: html(raw)
   :format: html

Architectural overview
======================

TODO: intro sentence

Components of PennyLane
#######################

TODO: intro to components (?)

TODO: more high-resolution picture ?

.. image:: architecture_diagram.png
	:width: 800px

1. `Devices <https://pennylane.ai/plugins.html>`_
2. `QNodes <https://pennylane.readthedocs.io/en/stable/code/qml_qnodes.html>`_

  - `Qubit operations <https://pennylane.readthedocs.io/en/latest/code/qml_operation.html#qubit-operations>`_ and `CV operations <https://pennylane.readthedocs.io/en/latest/code/qml_operation.html#cv-operation-base-classes>`_ 
  - `Templates <https://pennylane.readthedocs.io/en/latest/introduction/templates.html>`_
  - `Qubit observables <https://pennylane.readthedocs.io/en/latest/introduction/operations.html#qubit-observables>`_ and `CV observables <https://pennylane.readthedocs.io/en/latest/introduction/operations.html#cv-observables>`_
  - `Statistics <https://pennylane.readthedocs.io/en/stable/introduction/measurements.html>`_

3. `Interfaces <https://pennylane.readthedocs.io/en/latest/code/qml_interfaces.html>`_
4. `Optimizers <https://pennylane.readthedocs.io/en/stable/introduction/optimizers.html>`_

Devices
*******

In PennyLane, the abstraction of a quantum device is encompassed within the
:class:`Device` class, making it one of the basic components of the
library. It includes basic functionality that is shared for quantum
devices, independent of the qubit and CV models. PennyLane gives access to
multiple simulators and hardware chips through its plugins, each of these
devices is implemented as a custom class. These classes have the
``Device`` class as their parent class.

``QNodes`` are another important component in PennyLane, they serve as the
abstraction for quantum circuits to be run on a device.

The purpose of the ``Device`` class can be summarized as:

* Providing a common API for QNodes to execute a quantum circuit and request
  the measurement of the associated observable.
* Providing an easy way of developing a new device for PennyLane that
  can be used with QNodes.

Qubit based devices can use shared utilities by using the
:class:`QubitDevice`.

:html:`<div class="state_vec_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed"><p class="first admonition-title">Statevector simulators (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">`

.. list-table::
   :widths: 35 45 60 
   :header-rows: 1

   * - **Device**
     - **Shortname**
     - **Description**
   * - :class:`~.DefaultQubit`
     - ``"default.qubit"``
     - Default qubit device
   * - :class:`~.DefaultQubitTF`
     - ``"default.qubit.tf"``
     - Default qubit device written using TensorFlow
   * - :class:`~.DefaultTensor`
     - ``"default.tensor"``
     - Experimental Tensor Network simulator device
   * - :class:`~.DefaultTensorTF`
     - ``"default.tensor.tf"``
     - Experimental Tensor Network simulator device written using TensorFlow
   * - `Qiskit AerDevice <https://pennylaneqiskit.readthedocs.io/en/latest/devices/aer.html>`__
     - ``"qiskit.aer", backend="statevector_simulator"``
     - Qiskit Aer simulator in C++ (``statevector_simulator`` backend)
   * - `Qiskit BasicAer <https://pennylaneqiskit.readthedocs.io/en/latest/devices/basicaer.html>`__
     - ``"qiskit.basicaer", backend="statevector_simulator"``
     - Qiskit simulator in native Python with fewer dependencies (``statevector_simulator`` backend)
   * - `Cirq SimulatorDevice <https://pennylane-cirq.readthedocs.io/en/latest/devices/simulator.html>`__
     - ``"cirq.simulator"``
     - Cirq's simulator backend
   * - `Cirq MixedSimulatorDevice <https://pennylane-cirq.readthedocs.io/en/latest/devices/mixed_simulator.html>`__
     - ``"cirq.mixedsimulator"``
     - Cirq's density matrix simulator backend
   * - `Forest QVMDevice <https://pennylane-forest.readthedocs.io/en/latest/code/qvm.html>`__
     - ``"forest.qvm"`` (``qvm_url`` not set for pyQVM)
     - Forest QVM device supporting both the Rigetti Lisp QVM, as well as the built-in pyQuil pyQVM
   * - `Forest WavefunctionDevice <https://pennylane-forest.readthedocs.io/en/latest/code/wavefunction.html>`__
     - ``"forest.wavefunction"``
     - Wavefunction simulator device
   * - `Forest NumpyWavefunctionDevice <https://pennylane-forest.readthedocs.io/en/latest/code/numpy_wavefunction.html>`__
     - ``"forest.numpy_wavefunction"``
     - NumpyWavefunction simulator device

:html:`</div></div>`

:html:`<div class="hw_sim_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content2" class="collapsed"><p class="first admonition-title">Hardware simulators (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content2" class="collapse" data-parent="#aside1" style="height: 0px;">`

.. list-table::
   :widths: 25 30 50
   :header-rows: 1

   * - **Device**
     - **Shortname**
     - **Description**
   * - `Qiskit AerDevice <https://pennylaneqiskit.readthedocs.io/en/latest/devices/aer.html>`__
     - ``"qiskit.aer", backend="qasm_simulator"``
     - Qiskit Aer simulator in C++ (``qasm_simulator`` backend)
   * - `Qiskit BasicAer <https://pennylaneqiskit.readthedocs.io/en/latest/devices/basicaer.html>`__
     - ``"qiskit.basicaer", backend="qasm_simulator"``
     - Qiskit simulator in native Python with fewer dependencies (``qasm_simulator`` backend)
   * - `IBM Q Experience <https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html>`__
     - ``"qiskit.ibmq", backend="ibmq_qasm_simulator"``
     - IBM Q hardware device simulator
   * - `QVMDevice <https://pennylane-forest.readthedocs.io/en/latest/code/qvm.html>`__
     - ``"forest.qvm"``, (``qvm_url`` set for Lisp QVM)
     - Forest QVM device supporting both the Rigetti Lisp QVM, as well as the built-in pyQuil pyQVM

:html:`</div></div>`


:html:`<div class="hw_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content3" class="collapsed"><p class="first admonition-title">Hardware devices (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content3" class="collapse" data-parent="#aside1" style="height: 0px;">`

.. list-table::
   :widths: 25 30 50
   :header-rows: 1

   * - **Device**
     - **Shortname**
     - **Description**
   * - `IBM Q Experience <https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html>`__
     - ``"qiskit.ibmq"``, (must specify a hardware backend)
     - IBM Q hardware device, queue based access to IBMQ backends
   * - `Forest QPUDevice <https://pennylane-forest.readthedocs.io/en/latest/code/qpu.html>`__
     - ``"forest.qpu"``
     - Forest QPU device, session based access to Rigetti QPUs

:html:`</div></div>`

* QPUDevice (PennyLane-Forest)
  access to QPU (session based, paid access)

* IBMQDeivce (PennyLane-Qiskit)
  access to QPU (queue based, free access)

:html:`<div class="photonic_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content4" class="collapsed"><p class="first admonition-title">Photonic devices (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content4" class="collapse" data-parent="#aside1" style="height: 0px;">`
**Photonic devices**

* default.gaussian
* fock.simulator
* guassian.simulator

:html:`</div></div>`

TODO: devices table with links 

QNodes
******

A  quantum  node or ``QNode`` (represented by a subclass to
:class:`~.BaseQNode`) is an encapsulation of a function :math:`f(x;θ):R^m→R^n`
that is executed by means of quantum information processing on a quantum
device.

See for a list of qnodes :ref:`qml_qnodes`.

Design decisions
================

The following are more in-depth points related to how PennyLane works
internally.

The page could include a section on the following:

* Queing behaviour of operations

TODO: can this be somehow shared with the docstring of the ``QueingContext``? 

    In PennyLane, the construction of quantum gates is separated from the
    specific quantum node (:class:`BaseQNode`) that they belong to. However,
    including logic for this when creating an instance of :class:`Operator`
    does not align with the current architecture. Therefore, there is a need to
    use a high level object that holds information about the relationship
    between quantum gates and a quantum node.

    The ``QueuingContext`` class realizes this by providing access to the
    current QNode.  Furthermore, it provides the flexibility to have multiple
    objects record the creation of quantum gates.

    The QueuingContext class both acts as the abstract base class for all
    classes that expose a queue for Operations (so-called contexts), as well
    as the interface to said queues. The active contexts contain maximally one QNode
    and an arbitrary number of other contexts like the OperationRecorder.

* Variable system

  TODO: add general description, symbolic computation
