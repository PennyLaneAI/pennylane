.. role:: html(raw)
   :format: html

Architectural overview
======================

PennyLane allows optimization and machine learning of quantum and hybrid
quantum-classical computations by integrating several key components. A
*quantum device* can be matched with a *quantum node* to return statistics upon
evaluation. To process these, *interfaces* allow using familiar classical
frameworks. Built-in *optimizers* help on the way to finding desired
parameters to the quantum circuit.

In most cases, computations with PennyLane are performed on a local machine
using a simulator. Through using PennyLane plugins, one can, however, also
utilize remote quantum devices and simulators.

Components of PennyLane
#######################

.. image:: architecture_diagram.png
    :width: 800px

Devices
*******

In PennyLane, the abstraction of a quantum device is encompassed within the
:class:`~.Device` class, making it one of the basic components of the
library. It includes basic functionality that is shared for quantum
devices, independent of the qubit and CV models. PennyLane gives access to
multiple simulators and hardware chips through its plugins, each of these
devices is implemented as a custom class. These classes have the
``Device`` class as their parent class.

A ``QNode`` is another important component in PennyLane (later detailed), they
serve as the abstraction for quantum circuits to be run on a device.

The purpose of the ``Device`` class can be summarized as:

* Providing a common API for QNodes to execute a quantum circuit and request
  the measurement of the associated observable.
* Providing an easy way of developing a new device for PennyLane that
  can be used with QNodes.

Qubit based devices can use shared utilities by using the
:class:`~.QubitDevice`.

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
     - ``"forest.qvm"``, ``qvm_url`` needs to be unset to use the pyQVM
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
   :widths: 35 45 60 
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
   * - `Forest QVMDevice <https://pennylane-forest.readthedocs.io/en/latest/code/qvm.html>`__
     - ``"forest.qvm"``, ``qvm_url`` needs to be set to use the Lisp QVM
     - Forest QVM device supporting both the Rigetti Lisp QVM, as well as the built-in pyQuil pyQVM
   * - `AQT Ideal ion-trap simulator <https://pennylane-aqt.readthedocs.io/en/latest/devices.html#ideal-ion-trap-simulator>`__
     - ``"aqt.sim"``
     - Ideal noiseless ion-trap simulator
   * - `AQT Noisy ion-trap simulator <https://pennylane-aqt.readthedocs.io/en/latest/devices.html#noisy-ion-trap-simulator>`__
     - ``"aqt.noisy_sim"``
     - Noisy ion-trap simulator for more realisatic simulations

:html:`</div></div>`


:html:`<div class="hw_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content3" class="collapsed"><p class="first admonition-title">Hardware devices (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content3" class="collapse" data-parent="#aside1" style="height: 0px;">`

.. list-table::
   :widths: 35 45 60 
   :header-rows: 1

   * - **Device**
     - **Shortname**
     - **Description**
   * - `IBM Q Experience <https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html>`__
     - ``"qiskit.ibmq"`` *(must specify a hardware backend)*
     - IBM Q hardware device, queue based access to IBMQ backends
   * - `Forest QPUDevice <https://pennylane-forest.readthedocs.io/en/latest/code/qpu.html>`__
     - ``"forest.qpu"``
     - Forest QPU device, session based access to Rigetti QPUs

:html:`</div></div>`

:html:`<div class="photonic_list" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content4" class="collapsed"><p class="first admonition-title">Photonic devices (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content4" class="collapse" data-parent="#aside1" style="height: 0px;">`

.. list-table::
   :widths: 35 45 60 
   :header-rows: 1

   * - **Device**
     - **Shortname**
     - **Description**
   * - :class:`~.DefaultGaussian`
     - ``"default.gaussian"``
     - Default gaussian device
   * - `Strawberry Fields Fock device <https://pennylane-sf.readthedocs.io/en/latest/devices/fock.html>`__
     - ``"strawberryfields.fock"``
     - Fock device giving access to the Strawberry Fields Fock state simulator backend
   * - `Strawberry Fields Gaussian device <https://pennylane-sf.readthedocs.io/en/latest/devices/gaussian.html>`__
     - ``"strawberryfields.gaussian"``
     - Gaussian device giving access to the Strawberry Fields Fock state simulator backend


:html:`</div></div>`

QNodes
******

A  quantum  node or ``QNode`` (represented by a subclass to
:class:`~.BaseQNode`) is an encapsulation of a function :math:`f(x;\theta)=R^m\rightarrow R^n`
that is executed using quantum information processing on a quantum
device.

Each ``QNode`` represents the quantum circuit by building a
:class:`~.CircuitGraph` instance, but the way differentiation is done is custom
to the differentiation method offered by the ``QNode``.

For further details on QNodes, and for a full list of QNodes, refer to the 
:doc:`/code/qml_qnodes` module.

Interfaces
**********

The integration between classical and quantum computations is encompassed by
interfaces.

We refer to the :ref:`intro_interfaces` page for a more in-depth introduction
and a list of available interfaces.

Optimizers
**********

Optimizers are objects which can be used to automatically update the parameters
of a quantum or hybrid machine learning model.

We refer to the :ref:`intro_ref_opt` page for a more in-depth introduction
and a list of available optimizers.

Key design details
##################

The following are key design details related to how PennyLane works internally.

Queuing of operators
********************

In PennyLane, the construction of quantum gates is separated from the specific
quantum node (:class:`~.BaseQNode`) that they belong to. However, including
logic for this when creating an instance of :class:`~.Operator` does not align
with the current architecture. Therefore, there is a need to use a high-level
object that holds information about the relationship between quantum gates and
a quantum node.

The :class:`~.QueuingContext` class realizes this by providing access to the current
QNode.  Furthermore, it provides the flexibility to have multiple objects
record the creation of quantum gates.

The ``QueuingContext`` class both acts as the abstract base class for all
classes that expose a queue for Operations (so-called contexts), as well as the
interface to said queues. The active contexts contain maximally one QNode and
an arbitrary number of other contexts like the :class:`~.OperationRecorder`.

Variables
*********

Circuit parameters in PennyLane are tracked and updated using
:class:`~.Variable`. They play a key role in the evaluation of a ``QNode``.

We refer to the :ref:`qml_variable` page for a more in-depth description of how
``Variables`` are used during execution.
