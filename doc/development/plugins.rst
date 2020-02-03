.. role:: html(raw)
   :format: html

.. _plugin_overview:

Building a plugin
=================

Writing your own PennyLane plugin, to allow an external quantum library to take advantage of the
automatic differentiation ability of PennyLane, is a simple and easy process. In this section,
we will walk through the steps for creating your own PennyLane plugin. In addition, we also
provide two default reference plugins — :mod:`'default.qubit' <.default_qubit>` for basic pure
state qubit simulations, and :mod:`'default.gaussian' <.default_gaussian>` for basic
continuous-variable simulations.


What a plugin provides
----------------------

A quick primer on terminology of PennyLane plugins in this section:

* A plugin is an external Python package that provides additional quantum *devices* to PennyLane.

* Each plugin may provide one (or more) devices, that are accessible directly through
  PennyLane, as well as any additional private functions or classes.

* Depending on the scope of the plugin, you may wish to provide additional (custom)
  quantum operations and observables that the user can import.

.. important::

    In your plugin module, **standard NumPy** (*not* the wrapped Autograd version of NumPy)
    should be imported in all places (i.e., ``import numpy as np``).


Creating your device
--------------------

.. note::

    A handy `plugin template repository <https://github.com/XanaduAI/pennylane-plugin-template>`__
    is available, providing the boilerplate and file structure required to easily create your
    own PennyLane plugin, as well as a suite of integration tests to ensure the plugin
    returns correct expectation values.

The first step in creating your PennyLane plugin is to create your device class.
This is as simple as importing the abstract base class :class:`~.QubitDevice` from PennyLane,
and subclassing it:

.. code-block:: python

    from pennylane import QubitDevice

    class MyDevice(QubitDevice):
        """MyDevice docstring"""
        name = 'My custom device'
        short_name = 'example.mydevice'
        pennylane_requires = '0.1.0'
        version = '0.0.1'
        author = 'Ada Lovelace'

Here, we have begun defining some important class attributes that allow PennyLane to identify
and use the device. These include:

* :attr:`.Device.name`: a string containing the official name of the device

* :attr:`.Device.short_name`: the string used to identify and load the device by users of PennyLane

* :attr:`.Device.pennylane_requires`: the PennyLane version this device supports.
  Note that this class attribute supports pip *requirements.txt* style version ranges,
  for example:

  - ``pennylane_requires = "2"`` to support PennyLane version 2.x.x
  - ``pennylane_requires = ">=0.1.5,<0.6"`` to support a range of PennyLane versions

* :attr:`.Device.version`: the version number of the device

* :attr:`.Device.author`: the author of the device

Defining all these attributes is mandatory.


Supporting operations
---------------------

You must further tell PennyLane about the operations that your device supports
as well as potential further capabilities, by providing the following class attributes/properties:

* :attr:`.Device.operations`: a set of the supported PennyLane operations as strings, e.g.,

  .. code-block:: python

    operations = {"CNOT", "PauliX"}

  See :doc:`/introduction/operations` for a full list of operations
  supported by PennyLane.

  If your device does not natively support an operation that has the
  :meth:`~.Operation.decomposition` static method defined, PennyLane will
  attempt to decompose the operation before calling the device. For example,
  the :class:`~.Rot` `decomposition method <../_modules/pennylane/ops/qubit.html#Rot.decomposition>`_ will
  decompose the single-qubit rotation gate to :class:`~.RZ` and :class:`~.RY` gates.

  .. note::

      If the convention differs between the built-in PennyLane operation
      and the corresponding operation in the targeted framework, ensure that the
      conversion between the two conventions takes place automatically
      by the plugin device.

* :attr:`.Device._capabilities`: a dictionary containing information about the capabilities of
  the device. Keys currently supported include:

  * ``'model'`` (*str*): either ``'qubit'`` or ``'CV'``.

  * ``'inverse_operations'`` (*bool*): ``True`` if the device supports
    applying the inverse of operations. Operations which should be inverted
    have the property ``operation.inverse == True``.

Adding arguments to your device
--------------------------------

.. important::

    PennyLane supports both qubit and continuous-variable (CV) devices. However, from
    here onwards, we will demonstrate plugin development focusing on qubit-based devices
    inheriting from the :class:`~.QubitDevice` class.

Defining the ``__init__.py`` method of a custom device is not necessary; by default,
the :class:`~.QubitDevice` initialization will be called, where the user can pass the
following arguments:

* ``wires`` (*int*): the number of wires on the device.

* ``shots=1000`` (*int*): number of circuit evaluations/random samples used to estimate
  expectation values of observables in non-analytic mode.

* ``analytic=True`` (*bool*): If ``True``, the device calculates probability, expectation
  values, and variances analytically. If ``False``, a finite number of samples
  are used to estimate these quantities. Note that hardware devices should always set
  ``analytic=False``.

To add your own device arguments, or to override any of the above defaults, simply
overwrite the ``__init__.py`` method. For example, consider a device where the number
of wires is fixed to ``24``, cannot be used in analytic mode, and can accept a dictionary
of low-level hardware control options:

.. code-block:: python3

    class CustomDevice(QubitDevice):
        name = 'My custom device'
        short_name = 'example.mydevice'
        pennylane_requires = '0.1.0'
        version = '0.0.1'
        author = 'Ada Lovelace'

        operations = {"PauliX", "RX", "CNOT"}
        observables = {"PauliZ", "PauliX", "PauliY"}

        def __init__(self, shots=1024, hardware_options=None):
            super().__init__(wires=24, shots=shots, analytic=False)
            self.hardware_options = hardware_options or hardware_defaults

Note that we have also overridden the default shot number.

The user can now pass any of these arguments to the PennyLane device loader:

>>> dev = qml.device("example.mydevice", hardware_options={"t2": 0.1})
>>> dev.hardware_options
{"t2": 0.1}


Device execution
----------------

Once all the class attributes are defined, it is necessary to define some required class
methods, to allow PennyLane to apply operations and measure observables on your device.

To execute operations on the device, the following methods **must** be defined:

.. currentmodule:: pennylane.QubitDevice

.. autosummary::

    apply
    probability

The :class:`~.QubitDevice` class
provides the following convenience methods that may be used by the plugin:

.. autosummary::

    active_wires
    marginal_prob

In addition, if your qubit device generates its own computational basis samples for measured modes
after execution, you need to overwrite the following method:

.. autosummary::

    generate_samples

:meth:`~.generate_samples` should return samples with shape ``(dev.shots, dev.num_wires)``.
Furthermore, PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
:math:`q_0` is the most significant bit.

And thats it! The device has inherited :meth:`~.QubitDevice.expval`, :meth:`~.QubitDevice.var`,
and :meth:`~.QubitDevice.sample` methods, that accepts an observable (or tensor product of
observables) and returns the corresponding measurement statistic.



:html:`<div class="caution admonition" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed"><p class="first admonition-title">Advanced execution control (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">`

Additional flexibility is sometimes required for interfacing with more
complicated frameworks.

When PennyLane needs to evaluate a QNode, it accesses the :meth:`~.QubitDevice.execute` method of
your plugin, which, by default performs the following process:

.. code-block:: python

    self.check_validity(circuit.operations, circuit.observables)

    # apply all circuit operations
    self.apply(circuit.operations, rotations=circuit.diagonalizing_gates)

    # generate computational basis samples
    if (not self.analytic) or circuit.is_sampled:
        self._samples = self.generate_samples()

    # compute the required statistics
    results = self.statistics(circuit.observables)

    return self._asarray(results)

where

* ``circuit`` is a :class:`~.CircuitGraph` object

* :attr:`circuit.operations <pennylane.CircuitGraph.operations>` are the user-provided
  operations to be executed

* :attr:`circuit.observables <pennylane.CircuitGraph.observables>` are the user-provided
  observables to be measured

* :attr:`circuit.diagonalizing_gates <pennylane.CircuitGraph.diagonalizing_gates>` are the
  gates that rotate the circuit prior to measurement so that computational basis
  measurements are performed in the eigenbasis of the requested observables

* :meth:`.QubitDevice.statistics` returns the results of :meth:`.QubitDevice.expval`,
  :meth:`~.QubitDevice.var`, or :meth:`~.QubitDevice.sample` depending on the type
  of observable.

In advanced cases, the :meth:`.QubitDevice.execute` method, as well as
:meth:`.QubitDevice.statistics`, may be overwritten directly.
This provides full flexibility for handling the device execution yourself. However,
this may have unintended side-effects and is not recommended.


:html:`</div></div>`

.. _installing_plugin:


Identifying and installing your device
--------------------------------------

When performing a hybrid computation using PennyLane, one of the first steps is often to
initialize the quantum device(s). PennyLane identifies the devices via their ``short_name``,
which allows the device to be initialized in the following way:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device(short_name, wires=2)

where ``short_name`` is a string that uniquely identifies the device. The ``short_name``
should have the form ``pluginname.devicename``, using periods for delimitation.

PennyLane uses a setuptools ``entry_points`` approach to plugin discovery/integration.
In order to make the devices of your plugin accessible to PennyLane, simply provide the
following keyword argument to the ``setup()`` function in your ``setup.py`` file:

.. code-block:: python

    devices_list = [
            'example.mydevice1 = MyModule.MySubModule:MyDevice1'
            'example.mydevice2 = MyModule.MySubModule:MyDevice2'
        ],
    setup(entry_points={'pennylane.plugins': devices_list})

where

* ``devices_list`` is a list of devices you would like to register,

* ``example.mydevice1`` is the short name of the device, and

* ``MyModule.MySubModule`` is the path to your Device class, ``MyDevice1``.

To ensure your device is working as expected, you can install it in developer mode using
``pip install -e pluginpath``, where ``pluginpath`` is the location of the plugin. It will
then be accessible via PennyLane.

Testing
-------

All plugins should come with extensive unit tests, to ensure that the device supports the correct
gates and observables, and is applying them correctly. For an example of a plugin test suite, see
``tests/test_default_qubit.py`` and ``tests/test_default_gaussian.py`` in the main
`PennyLane repository <https://github.com/XanaduAI/pennylane/>`_.

Integration tests to check that the expectation values, variance, and samples are correct for
various circuits and observables are provided in the
`PennyLane Plugin Template <https://github.com/XanaduAI/pennylane-plugin-template>`__ repository.

In general, as all supported operations have their gradient formula defined and tested by
PennyLane, testing that your device calculates the correct gradients is not required — just
that it *applies* and *measures* quantum operations and observables correctly.


Supporting new operations
-------------------------

If you would like to support an operation that is not currently supported by
PennyLane, you can subclass the :class:`~.Operation` class, and
define the number of parameters the operation takes, and the number of wires the operation
acts on. For example, to define a custom gate depending on parameter :math:`\phi`,

.. code-block:: python

    class CustomGate(Operation):
        """Custom gate"""
        num_params = 2
        num_wires = 1
        par_domain = 'R'

        grad_method = 'A'
        grad_recipe = None

        @staticmethod
        def _matrix(*params):
            """Returns the matrix representation of the operator for the
            provided parameter values, in the computational basis."""
            return np.array([[params[0], 1], [1, -params[1]]]) / np.sqrt(2)

        @staticmethod
        def decomposition(*params, wires):
            """(Optional) Returns a list of PennyLane operations that decompose
            the custom gate."""
            return [qml.RZ(params[0]/2, wires=wires[0]), qml.PauliX(params[1], wires=wires[0])]

where

* :attr:`~.Operator.num_params`: the number of parameters the operation takes

* :attr:`~.Operator.num_wires`: the number of wires the operation acts on.

  You may use :attr:`pennylane.operation.All` to represent an operation that
  acts on all wires, or :attr:`pennylane.operation.Any` to represent an operation that
  can act on any number of wires (for example, operations where the number of
  wires they act on is a function of the operation parameter).

* :attr:`~.Operator.par_domain`: the domain of the gate parameters; ``'N'`` for natural
  numbers (including zero), ``'R'`` for floats, ``'A'`` for arrays of floats/complex numbers,
  and ``None`` if the gate does not have free parameters

* :attr:`~.Operation.grad_method`: the gradient computation method; ``'A'`` for the analytic
  method, ``'F'`` for finite differences, and ``None`` if the operation may not be differentiated

* :attr:`~.Operation.grad_recipe`: The gradient recipe for the analytic ``'A'`` method.
  This is a list with one tuple per operation parameter. For parameter :math:`k`, the tuple is of
  the form :math:`(c_k, s_k)`, resulting in a gradient recipe of

  .. math:: \frac{d}{d\phi_k}f(O(\phi_k)) = c_k\left[f(O(\phi_k+s_k))-f(O(\phi_k-s_k))\right].

  where :math:`f` is an expectation value that depends on :math:`O(\phi_k)`, an example being

  .. math:: f(O(\phi_k)) = \braket{0 | O^{\dagger}(\phi_k) \hat{B} O(\phi_k) | 0}

  which is the simple expectation value of the operator :math:`\hat{B}` evolved via the gate
  :math:`O(\phi_k)`.

  Note that if ``grad_recipe = None``, the default gradient recipe is
  :math:`(c_k, s_k)=(1/2, \pi/2)` for every parameter.

The user can then import this operation directly from your plugin, and use it when defining a QNode:

.. code-block:: python

    import pennylane as qml
    from MyModule.MySubModule import CustomGate

    @qnode(dev1)
    def my_qfunc(phi):
        qml.Hadamard(wires=0)
        CustomGate(phi, theta, wires=0)
        return qml.expval(qml.PauliZ(0))

.. warning::

    If you are providing custom operations not natively supported by PennyLane, it is recommended
    that the plugin unit tests **do** provide tests to ensure that PennyLane returns the correct
    gradient for the custom operations.

Supporting new observables
--------------------------

Custom observables can be added in an identical manner to operations above, but with three
small changes:

* The :class:`~.Observable` class should instead be subclassed.

* The class attribute :attr:`~.Observable.eigvals` should be defined, returning
  the eigenvalues of the observable.

* The method :meth:`~.Observable.diagonalizing_gates` should be defined. This method
  returns a list of PennyLane :class:`~.Operation` objects that diagonalize the observable
  in the computational basis. This is used to support devices that can only perform
  measurements in the computational basis.

For example:

.. code-block:: python

    class CustomObservable(Observable):
        """Custom observable"""
        num_params = 0
        num_wires = 1
        par_domain = None
        eigvals = np.array([0.2, 0.1])

        def diagonalizing_gates(self):
            return [PauliX(wires=self.wires), Hadamard(wires=self.wires)]

        @staticmethod
        def _matrix(*params):
            return np.array([[0, 1], [1, 0]]) / np.sqrt(2)


:html:`<div class="note admonition" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed"><p class="first admonition-title">CV devices and operations (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">`

**Note: CV devices currently subclass from the base** :class:`~.Device` **class. However, this
class is deprecated, and a new** ``CVDevice`` **class will be available soon.**

For custom continuous-variable operations or observables, the :class:`~.CVOperation` or
:class:`~.CVObservable` classes must be subclassed instead.

In addition, for Gaussian CV operations, you may need to provide the static class method
:meth:`~.CV._heisenberg_rep` that returns the Heisenberg representation of the operator given
its list of parameters:

.. code-block:: python

    class Custom(CVOperation):
        """Custom gate"""
        n_params = 2
        n_wires = 1
        par_domain = 'R'
        grad_method = 'A'
        grad_recipe = None

        @staticmethod
        def _heisenberg_rep(params):
            return function(params)

* For operations, the ``_heisenberg_rep`` method should return the matrix of the linear
  transformation carried out by the gate for the given parameter values. This is used internally for
  calculating the gradient using the analytic method (``grad_method = 'A'``).

* For observables, this method should return a real vector (first-order observables) or symmetric
  matrix (second-order observables) of coefficients which represent the expansion of the observable in
  the basis of monomials of the quadrature operators.

  - For single-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x, \p)`.

  - For multi-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1,
    \ldots)`, where :math:`\x_k` and :math:`\p_k` are the quadrature operators of qumode :math:`k`.

Non-Gaussian CV operations and observables are currently only supported via the finite difference
method of gradient computation.

:html:`</div></div>`
