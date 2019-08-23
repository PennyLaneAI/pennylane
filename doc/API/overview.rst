.. _developer_overview:

Building a plugin
=================

Writing your own PennyLane plugin, to allow an external quantum library to take advantage of the automatic differentiation ability of PennyLane, is a simple and easy process. In this section, we will walk through the steps for creating your own PennyLane plugin. In addition, we also provide two default reference plugins — :mod:`'default.qubit' <.default_qubit>` for basic pure state qubit simulations, and :mod:`'default.gaussian' <.default_gaussian>` for basic Gaussian continuous-variable simulations.


What a plugin provides
----------------------

A quick primer on terminology of PennyLane plugins in this section:

* A plugin is an external Python package that provides additional quantum *devices* to PennyLane.

* Each plugin may provide one (or more) devices, that are accessible directly through PennyLane, as well as any additional private functions or classes.

* Depending on the scope of the plugin, you may wish to provide additional (custom) quantum operations and observables that the user can import.

.. important::

    In your plugin module, **standard NumPy** (*not* the wrapped NumPy module provided by PennyLane) should be imported in all places (i.e., ``import numpy as np``).


Creating your device
--------------------

The first step in creating your PennyLane plugin is to create your device class. This is as simple as importing the abstract base class :class:`~.Device` from PennyLane, and subclassing it:

.. code-block:: python

    from pennylane import Device

    class MyDevice(Device):
        """MyDevice docstring"""
        name = 'My custom device'
        short_name = 'example.mydevice'
        pennylane_requires = '0.1.0'
        version = '0.0.1'
        author = 'Ada Lovelace'

Here, we have begun defining some important class attributes that allow PennyLane to identify and use the device. These include:

* :attr:`~.Device.name`: a string containing the official name of the device

* :attr:`~.Device.short_name`: the string used to identify and load the device by users of PennyLane

* :attr:`~.Device.pennylane_requires`: the PennyLane version this device supports. Note that this class attribute supports pip *requirements.txt* style version ranges, for example:

  - ``pennylane_requires = "2"`` to support PennyLane version 2.x.x
  - ``pennylane_requires = ">=0.1.5,<0.6"`` to support a range of PennyLane versions

* :attr:`~.Device.version`: the version number of the device

* :attr:`~.Device.author`: the author of the device

Defining all these attributes is mandatory.


Supporting operators and observables
------------------------------------

You must further tell PennyLane about the operations and observables that your device supports as well as potential further capabilities, by providing the following class attributes/properties:

* :attr:`~.Device.operations`: a set of the supported PennyLane operations as strings, e.g.,

  .. code-block:: python

    operations = {"CNOT", "PauliX"}

  This is used to decide whether an operation is supported by your device in the default implementation of the public method :meth:`~.Device.supports_operation`.

* :attr:`~.Device.observables`: set of the supported PennyLane observables as strings, e.g.,

  .. code-block:: python

    observables = {"QuadOperator", "NumberOperator", "X", "P"}

  This is used to decide whether an observable is supported by your device in the default implementation of the public method :meth:`~.Device.supports_observable`.

* :attr:`~.Device._capabilities`: (optional) a dictionary containing information about the capabilities of the device. At the moment, only the key ``'model'`` is supported, which may return either ``'qubit'`` or ``'CV'``. Alternatively, you may use this class dictionary to return additional information to the user — this is accessible from the PennyLane frontend via the public method :meth:`~.Device.capabilities`.

For a better idea of how to best implement :attr:`~.Device.operations` and :attr:`~.Device.observables`, refer to the two reference plugins.


Applying operations
-------------------

Once all the class attributes are defined, it is necessary to define some required class methods, to allow PennyLane to apply operations to your device.

When PennyLane needs to evaluate a QNode, it accesses the :meth:`~.Device.execute` method of your plugin, which, by default performs the following process:

.. code-block:: python

    results = []

    with self.execution_context():
        self.pre_apply()
        for operation in queue:
            self.apply(operation.name, operation.wires, operation.parameters)
        self.post_apply()

        self.pre_measure()

        for obs in observables:
            if obs.return_type is Expectation:
                results.append(self.expval(obs.name, obs.wires, obs.parameters))
            elif obs.return_type is Variance:
                results.append(self.var(obs.name, obs.wires, obs.parameters))

        self.post_measure()

        return np.array(results)

where ``queue`` is a list of PennyLane :class:`~.Operation` instances to be applied, and ``observables`` is a list of PennyLane :class:`~.Observable` instances to be measured and returned. In most cases, there are therefore a minimum of three methods that any device **must** implement:

* :meth:`~.Device.apply`: This accepts an operation name (as a string), the wires (subsystems) to apply the operation to, and the parameters for the operation, and should apply the resulting operation to given wires of the device.

* :meth:`~.Device.expval`: This accepts an observable name (as a string), the wires (subsystems) to measure, and the parameters for the observable. It is expected to return the resulting expectation value from the device.

* :meth:`~.Device.var`: This accepts an observable name (as a string), the wires (subsystems) to measure, and the parameters for the observable. It is expected to return the resulting variance of the measured observable value from the device.

  .. note:: Currently, PennyLane only supports measurements that return a scalar value.

However, additional flexibility is sometimes required for interfacing with more complicated frameworks. In such cases, the following (optional) methods may also be implemented:

* :meth:`~.Device.__init__`: By default, this method receives the ``short_name`` of the device, number of wires (``self.num_wires``), and number of shots ``self.shots``. This is the right place to set up your device. You may add parameters while overwriting this method if you need to add additional options that the user must pass to the device on initialization. Make sure that you call ``super().__init__(self.short_name, wires, shots)`` at some point here.

* :meth:`~.Device.execution_context`: Here you may return a context manager for the circuit execution phase (see above). You can implement this method if the quantum library for which you are writing the device requires such an execution context while applying operations and measuring results from the device.

* :meth:`~.Device.pre_apply`: for any setup/code that must be executed before applying operations

* :meth:`~.Device.post_apply`: for any setup/code that must be executed after applying operations

* :meth:`~.Device.pre_measure`: for any setup/code that must be executed before measuring observables

* :meth:`~.Device.post_measure`: for any setup/code that must be executed after measuring observables

.. warning:: In advanced cases, the :meth:`~.Device.execute` method may be overwritten directly. This provides full flexibility for handling the device execution yourself. However, this may have unintended side-effects and is not recommended — if possible, try implementing a suitable subset of the methods provided above.


Identifying and installing your device
--------------------------------------

When performing a hybrid computation using PennyLane, one of the first steps is often to initialize the quantum device(s). PennyLane identifies the devices via their ``short_name``, which allows the device to be initialized in the following way:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device(short_name, wires=2)

where ``short_name`` is a string that uniquely identifies the device. The ``short_name`` has the following form: ``pluginname.devicename``. Examples include ``'default.qubit'`` and ``'default.gaussian'`` which are provided as reference plugins by PennyLane, as well as ``'strawberryfields.fock'``, ``'strawberryfields.gaussian'``, ``'projectq.simulator'``, and ``'projectq.ibm'``, which are provided by the `PennyLane StrawberryFields <https://github.com/XanaduAI/pennylane-sf>`_ and `PennyLane ProjectQ <https://github.com/XanaduAI/pennylane-pq>`_ plugins, respectively.

PennyLane uses a ``setuptools`` ``entry_points`` approach to plugin discovery/integration. In order to make the devices of your plugin accessible to PennyLane, simply provide the following keyword argument to the ``setup()`` function in your ``setup.py`` file:

.. code-block:: python

    devices_list = [
            'example.mydevice1 = MyModule.MySubModule:MyDevice1'
            'example.mydevice2 = MyModule.MySubModule:MyDevice2'
        ],
    setup(entry_points={'pennylane.plugins': devices_list})

where ``devices_list`` is a list of devices you would like to register, ``example.mydevice1`` is the short name of the device, and ``MyModule.MySubModule`` is the path to your Device class, ``MyDevice1``.

To ensure your device is working as expected, you can install it in developer mode using ``pip install -e pluginpath``, where ``pluginpath`` is the location of the plugin. It will then be accessible via PennyLane.


Testing
-------

All plugins should come with extensive unit tests, to ensure that the device supports the correct gates and observables, and is applying them correctly. For an example of a plugin test suite, see ``tests/test_default_qubit.py`` and ``tests/test_default_gaussian.py`` in the main `PennyLane repository <https://github.com/XanaduAI/pennylane/>`_.

In general, as all supported operations have their gradient formula defined and tested by PennyLane, testing that your device calculates the correct gradients is not required — just that it *applies* and *measures* quantum operations and observables correctly.


Supporting new operations
-------------------------

If you would like to support an operation or observable that is not currently supported by PennyLane, you can subclass the :class:`~.Operation` and :class:`~.Observable` classes, and define the number of parameters the operation takes, and the number of wires the operation acts on. For example, to define the Ising gate :math:`XX_\phi` depending on parameter :math:`\phi`,

.. code-block:: python

    class Ising(Operation):
        """Ising gate"""
        num_params = 1
        num_wires = 2
        par_domain = 'R'
        grad_method = 'A'
        grad_recipe = None

where

* :attr:`~.Operation.num_params`: the number of parameters the operation takes

* :attr:`~.Operation.num_wires`: the number of wires the operation acts on

* :attr:`~.Operation.par_domain`: the domain of the gate parameters; ``'N'`` for natural numbers (including zero), ``'R'`` for floats, ``'A'`` for arrays of floats/complex numbers, and ``None`` if the gate does not have free parameters

* :attr:`~.Operation.grad_method`: the gradient computation method; ``'A'`` for the analytic method, ``'F'`` for finite differences, and ``None`` if the operation may not be differentiated

* :attr:`~.Operation.grad_recipe`: The gradient recipe for the analytic ``'A'`` method. This is a list with one tuple per operation parameter. For parameter :math:`k`, the tuple is of the form :math:`(c_k, s_k)`, resulting in a gradient recipe of

  .. math:: \frac{d}{d\phi_k}f(O(\phi_k)) = c_k\left[f(O(\phi_k+s_k))-f(O(\phi_k-s_k))\right].

  where :math:`f` is an expectation value that depends on :math:`O(\phi_k)`, an example being

  .. math:: f(O(\phi_k)) = \braket{0 | O^{\dagger}(\phi_k) \hat{B} O(\phi_k) | 0}

  which is the simple expectation value of the operator :math:`\hat{B}` evolved via the gate :math:`O(\phi_k)`.

Note that if ``grad_recipe = None``, the default gradient recipe is :math:`(c_k, s_k)=(1/2, \pi/2)` for every parameter.

The user can then import this operation directly from your plugin, and use it when defining a QNode:

.. code-block:: python

    import pennylane as qml
    from MyModule.MySubModule import Ising

    @qnode(dev1)
    def my_qfunc(phi):
        qml.Hadamard(wires=0)
        Ising(phi, wires=[0,1])
        return qml.expval(qml.PauliZ(0))

.. warning::

    If you are providing custom operations not natively supported by PennyLane, it is recommended that the plugin unittests **do** provide tests to ensure that PennyLane returns the correct gradient for the custom operations.


Supporting new CV operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom continuous-variable operations or observables, the :class:`~.CVOperation` or :class:`~.CVObservable` classes must be subclassed instead.

In addition, for Gaussian CV operations, you may need to provide the static class method :meth:`~.CV._heisenberg_rep` that returns the Heisenberg representation of the operator given its list of parameters:

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

* For operations, the ``_heisenberg_rep`` method should return the matrix of the linear transformation carried out by the gate for the given parameter values. This is used internally for calculating the gradient using the analytic method (``grad_method = 'A'``).

* For observables, this method should return a real vector (first-order observables) or symmetric matrix (second-order observables) of coefficients which represent the expansion of the observable in the basis of monomials of the quadrature operators.

  - For single-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x, \p)`.
  - For multi-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`, where :math:`\x_k` and :math:`\p_k` are the quadrature operators of qumode :math:`k`.

Non-Gaussian CV operations and observables are currently only supported via the finite difference method of gradient computation.
