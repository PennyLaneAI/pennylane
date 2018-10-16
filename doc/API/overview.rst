Overview of the developer API
=============================

Writing your own OpenQML plugin, to allow an external quantum library to take advantage of the automatic differentiation ability of OpenQML, is a simple and easy process. In this section, we will walk through the steps for creating your own OpenQML plugin. In addition, we also provide two default reference plugins - ``'default.qubit'`` for basic pure state qubit simulations, and ``'default.gaussian'`` for basic Gaussian continuous-variable simulations.


.. note::

    A quick primer on terminology of OpenQML plugins in this section. A plugin is an external Python package that provides additional quantum *devices* to OpenQML. Each plugin may provide one (or more) devices, that are accessible directly by OpenQML, as well as any additional private functions or classes.

    Once installed, these devices can be loaded directly from OpenQML without any additional steps required by the user - however, depending on the scope of the plugin, the user may have to import additional operations.

.. note:: In your plugin module, vanilla NumPy should be imported in all places: ``import numpy as np``.


The device short name
---------------------

When performing a hybrid computation using OpenQML, one of the first steps is often to initialize the quantum devices which will be used by quantum nodes (:class:`~.QNode`). To do so, you load the device as follows:

.. code-block:: python

    import openqml as qm
    dev1 = qm.device(short_name)

where ``short_name`` is a string which uniquely identifies the device provided. In general, the short name has the following form: ``pluginname.devicename``. Examples include ``default.qubit`` and ``default.gaussian`` which are provided as reference plugins by OpenQML, as well as ``strawberryfields.fock``, ``strawberryfields.gaussian``, ``projectq.ibmqx4``, which are provided by the `StrawberryFields <https://github.com/XanaduAI/openqml-sf>`_ and `ProjectQ <https://github.com/XanaduAI/openqml-pq>`_ OpenQML plugins respectively.

Creating your device
--------------------

The first step in creating your OpenQML plugin is creating your device class. This is as simple as importing the abstract base class :class:`~.Device` from OpenQML, and subclassing it:

.. code-block:: python

    from openqml import Device

    class MyDevice(Device):
        """MyDevice docstring"""
        name = 'My custom device'
        short_name = 'example.mydevice'
        api_version = '0.1.0'
        version = '0.0.1'
        author = 'John Smith'

Here, we have begun defining some important class attributes ('identifiers') that allow OpenQML to identify the device. These include:

* ``name``: a string containing the official name of the device
* ``short_name``: the string used to identify and load the device by users of OpenQML
* ``api_version``: the version number of OpenQML that this device was made for. If the user attempts to load the device on a different version of OpenQML, a :class:`~.DeviceError` will be raised.
* ``version``: the version number of the device.
* ``author``: the author of the device.

Note that, apart from ``short_name``, these are all optional. ``short_name``, however, **must** be defined, so that it is accessible from the OpenQML interface.

Supporting operators and expectations
-------------------------------------

There are also three private class attributes to be defined for your custom device:

* ``_operator_map``: a dictionary mapping an OpenQML supported operation (string) to the corresponding function/operation in the plugin. The keys are accessible to the user via the public attribute :attr:`~.Device.gates` and public method :meth:`~.Device.supported`.

* ``_observable_map``: a dictionary mapping an OpenQML supported expectation (string) to the corresponding function/operation in the plugin. The keys are accessible to the user via the public attribute :attr:`~.Device.observables` and public method :meth:`~.Device.supported`.

* ``_capabilities``: (optional) a dictionary containing information about the capabilities of the device. At the moment, only the key ``'model'`` is supported, which may return either ``'qubit'`` or ``'CV'``. Alternatively, you may use this class dictionary to return additional information to the user - this is accessible from the OpenQML frontend via the public method :meth:`~.Device.capabilities`.

For example, a very basic operator map that supports only two gates might look like so:

.. code-block:: python

    _operator_map = {'CNOT': cnot_function, 'PauliX': X_function}

where ``'CNOT'`` represents the built-in operation :class:`~.CNOT`, and ``'PauliX'`` represents the built-in operation :class:`~.ops.PauliX`. The functions in the dictionary can be of any form you like, and can exist in the plugin within the same file, separate files, or may even be imported from a different library. As long as the corresponding key representing the supported operator is there, OpenQML will allow that operation to be placed on the device.

For a better idea of how the ``_operator_map`` and ``_observable_map`` work, refer to the two reference plugins.

Applying operations
-------------------

Once all the class attributes are defined, it is necessary to define some required class methods, to allow OpenQML to apply operations to your device.

When OpenQML needs to evaluate a QNode, it accesses the :meth:`~.Device.execute` method, which performs the following process:

.. code-block:: python

    with self.execution_context():
        self.pre_apply()
        for operation in queue:
            self.apply(operation.name, operation.wires, operation.parameters)
        self.post_apply()

        self.pre_expectations()
        expectations = [self.expectation(observable.name, observable.wires, observable.parameters) for observable in observe]
        self.post_expectations()

        return np.array(expectations)

In most cases, there are a minimum of two methods that need to be defined:

* :meth:`~.Device.apply`: this accepts an operation name (as a string), the wires (subsystems) to apply the operation to, and the parameters for the operation, and applies the resulting operation to the device.

* :meth:`~.Device.expectation`: this accepts an observable name (as a string), the wires (subsystems) to apply the operation to, and the parameters for the expectation, returns the resulting expectation value from the device.

  .. note:: Currently, OpenQML only supports single-wire observables.

However, additional flexibility is sometimes required for interfacing with more complicated frameworks. In such cases, the following (optional) methods may also be defined:

* :meth:`~.Device.__init__`: by default, receives the ``short_name`` of the device, number of wires (``self.num_wires``), and number of shots ``self.shots``. You may overwrite this if you need to add additional options that the user must pass to the device on initialization - however, ensure that you call ``super().__init__(self.short_name, wires, shots)`` at some point here.

* :meth:`~.Device.execution_context`: this returns a context manager that may be required for applying operations and measuring expectation values from the device.

* :meth:`~.Device.pre_apply`: for any setup/code that must be executed before applying operations.

* :meth:`~.Device.post_apply`: for any setup/code that must be executed after applying operations.

* :meth:`~.Device.pre_expectations`: for any setup/code that must be executed before measuring observables.

* :meth:`~.Device.post_expectations`: for any setup/code that must be executed after measuring observables.

.. warning:: In advanced cases, the :meth:`~.Device.execute` method may be overwritten, to provide complete flexibility for handling device execution. However, this may have unintended side-effects and is not recommended - if possible, try implementing a suitable subset of the methods provided above.


Installation
------------

OpenQML uses a ``setuptools`` ``entry_points`` approach to plugin integration. In order to make your plugin accessible from OpenQML, simply provide the following keyword argument to the ``setup()`` function in your ``setup.py`` file:

.. code-block:: python

    devices_list = [
            'example.mydevice1 = MyModule.MySubModule:MyDevice1'
            'example.mydevice2 = MyModule.MySubModule:MyDevice2'
        ],
    setup(entry_points={'openqml.plugins': devices_list})

where the ``devices_list`` is a list of devices you would like to register, ``example.mydevice1`` is the short name of the device, and ``MyModule.MySubModule`` is the path to your Device class, ``MyDevice1``.

To ensure your device is working as expected, you can install it in developer mode using ``pip install -e .``. It will then be accessible via OpenQML.

Testing
-------

All plugins should come with extensive unit tests, to ensure that the device supports the correct gates and observables, and is applying them correctly. For an example of a plugin test suite, see ``tests/test_default_qubit.py`` and ``tests/test_default_gaussian.py``.

In general, as all supported operations have their gradient formula defined and tested by OpenQML, testing that your device calculates the correct gradients is not required - just that it *applies* and *measures* quantum operations and observables correctly.


Supporting new operations
----------------------

If you would like to support an operation or observable that is not currently supported by OpenQML, you can subclass the :class:`~.Operation` and :class:`~.Expectation` classes, and define the number of parameters the operation takes, and the number of wires the operation acts on. For example, to define the Ising gate :math:`XX_\phi` depending on parameter :math:`\phi`,

.. code-block:: python

    class Ising(Operation):
        """Ising gate"""
        num_params = 1
        n_wires = 2
        par_domain = 'R'
        grad_method = 'A'
        grad_recipe = None

where

* :attr:`~.Operation.num_params`: the number of parameters the operation takes.

* :attr:`~.Operation.num_wires`: the number of wires the operation acts on.

* :attr:`~.Operation.par_domain`: the domain of the gate parameters; ``'N'`` for natural numbers (including zero), ``'R'`` for floats, ``'A'`` for arrays of floats/complex numbers, and ``None`` if the gate does not have free parameters.

* :attr:`~.Operation.grad_method`: The gradient computation method; ``'A'`` for the analytic method, ``'F'`` for finite differences, and ``None`` if the operation may not be differentiated.

* :attr:`~.Operation.grad_recipe`: The gradient recipe for the analytic ``'A'`` method. This is a list with one tuple per operation parameter. For parameter :math:`k`, the tuple is of the form :math:`(c_k, s_k)`, resulting in a gradient recipe of

  .. math:: \frac{d}{d\phi_k}O = c_k\left[O(\phi_k+s_k)-O(\phi_k-s_k)\right].

  Note that if ``grad_recipe=None``, the default gradient recipe is :math:`(c_k, s_k)=(1/2, \pi/2)` for every parameter.

The user can then import this operation directly from your plugin, and use it when defining a QNode:

.. code-block:: python

    import openqml as qm
    from MyModule.MySubModule import SqrtX

    @qnode(dev1)
    def my_qfunc(x):
        qm.Hadamard(0)
        SqrtX(0)
        return qm.expval.PauliZ(0)

In this case, as the plugin is providing a custom operation not supported by OpenQML, it is recommended that the plugin unittests **do** provide tests to ensure that OpenQML returns the correct gradient for the custom operations.

.. note::

    If you are providing a custom/unsupported continuous-variable operation or expectation, you must subclass the :class:`~.CVOperation` or :class:`~.CVExpectation` classes instead.

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

    * This method should return the matrix of the linear transformation carried out by the gate for the given parameter values, and is used for calculating the gradient using the analytic method (``grad_method = 'A'``).

    * For observables, this method should return a real vector (first-order observables) or symmetric matrix (second-order observables) of coefficients of the quadrature operators :math:`\x` and :math:`\p`.

      - For single-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x, \p)`.
      - For multi-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

    Non-Gaussian CV operations and expectations are currently only supported via the finite difference method of gradient computation.
