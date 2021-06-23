Device Tracker
==============

.. currentmodule:: pennylane.device_tracker

.. code:: ipython3

    import pennylane as qml
    from pennylane import numpy as np

.. code:: ipython3

    dev = qml.device('default.qubit', wires=1, shots=100)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    x = np.array(0.1)

Default usage
-------------

.. code:: ipython3

    with qml.track(dev) as tracker:
        g = qml.grad(circuit)(x)
        circuit(x, shots=10)

.. code:: ipython3

    tracker.totals




.. parsed-literal::

    defaultdict(int, {'executions': 4, 'shots': 310})



.. code:: ipython3

    tracker.history




.. parsed-literal::

    defaultdict(list, {'executions': [1, 1, 1, 1], 'shots': [100, 100, 100, 10]})



.. code:: ipython3

    tracker.current




.. parsed-literal::

    {'executions': 1, 'shots': 10}



Reset on enter
--------------

If you want to reuse the same device tracker across multiple runtime
contexts, you can specify ``reset_on_enter=False``. That way, stored
data is not zeroed out upon entering a new context.

.. code:: ipython3

    with qml.track(dev, record="totals", reset_on_enter=False) as tracker:
        qml.grad(circuit)(x)
    
    print("Now to reuse the tracker: ")
    with tracker:
        circuit(x)


.. parsed-literal::

    Total: executions = 1	shots = 100	
    Total: executions = 2	shots = 200	
    Total: executions = 3	shots = 300	
    Now to reuse the tracker: 
    Total: executions = 4	shots = 400	


Record Options
--------------

.. code:: ipython3

    with qml.track(dev, record="totals") as tracker:
        g = qml.grad(circuit)(x)
        circuit(x, shots=10)


.. parsed-literal::

    Total: executions = 1	shots = 100	
    Total: executions = 2	shots = 200	
    Total: executions = 3	shots = 300	
    Total: executions = 4	shots = 310	


.. code:: ipython3

    with qml.track(dev, record="current") as tracker:
        g = qml.grad(circuit)(x)
        circuit(x, shots=10)


.. parsed-literal::

    Current: executions = 1	shots = 100	
    Current: executions = 1	shots = 100	
    Current: executions = 1	shots = 100	
    Current: executions = 1	shots = 10	


Custom Record
~~~~~~~~~~~~~

We can also pass in our own custom record function. This function *must*
take ``totals``, ``history``, and ``current`` as keyword arguments.

To demonstrate, we will log the information totals instead of using
``print``. To do so, we first need to import and setup logging:

.. code:: ipython3

    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

Then we create our custom recorder function.

.. code:: ipython3

    def custom_recorder(totals = dict(), history= dict(), current=dict()):
        logger.info(f"Executions: {totals['executions']} Shots: {totals['shots']}")

And finally we pass this function to the ``record`` keyword:

.. code:: ipython3

    with qml.track(dev, record=custom_recorder) as track:
        circuit(x, shots=100)
        circuit(x, shots=10)


.. parsed-literal::

    INFO:root:Executions: 1 Shots: 100
    INFO:root:Executions: 2 Shots: 110


Update
------

The device specifies which quantities for the tracker to store by
default, but we can add more information in as well. If you specify
``update="timings"``, the difference in ``time.time()`` between
subsequent ``update`` calls will also get stored.

.. code:: ipython3

    with qml.track(dev, record="totals", update="timings") as tracker:
        g = qml.grad(circuit)(x)


.. parsed-literal::

    Total: executions = 1	shots = 100	time = 0.0013034343719482422	
    Total: executions = 2	shots = 200	time = 0.0031752586364746094	
    Total: executions = 3	shots = 300	time = 0.0040128231048583984	


.. code:: ipython3

    tracker.history['time']




.. parsed-literal::

    [0.0013034343719482422, 0.0018718242645263672, 0.0008375644683837891]



More customization
------------------

While custom record functions are the easiest, you can customize any
part of the tracker by creating your own subclass.

The necessary methods for a device tracker are: \* ``__init__`` \*
``__enter__`` \* ``__exit__`` \* ``update`` \* ``reset`` \* ``record``

Here, let’s create a custom tracker that also logs the current
system-wise CPU utilization percentage. We do this using the ``psutil``
package.

.. code:: ipython3

    import psutil

All we need to do is create our own class, and slightly modify the
``update`` function.

.. code:: ipython3

    class MyTracker(qml.device_tracker.PrintCurrent):
        
        def update(self, **current):
            current["cpu_percent"] = psutil.cpu_percent()
            super().update(**current)

.. code:: ipython3

    with MyTracker(dev) as tracker:
        g = qml.grad(circuit)(x)


.. parsed-literal::

    Current: executions = 1	shots = 100	cpu_percent = 16.2	
    Current: executions = 1	shots = 100	cpu_percent = 0.0	
    Current: executions = 1	shots = 100	cpu_percent = 0.0	


Device Necessities
------------------

For the tracker to work, the device should include the code:

::

   if self.tracker.tracking:
       self.tracker.update(executions=1, shots=self._shots)
       self.tracker.record()

near the end of its ``execute`` method. Since this relies on a tracker
attribute, the device should have ``self.tracker`` initialized to
``DefaultTracker`` upon initialization inside ``__init__``:

::

   self.tracker = qml.device_tracker.DefaultTracker()

Devices can store any additional information with the update method. For
example, a remote device could store price:

::

   self.tracker.update(price=0.10)

The device can also call the ``update`` and ``record`` methods where
ever it wants. For example, a device could have a separate call to these
functions in ``batch_execute``:

::

   if self.tracker.tracking:
       self.tracker.update(batch_execute=1, batch_length=len(circuits))
       self.tracker.record()

