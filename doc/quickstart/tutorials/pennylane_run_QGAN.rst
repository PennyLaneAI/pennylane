.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_pennylane_run_QGAN.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_pennylane_run_QGAN.py:


.. _quantum_GAN:

Quantum Generative Adversarial Network
======================================

This demo constructs a Quantum Generative Adversarial Network (QGAN)
(`Lloyd and Weedbrook
(2018) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.040502>`__,
`Dallaire-Demers and Killoran
(2018) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.012324>`__)
using two subcircuits, a *generator* and a *discriminator*. The
generator attempts to generate synthetic quantum data to match a pattern
of “real” data, while the discriminator, tries to discern real data from
fake data. The gradient of the discriminator’s output provides a
training signal for the generator to improve its fake generated data.

Imports
~~~~~~~


.. code-block:: default


    # As usual, we import PennyLane, the PennyLane-provided version of NumPy,
    # and an optimizer.

    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.optimize import GradientDescentOptimizer







We also declare a 3-qubit device.


.. code-block:: default



    dev = qml.device("default.qubit", wires=3)







Classical and quantum nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In classical GANs, the starting point is to draw samples either from
some “real data” distribution, or from the generator, and feed them to
the discriminator. In this QGAN example, we will use a quantum circuit
to generate the real data.

For this simple example, our real data will be a qubit that has been
rotated (from the starting state :math:`\left|0\right\rangle`) to some
arbitrary, but fixed, state.


.. code-block:: default



    def real(phi, theta, omega):
        qml.Rot(phi, theta, omega, wires=0)








For the generator and discriminator, we will choose the same basic
circuit structure, but acting on different wires.

Both the real data circuit and the generator will output on wire 0,
which will be connected as an input to the discriminator. Wire 1 is
provided as a workspace for the generator, while the discriminator’s
output will be on wire 2.


.. code-block:: default



    def generator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=1)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=1)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(w[6], wires=0)
        qml.RY(w[7], wires=0)
        qml.RZ(w[8], wires=0)


    def discriminator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=2)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=2)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=2)
        qml.CNOT(wires=[1, 2])
        qml.RX(w[6], wires=2)
        qml.RY(w[7], wires=2)
        qml.RZ(w[8], wires=2)








We create two QNodes. One where the real data source is wired up to the
discriminator, and one where the generator is connected to the
discriminator.


.. code-block:: default



    @qml.qnode(dev)
    def real_disc_circuit(phi, theta, omega, disc_weights):
        real(phi, theta, omega)
        discriminator(disc_weights)
        return qml.expval(qml.PauliZ(2))


    @qml.qnode(dev)
    def gen_disc_circuit(gen_weights, disc_weights):
        generator(gen_weights)
        discriminator(disc_weights)
        return qml.expval(qml.PauliZ(2))








Cost
~~~~

There are two ingredients to the cost here. The first is the probability
that the discriminator correctly classifies real data as real. The
second ingredient is the probability that the discriminator classifies
fake data (i.e., a state prepared by the generator) as real.

The discriminator’s objective is to maximize the probability of
correctly classifying real data, while minimizing the probability of
mistakenly classifying fake data.

The generator’s objective is to maximize the probability that the
discriminator accepts fake data as real.


.. code-block:: default



    def prob_real_true(disc_weights):
        true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
        # convert to probability
        prob_real_true = (true_disc_output + 1) / 2
        return prob_real_true


    def prob_fake_true(gen_weights, disc_weights):
        fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
        # convert to probability
        prob_fake_true = (fake_disc_output + 1) / 2
        return prob_fake_true  # generator wants to minimize this prob


    def disc_cost(disc_weights):
        cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
        return cost


    def gen_cost(gen_weights):
        return -prob_fake_true(gen_weights, disc_weights)








Optimization
~~~~~~~~~~~~

We initialize the fixed angles of the “real data” circuit, as well as
the initial parameters for both generator and discriminator. These are
chosen so that the generator initially prepares a state on wire 0 that
is very close to the :math:`\left| 1 \right\rangle` state.


.. code-block:: default


    phi = np.pi / 6
    theta = np.pi / 2
    omega = np.pi / 7
    np.random.seed(0)
    eps = 1e-2
    gen_weights = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=[9])
    disc_weights = np.random.normal(size=[9])







We begin by creating the optimizer:


.. code-block:: default


    opt = GradientDescentOptimizer(0.1)







In the first stage of training, we optimize the discriminator while
keeping the generator parameters fixed.


.. code-block:: default


    for it in range(50):
        disc_weights = opt.step(disc_cost, disc_weights)
        cost = disc_cost(disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it + 1, cost))




.. code-block:: pytb

    Traceback (most recent call last):
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 394, in _memory_usage
        out = func()
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/sphinx_gallery/gen_rst.py", line 382, in __call__
        exec(self.code, self.globals)
      File "/home/maria/Desktop/XANADU/pennylane/examples/pennylane_run_QGAN.py", line 177, in <module>
        disc_weights = opt.step(disc_cost, disc_weights)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/optimize/gradient_descent.py", line 63, in step
        g = self.compute_grad(objective_fn, x, grad_fn=grad_fn)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/optimize/gradient_descent.py", line 87, in compute_grad
        g = autograd.grad(objective_fn)(x)  # pylint: disable=no-value-for-parameter
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/wrap_util.py", line 20, in nary_f
        return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/differential_operators.py", line 24, in grad
        vjp, ans = _make_vjp(fun, x)
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/core.py", line 10, in make_vjp
        end_value, end_node =  trace(start_node, fun, x)
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/tracer.py", line 10, in trace
        end_box = fun(start_box)
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/wrap_util.py", line 15, in unary_f
        return fun(*subargs, **kwargs)
      File "/home/maria/Desktop/XANADU/pennylane/examples/pennylane_run_QGAN.py", line 142, in disc_cost
        cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
      File "/home/maria/Desktop/XANADU/pennylane/examples/pennylane_run_QGAN.py", line 135, in prob_fake_true
        fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/decorator.py", line 60, in wrapper
        return qnode(*args, **kwargs)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/qnode.py", line 678, in __call__
        return self.evaluate(args, **kwargs)  # args as one tuple
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/tracer.py", line 44, in f_wrapped
        ans = f_wrapped(*argvals, **kwargs)
      File "/home/maria/Desktop/XANADU/venv_xanadu/lib/python3.6/site-packages/autograd/tracer.py", line 48, in f_wrapped
        return f_raw(*args, **kwargs)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/qnode.py", line 710, in evaluate
        self.construct(args, kwargs)
      File "/home/maria/Desktop/XANADU/pennylane/pennylane/qnode.py", line 373, in construct
        res = self.func(*variables, **keyword_values)
      File "/home/maria/Desktop/XANADU/pennylane/examples/pennylane_run_QGAN.py", line 107, in gen_disc_circuit
        return qml.expval(qml.PauliZ(2))
    TypeError: 'module' object is not callable




At the discriminator’s optimum, the probability for the discriminator to
correctly classify the real data should be close to one.


.. code-block:: default


    print(prob_real_true(disc_weights))



For comparison, we check how the discriminator classifies the
generator’s (still unoptimized) fake data:


.. code-block:: default


    print(prob_fake_true(gen_weights, disc_weights))



In the adverserial game we have to now train the generator to better
fool the discriminator (we can continue training the models in an
alternating fashion until we reach the optimum point of the two-player
adversarial game).


.. code-block:: default


    for it in range(200):
        gen_weights = opt.step(gen_cost, gen_weights)
        cost = -gen_cost(gen_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it, cost))



At the optimum of the generator, the probability for the discriminator
to be fooled should be close to 1.


.. code-block:: default


    print(prob_fake_true(gen_weights, disc_weights))



At the joint optimum the overall cost will be close to zero.


.. code-block:: default


    print(disc_cost(disc_weights))



The generator has successfully learned how to simulate the real data
enough to fool the discriminator.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.078 seconds)


.. _sphx_glr_download_tutorials_pennylane_run_QGAN.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: pennylane_run_QGAN.py <pennylane_run_QGAN.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: pennylane_run_QGAN.ipynb <pennylane_run_QGAN.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
