.. role:: html(raw)
   :format: html

.. _varcirc:

Variational circuits
====================

A variational circuit is a parameterized quantum circuit :math:`U(x, \bm{\theta})` together with an observable :math:`\hat{B}` that is measured after applying the circuit to an initial state such as the ground or vacuum state :math:`| 0 \rangle`. 

The expectations :math:`\langle 0 | U(x, \bm{\theta})^{\dagger} \hat{B} U(x, \bm{\theta}) | 0 \rangle` of one or more such circuits - possibly with some classical post-processing - defines a scalar cost for a given task. The parametrized circuit is optimised with respect to the objective (see Figure \ref{var_principle}).


:html:`<br>`

.. figure:: ../_static/variational_rough.png
    :align: center
    :width: 40%
    :target: javascript:void(0);

    The principle of a *variational circuit*. 

:html:`<br>`

Typically, variational circuits are trained by a classical optimization algorithm that makes queries to the quantum device. The optimization is an iterative scheme that searches better candidates for the parameters :math:`\theta` with every step.

Variational circuits have been proposed for various near-term applications, such as optimization, quantum chemistry, machine learning, data compression and feature embeddings.

.. note:: Variational circuits have become popular as a way to think about quantum algorithms for near-term quantum devices. Such devices can only run short gate sequences, and have a high error. Usually, a quantum algorithm is decomposed into a set of standard elementary operations, which are in turn implemented by the quantum hardware. The intriguing idea of variational circuit for near-term devices is to merge this two-step procedure into a single step by "learning" the circuit on the noisy device for a given task. This way, the "natural" tunable gates of a device can be used to formulate the algorithm, without the detour via a fixed elementary gate set. Furthermore, systematic errors can automatically be corrected during optmization.


Building the circuit
--------------------

:html:`<br>`

.. figure:: ../_static/circuit_embedding.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    Example circuit showing how the argument :math:`x` and the function parameters :math:`\bm{\theta}` enter the quantum circuit. Circuits can also contain gates which have no free parameters (e.g., a CNOT).

:html:`<br>`

Both the input :math:`x` and the function parameters :math:`\bm{\theta}` **enter the quantum circuit** in the same way: as arguments for the circuit's gates. This allows us to convert *classical information* (the values of :math:`x` and :math:`\bm{\theta}`) into *quantum information* (the quantum state :math:`U(x;\bm{\theta})|0\rangle`).

Quantum information is turned **back into classical information** by evaluating the expectation value of the observable :math:`\hat{B}`,

.. math:: f(x; \bm{\theta}) = \langle \hat{B} \rangle = \langle 0 | U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) | 0 \rangle.


Beyond the basic rule that the inputs and parameters :math:`(x;\bm{\theta})` are used as the arguments of gates, exactly how the gates are arranged, the **circuit architecture**, is essentially arbitrary. 

.. note:: As shown in the figure above, the circuit can also include additional gates which have no free parameter associated with them. 


Example 
-------

Data-embedding 
~~~~~~~~~~~~~~

One straightforward embedding strategy is for the first few gates in the circuit to be responsible for embedding the input :math:`x` into a quantum state (which functions as a feature map [schuld2018quantum]_), while the subsequent gates have the parameters :math:`\bm{\theta}` as arguments. 

As an example, consider a photonic quantum computer (similar examples can be constructed for qubits). For simplicity, we temporarily omit the parameters :math:`\bm{\theta}`. We take the initial state to be the *vacuum* state and the measured observable :math:`\hat{B}` to be the position operator :math:`x`. The vacuum state has expectation value :math:`\langle\hat{x}\rangle = \langle 0 | \hat{x} | 0 \rangle = 0`. 

Suppose we have an input :math:`x`, which has :math:`N` dimensions. We can embed this into a quantum circuit with :math:`N` wires using the :class:`Displacement gate <pennylane.ops.cv.Displacement>`. For every component :math:`x_i` of :math:`x`, we apply :math:`D(x_i)` to wire :math:`i`. 

Measurement of the :class:`position expectation value <pennylane.expval.cv.X>` on each wire will then give the result 

.. math:: (\langle \hat{x_1} \rangle, \cdots, \langle \hat{x_N} \rangle ) = (x_1, \dots, x_N).

Thus, the displacement gate — combined with vacuum input and position measurements — can be used to directly encode data into a photonic quantum computer.

Data processing
~~~~~~~~~~~~~~~

To complete our picture of a quantum node, we would like to further process the embedded data from the example above. As it stands, our example circuit currently represents the *identity* :math:`f(x)=x`, which has no free parameters. By introducing additional gates, with parameters :math:`\bm{\theta}`, we can start building up more complex functions.

For clarity, we restrict to a one-dimensional input :math:`x` and add in a single :class:`Rotation gate <pennylane.ops.cv.Rotation>`, with free parameter :math:`\theta`. After applying this gate, the quantum node evaluated by our circuit becomes

.. math:: f(x;\theta) = x\cos(\theta).

Thus, with only two quantum gates (Displacement and Rotation), we can evaluate quantum functions with the above form. 

The above examples were kept very simple to illustrate the principles behind embedding data and parameters into quantum circuits. Indeed, the qfunc evaluated in the example is tractable classically. However, by increasing the number of subsystems and the circuit depth, the corresponding qfuncs can become progressively harder to evaluate classically, and a quantum device must be used.


Architectures
-------------

Beyond this simple example, many architectures have been proposed in the literature. The favourable properties of an architecture vary from task to task, and -- for example in machine learning, where there is a trade-off of flexibily and regularization -- it is not always clear what makes a good ansatz. Investigations of the expressive power of different approaches have begun [du2018expressive]_. One goal of Penny Lane is to facilitate such studies across  platforms.

To give a rough summary, let us distinguish three different types of architectures, namely **layered gate architectures**, **alternating operator architectures** and **tensor network architectures**.


Layered gate architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~

A layer is a sequence of gates that is repeated in the variational circuit. The number of repetitions of a layer forms a hyperparameter of the variational circuit. 


We can often decompose a layer further into two overall unitaries :math:`A` and :math:`B`. 

:html:`<br>`

.. figure:: ../_static/vc_general.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

Block :math:`A` contains single-qubit gates applied to every subsystem. Block :math:`B` consists of single subsystem, as well as entangling gates. 

:html:`<br>`

.. figure:: ../_static/vc_gatearchitecture.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

The different versions of layered gate architectures differ in three regards:

* Whether only :math:`A`, only :math:`B`, or both :math:`A` and :math:`B` are parametrized
* Which types of gates are used in :math:`A` and :math:`B`
* Whether the gates in Block :math:`B` are arranged randomly, fixed, or structured by a hyperparameter

We demonstrate this with discrete and continuous-variable models.

A parametrized, B fixed
***********************

In the simplest case of qubit-based devices we can use general SU(2) gates or rotations :math:`R` in Block :math:`A` and let :math:`B` be fixed,

:html:`<br>`

.. figure:: ../_static/vc_staticent.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

A parametrized, B parametrized
******************************

We can also have both :math:`A` and :math:`B` parametrized and the arrangements of the two-qubit gates depends on a hyperparameter defining the range of two-qubit gates. See also [romero2017quantum]_, [schuld2018circuit]_.

:html:`<br>`

.. figure:: ../_static/vc_cc.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`


A fully parametrized architecture specific to continuous-variable systems has been proposed in [schuld2018quantum]_. 

:html:`<br>`

.. figure:: ../_static/vc_cvkernels.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`

The entangling layer is represented by an interferometer, a passive optical element made up of individual beam splitters and phase shifters. Block :math:`A` consists of single-mode gates which consecutively increase the order of the quadrature operator in the generator: While the displacement :math:`D` is an order-1 operator, the quadratic phase gate :math:`Q` is order-2 and the cubic phase gate :math:`V` order-3 in the position operator. 

A fixed, B parametrized
***********************

An example for fixing the single qubit gates is a so-called *IQP* circuit, where :math:`A` consists of Hadamard gates and :math:`B` is made up of parametrized diagonal one- and two-qubit gates [havlicek2018supervised]_. 

:html:`<br>`

.. figure:: ../_static/vc_iqp.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

IQP circuits can also be constructed for continuous-variable systems.

:html:`<br>`

.. figure:: ../_static/vc_iqp_cv.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

All gates used, :math:`T`, (controlled) :math:`Z` rotations and a fixed :math:`CZ` gate, are diagonal in the computational basis.

Other structures
****************

Transcending the simple two-block structure allows to build more complex layers, such as this layer of a photonic neural network which emulates how information is processed in classical neural nets [killoran2018continuous]_ [steinbrecher2018quantum]_. 

:html:`<br>`

.. figure:: ../_static/vc_cvqnn.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

Here we use squeezers :math:`S`, cubic phase gates :math:`V` and continuous-variable (controlled) :math:`Z` gates.


Alternating operator architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The alternating operator architecture was first introduced in Farhi and Goldstone's *Quantum Approximate Optimization Algorithm* (QAOA) [farhi2014quantum]_ and later used for machine learning (for example [verdon2017quantum]_) and other domain-specific applications (i.e., [fingerhuth2018quantum]_). 

Again, we use layers of two blocks. The difference is that this time the unitaries representing these blocks are defined via Hamiltonians :math:`A` and :math:`B` which are evolved for a short time :math:`\Delta t`.

:html:`<br>`

.. figure:: ../_static/vc_aoa.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`


.. note:: The idea of this ansatz is based on adiabatic quantum computing, in which the sytem starts in a Hamiltonian :math:`A` and is slowly transformed to a target Hamiltonian :math:`B`. The system starts in the ground state of :math:`A` and adiabatically evolves to the ground state of  :math:`B`. Streptoscopic, or quickly alternating applications of  :math:`A` and  :math:`B` for very short times  :math:`\Delta t` can be used as a heuristic to approximate this evolution.

Tensor network architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amongst the architectures that do not consist of layers, but a single fixed structure, are gate sequences inspired by tensor networks [huggins2018towards]_ [du2018expressive]_. The simplest one is a tree architectures that consecutively entangle subsets of qubits:

:html:`<br>`

.. figure:: ../_static/vc_tree.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

Another option is a *matrix product state*. The unitaries can be decomposed in different ways, and their size corresponds to the "bond dimension" of the matrix product state - the higher the bond dimension, the more complex the circuit ansatz.

:html:`<br>`

.. figure:: ../_static/vc_mps.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`

.. note:: Tensor networks such as matrix product states were invented to simulate quantum systems efficiently on classical computers (even though the computation of expectation values can still be hard). Hence, tensor network architectures do not necessarily give rise to classically intractable quantum nodes. 

.. rubric:: Footnotes

.. [#] For example, see the following non-exhaustive list: [farhi2017quantum]_ [benedetti2018generative]_   [dallaire2018quantum]_  INTEGRATE!

