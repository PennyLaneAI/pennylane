.. role:: html(raw)
   :format: html

.. _varcirc:

Variational circuits
====================

A variational circuit is a parameterized quantum circuit :math:`U(\bm{\gamma})` together with an observable :math:`\hat{O}` that is measured after applying the circuit to an initial state such as the ground or vacuum state :math:`| 0 \rangle`. 

The expectations :math:`\langle 0 | U(\gamma)^{\dagger} \hat{O} U(\gamma) | 0 \rangle` of one or more such circuits - possibly with some classical post-processing - defines a scalar cost for a given task. The parametrized circuit is optimised with respect to the objective (see Figure \ref{var_principle}).


:html:`<br>`

.. figure:: ../_static/variational_rough.pdf
    :align: center
    :width: 40%
    :target: javascript:void(0);

    The principle of a *variational circuit*. 

:html:`<br>`


Typically, variational circuits are trained by a classical optimization algorithm that makes queries to the quantum device. The optimization is an iterative scheme that searches better candidates for the parameters :math:`\gamma` with every step.

Applications
-------------


Variational circuits have become popular as a way to think about quantum algorithms for near-term quantum devices. Such devices can only run short gate sequences, and have a high error. Usually, a quantum algorithm is decomposed into a set of standard elementary operations, which are in turn implemented by the quantum hardware. The intriguing idea of variational circuit for near-term devices is to merge this two-step procedure into a single step by "learning" the circuit on the noisy device for a given task. This way, the "natural" tunable gates of a device can be used to formulate the algorithm, without the detour via a fixed elementary gate set. Furthermore, systematic errors can automatically be corrected during optmization.

Variational circuits have been proposed for various near-term applications, such as

* optimization [Ref Farhi], 
* quantum chemistry [...], 
* variational autoencoders [Ref Jonny], 
* variational classifiers [REF Schuld 2018], and
* feature embeddings [REF]


Architectures
-------------
The core of a variational circuit is the \textit{architecture} or the fixed gate sequence that is the skeleton of the algorithm. The favourable properties of an architecture certainly vary from task to task, and -- for example in machine learning, where there is a trade-off of flexibily and regularization -- it is not always clear what makes a good ansatz. Investigations of the expressive power of different approaches have begun [REF new paper]. One goal of Penny Lane is to facilitate such studies across  platforms.

To give a rough summary of variational architectures that have been proposed in the literature, let us distinguish three different types of architectures, *gate*, *alternating operator* and *tensor net-based architectures*.


Gate-based architectures
************************

Gate-based architectures define a layer architecture. The number of repetitions of a layer forms a hyperparameter of the variational circuit. 


For qubit gates, we can often decompose a layer further into two blocks :math:`A` and :math:`B`. 

:html:`<br>`

.. figure:: ../_static/vc_general.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`

Block :math:`A` contains single-qubit gates applied to every qubit. Block :math:`B` also consists of entangling two-qubit gates. 

:html:`<br>`

.. figure:: ../_static/vc_gatearchitecture.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`

The architectures differ in two regards:

* Whether only :math:`A`, only :math:`B`, or both :math:`A` and :math:`B` are parametrized
* Whether the gates in Block :math:`B` are arranged randomly, fixed, or structured by a hyperparameter

In the simplest case we can use SU(2) gates in Block :math:`A` and let :math:`B` be fixed,

:html:`<br>`

.. figure:: ../_static/vc_staticent.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`


In [Jonny Autoencoder arxiv1612.02806, SCHULD CC] we have both :math:`A` and :math:`B` parametrized and the arrangements of the two-qubit gates depends on a hyperparameter defining the range of two-qubit gates.

:html:`<br>`

.. figure:: ../_static/vc_cc.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`

[HAVELIC] use an IQP circuit where :math:`A` consists of Hadamards and :math:`B` is made up of parametrized diagonal one- and two-qubit gates. 

:html:`<br>`

.. figure:: ../_static/vc_iqp.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`

IQP circuits can also be constructed for continuous-variable systems.

:html:`<br>`

.. figure:: ../_static/vc_iqp_cv.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`


An architecture specific to continuous-variable systems has been proposed in [Schuld Killoran]. The entangling layer is represented by an interferometer, a passive optical element made up of individual beam splitters and phase shifters. Block :math:`A` consists of single-mode gates which consecutively increase the order of the quadrature operator in the generator: While the displacement is an order-1 operator, the quadratic phase gate is order-2 and the cubic phase gate order-3. [Explain BETTER] 

:html:`<br>`

.. figure:: ../_static/vc_cvkernels.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`


Transcending the simple two-block structure allows to build more complex layers, such as this layer of a photonic neural network which emulates how information is processed in classical neural nets [REF]. 

:html:`<br>`

.. figure:: ../_static/vc_cvqnn.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`



Alternating operator architectures
**********************************

The alternating operator structure was first introduced in Farhi and Goldstone's Quantum Approximate Optimization Algorithm (QAOA) [REF] and later used for machine learning [GUILLAUME PAPER] and optimization [MARK PAPER, others?]. The idea is based on adiabatic quantum computing, in which the sytem starts in a Hamiltonian :math:`A` and is slowly transformed to a target Hamiltonian :math:`B`. The system starts in the ground state of :math:`A` and adiabatically evolves to the ground state of  :math:`B`. Streptoscopic, or quickly alternating applications of  :math:`A` and  :math:`B` for very short times  :math:`\Delta t` can be used as a heuristic to approximate this evolution.

:html:`<br>`

.. figure:: ../_static/vc_aoa.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`


Tensor network architectures
****************************

Other architectures for variational circuits are inspired by tensor networks. For example, a three tensor network translates into a circuit that consecutively entangles subsets of qubits.

:html:`<br>`

.. figure:: ../_static/vc_tree.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);

    TEXT. 

:html:`<br>`



