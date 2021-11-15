qml.hf
======

This module provides the functionality to perform differentiable Hartree-Fock (HF) calculations and
construct molecular Hamiltonians that can be differentiated with respect to nuclear coordinates and
basis set parameters.

The differentiable HF solver allows computing exact gradients with respect to
molecular geometry, basis set, and circuit parameters simultaneously using the techniques of
automatic differentiation available in `Autograd <https://github.com/HIPS/autograd>`__. This makes
the solver more versatile and robust compared to non-differentiable tools that mainly rely
on numerical approaches for computing gradients, which can lead to inaccuracies and instability.
Additionally, optimizing the basis set parameters allows reaching lower ground-state energies
without increasing the size of the basis set. Overall, the solver allows users to execute end-to-end
differentiable algorithms for quantum chemistry. 

.. currentmodule:: pennylane.hf

.. automodapi:: pennylane.hf
    :no-inheritance-diagram:

Using the differentiable HF solver
----------------------------------

The HF solver computes the integrals over basis functions, constructs the relevant matrices, and
performs self-consistent-field iterations to obtain a set of optimized molecular orbital
coefficients. These coefficients and the computed integrals over basis functions are used to
construct the one- and two-body electron integrals in the molecular orbital basis, which can be
used to generate differentiable second-quantized Hamiltonians in the fermionic and qubit basis.

The following code shows the construction of the Hamiltonian for the hydrogen molecule where the
geometry of the molecule and the basis set parameters are all differentiable.

.. code-block:: python3

    import pennylane as qml
    from pennylane import numpy as np

    symbols = ["H", "H"]
    # This initial geometry is suboptimal and will be optimized by the algorithm
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=True)

    # The exponents and contraction coefficients of the Gaussian basis functions
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                      [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                      [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

We create a molecule object with differentiable atomic coordinates and basis set parameters and then
construct the Hamiltonian.

.. code-block:: python3

    mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args_mol = [geometry, alpha, coeff] # initial values of the differentiable parameters

    hamiltonian = qml.hf.generate_hamiltonian(mol)(*args_mol)

>>> print(hamiltonian)
  ((-0.3596823592263728+0j)) [I0]
+ ((-0.11496335836149135+0j)) [Z3]
+ ((-0.1149633583614913+0j)) [Z2]
+ ((0.1308241430373499+0j)) [Z0]
+ ((0.1308241430373499+0j)) [Z1]
+ ((0.10316898251626505+0j)) [Z0 Z2]
+ ((0.10316898251626505+0j)) [Z1 Z3]
+ ((0.1532995999427217+0j)) [Z0 Z3]
+ ((0.1532995999427217+0j)) [Z2 Z1]
+ ((0.1540549585580985+0j)) [Z0 Z1]
+ ((0.1609686663985837+0j)) [Z3 Z2]
+ ((-0.05013061742645664+0j)) [Y0 X2 X3 Y1]
+ ((-0.05013061742645664+0j)) [X0 Y2 Y3 X1]
+ ((0.05013061742645664+0j)) [Y0 X2 Y3 X1]
+ ((0.05013061742645664+0j)) [X0 Y2 X3 Y1]

The generated Hamiltonian can be used in a circuit where the molecular geometry, the basis set
parameters, and the circuit parameters are optimized simultaneously. Further information about
molecular geometry optimization with PennyLane is provided in this
`paper <https://arxiv.org/abs/2106.13840>`__ and this
`demo <https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html>`__ .

.. code-block:: python3

    dev = qml.device("default.qubit", wires=4)
    hf_state = np.array([1, 1, 0, 0])
    params = [np.array([0.0], requires_grad=True)] # initial values of the circuit parameters

    def generate_circuit(mol):
        @qml.qnode(dev)
        def circuit(*args):
            qml.BasisState(hf_state, wires=[0, 1, 2, 3])
            qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
            return qml.expval(qml.hf.generate_hamiltonian(mol)(*args[1:]))
        return circuit

Now that the circuit is defined, we can create a geometry and parameter optimization loop:

.. code-block:: python3

    for n in range(21):

        mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
        args = [params, geometry, alpha, coeff] # initial values of the differentiable parameters

        # compute gradients with respect to the circuit parameters and update the parameters
        g_params = qml.grad(generate_circuit(mol), argnum = 0)(*args)
        params = params - 0.25 * g_params[0]

        # compute gradients with respect to the nuclear coordinates and update geometry
        g_coor = qml.grad(generate_circuit(mol), argnum = 1)(*args)
        geometry = geometry - 0.5 * g_coor

        # compute gradients with respect to the Gaussian exponents and update the exponents
        g_alpha = qml.grad(generate_circuit(mol), argnum = 2)(*args)
        alpha = alpha - 0.25 * g_alpha

        # compute gradients with respect to the Gaussian contraction coefficients and update them
        g_coeff = qml.grad(generate_circuit(mol), argnum = 3)(*args)
        coeff = coeff - 0.25 * g_coeff

        if n%5 == 0:
            print(f'Step: {n}, Energy: {generate_circuit(mol)(*args)}, Maximum Absolute Force: {abs(g_coor).max()}')


Running this optimization, we get the following output in atomic units:

.. code-block:: text

    Step: 0, Energy: -1.0491709019853235, Maximum Force: 0.15801947189250276
    Step: 5, Energy: -1.134986263549686, Maximum Force: 0.037660772858684355
    Step: 10, Energy: -1.1399960673348044, Maximum Force: 0.005175326423590643
    Step: 15, Energy: -1.1403213849556897, Maximum Force: 0.0004138322831406249
    Step: 20, Energy: -1.1403680839521801, Maximum Force: 8.223301624310508e-06

Note that the computed energy is lower than the Full-CI energy, -1.1373060483 Ha, computed without
optimizing the basis set parameters.

The components of the HF solver can also be differentiated individually. For instance, the overlap
integral can be differentiated with respect to the basis set parameters as follows

.. code-block:: python3

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], requires_grad=False)
    alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                    [3.42525091, 0.62391373, 0.1688554]], requires_grad = True)
    coeff = np.array([[0.15432897, 0.53532814, 0.44463454],
                    [0.15432897, 0.53532814, 0.44463454]], requires_grad = True)

    mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)
    args = [alpha, coeff]

    a = mol.basis_set[0]
    b = mol.basis_set[1]

    g_alpha = qml.grad(qml.hf.generate_overlap(a, b), argnum = 0)(*args)
    g_coeff = qml.grad(qml.hf.generate_overlap(a, b), argnum = 1)(*args)

>>> print(g_alpha)
[[ 0.00169332 -0.14826928 -0.37296693]
 [ 0.00169332 -0.14826928 -0.37296693]]
