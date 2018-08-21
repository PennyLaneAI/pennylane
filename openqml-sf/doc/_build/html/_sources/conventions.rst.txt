Conventions
===========


Measurements in QM
------------------

Let us have a quick recap of how measurements work in quantum mechanics.
A general (full) measurement is defined by a set of measurement operators :math:`\{A_i\}`,
with the property

.. math:: \sum_k A_k^\dagger A_k = \I.

If the state before the measurement is :math:`\rho`, the probability of obtaining result :math:`k` is

.. math:: p_{A_k} = \tr(A_k^\dagger A_k \rho).

The aforementioned property of the measurement operators guarantees that the probabilities sum up to one for
any state :math:`\rho`.
If the measurement yields the result :math:`k`, the (conditional) state after the measurement is

.. math:: \rho_{A_k}' = \frac{A_k \rho A_k^\dagger}{p_{A_k}}.

If we ignore the outcome but wish to obtain the state after the measurement process,
we may take the convex combination of these conditional states multiplied by their respective probabilities:

.. math:: \rho_{A}' = \sum_k p_{A_k} \rho_{A_k}' = \sum_k A_k \rho A_k^\dagger.

If each measurement outcome is associated with a real number :math:`a_k`, we may obtain the expectation value
of this random variable as

.. math:: \expect{A} = \sum_k a_k p_{A_k} = \tr(\sum_k a_k A_k^\dagger A_k \rho) = \tr(\tilde{A} \rho),

where :math:`\tilde{A} = \sum_k a_k A_k^\dagger A_k` is a hermitian operator.

A special case called a *projective measurement* is obtained when the measurement operators
:math:`\{A_i\}` are a set of mutually orthogonal projectors:
:math:`A_i^\dagger = A_i` and :math:`A_i A_j = \delta_{ij} A_i`.
In this case many of the above equalities are simplified, for example
:math:`\sum_k A_k^\dagger A_k = \sum_k A_k = \I` and :math:`p_{A_k} = \tr(A_k \rho)`.
Now the operator :math:`\tilde{A} = \sum_k a_k A_k` is called a *hermitian observable* corresponding to the measurement,
and it is given directly as a spectral decomposition in terms of the projectors.
Moreover, in this case we conveniently have for any analytic function :math:`f`

.. math:: \expect{f(A)} = \sum_k f(a_k) p_{A_k} = \tr(\sum_k f(a_k) A_k \rho) = \tr(f(\tilde{A}) \rho).


.. note:: One can consider unitary gates (which are deterministic operations) as "measurements" with just a single operator, the unitary itself.


Multiple measurements
---------------------

Two consecutive measurements :math:`\{A_i\}` and :math:`\{B_i\}` can be combined into a single measurement
with the measurement operators :math:`C_{ji} = B_j A_i`:

.. math:: p_{(B_j|A_i)} = \tr(B_j^\dagger B_j \rho_{A_i}') = \tr(B_j^\dagger B_j A_i \rho A_i^\dagger) / p_{A_i} = \tr(C_{ji} \rho C_{ji}^\dagger) / p_{A_i},

and thus the joint p.d.f. is obtained as

.. math:: p_{(B_j, A_i)} = p_{(B_j|A_i)} p_{A_i} = \tr(C_{ji}^\dagger C_{ji} \rho).

Note that the relative order of the measurements in general matters, different orders produce different probability distributions.

If there are any unitaries between the measurements we may fold them into the latter measurement:
:math:`B_j U A_i = U (U^\dagger B_j U) A_i = U B_j' A_i`.

Things are greatly simplified if the measurement operators commute pairwise, :math:`[A_i, B_j] = 0`.
In this case the joint p.d.f. does not depend on the order of the measurements. Furthermore,
the expectation values can be obtained as

.. math::
   \expect{A} = \sum_{ij} a_i p_{(B_j, A_i)} = \sum_{ij} a_i \tr(B_j A_i \rho A_i^\dagger B_j^\dagger) = \tr(\sum_i a_i A_i^\dagger A_i \rho) = \tr(\tilde{A} \rho),

   \expect{B} = \sum_{ij} b_j p_{(B_j, A_i)} = \sum_{ij} b_j \tr(B_j A_i \rho A_i^\dagger B_j^\dagger) = \sum_i \tr(\tilde{B} A_i \rho A_i^\dagger) = \tr(\tilde{B} \rho).

If both A and B are projective measurements and they additionally commute, the combined measurement C is also projective.


.. _measurements:

Observables and measurements in OpenQML and Strawberry Fields
-------------------------------------------------------------

In Strawberry Fields we may perform state preparations, gates and measurements on the quantum register.
SF is currently designed to model a single execution of a quantum circuit in the lab.
The probability distribution of the results of a measurement is determined by the (reduced) system state at the time of the measurement.
Each measurement probabilistically collapses the system state, and thus the results of any following measurements may depend on the results of the preceding ones.
After the program is finished, in addition to the measured results we may treat the final state as an output of the circuit.

.. note:: The DAG (Directed Acyclic Graph) used to describe the quantum circuit in SF corresponds (via transitive closure) to a strict partial order of the quantum operations.
	  We denote :math:`A < B` if :math:`A` has to be executed before :math:`B`.


Assuming the program itself is deterministic, if there are no measurements we obtain the same final state each time (which in itself is a quantum probability distribution).
If there are measurements :math:`A, B, \ldots` in the program, each run of the program instead samples the joint p.d.f. of the measurement results,
which are not necessarily statistically independent.
Due to entanglement, even if two measurements seem independent on the circuit DAG (in the sense that neither :math:`A<B` nor :math:`B<A`),
their results of may depend on each other.

.. note::

   Example (ignoring normalization here): Consider the state :math:`\ket{00}+\ket{11}`. If we measure both of the two subsystems separately in the computational basis, for both measurements
   0 and 1 are equally likely outcomes, but the results always are perfectly correlated.

However, if neither :math:`A<B` nor :math:`B<A`, the measurements necessarily commute, and the relative order in which the measurements are performed does not affect the joint p.d.f. of the results.
Furthermore, if we are only interested in the expectation values of the observables (and don't care about correlations), we may compute them as shown in the previous section.

Let us now assume we wish to obtain (estimates for) the expectation values :math:`\expect{A}, \expect{B}, \ldots` of a set of measurements using SF.

#. The simplest method is to run the circuit :math:`n` times and average the results for each measurement, resulting in an estimate for the expectation values.
   This is computationally inefficient for simulators, but may be a valid method for a hardware backend.

#. If :math:`A` does not causally depend on any other measurements (this is always true for the first measurement we make in the program),
   we may run the program until we reach :math:`A` and sample the state at that point :math:`n` times.
   This will not work if :math:`A` depends on preceding measurements.
   **This feature is not yet implemented in SF.**
   Alternatively, for certain types of measurements we may extract the state object before :math:`A` happens,
   and then call the appropriate expectation value method on it to obtain the EV directly.

#. The method in the previous item can be used for any number of measurements if they are all consecutive
   (or made consecutive by eliminating unitaries between them as explained in the previous section, thereby modifying the measurement operators),
   and all commute with each other.

#. The ideal solution would be to automate the the computation of expectation values of an arbitrary number of measurements
   as efficiently as possible. At least it would be possible to topologically sort the circuit DAG into a Command sequence consisting of three parts, A+B+C,
   where only B contains measurements and A and C are as long as possible. By isolating the measurements into a short program sequence
   we may benefit by saving and re-using the state after A during the sampling in part B. If we are not interested in the final state, executing C is not necessary at all.

When estimating expectation values like this, we will not obtain a specific collapsed state (and usually will not need one).
If one is needed, the option that makes the most sense for commuting measurements would probably be

.. math:: \rho' = \sum_{ij\ldots} (A_i B_j \cdots) \rho (A_i B_j \cdots)^\dagger.


In summary, both :mod:`openqml.plugins.strawberryfields` and :mod:`openqml.plugins.dummy_plugin` currently are only designed to return expectation values of measurements which are

#. projective,
#. grouped next to each other, and
#. all commute with each other.
