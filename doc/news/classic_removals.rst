.. _classic_removals:

Breaking changes and hard removals for PennyLane 2
==================================================

PennyLane is the middle of a major overhaul to refocus around the infrastructure trully
needed for the fault tolerant compilation and execution of useful, scalable applications.
In order to do so, we are making hard breaking changes and removals for things that
previously would have gotten a deprecation cycle or been left in a low maintainence state.

While these changes are also discussed in the changelog, they are repeated here.

* The :class:`pennylane.resource.Resources`, :class:`~.ResourceOperator`, and :class:`~.ErrorOperator` classes as well as the entire :mod:`pennylane.resource.error` module have been removed.
  [(#9786)](https://github.com/PennyLaneAI/pennylane/pull/9786)

* All Continuous Variable (CV) operators, base classes, devices, and ``qp.gradients.param_shift_cv``
  have been removed.
  [(#9869)](https://github.com/PennyLaneAI/pennylane/pull/9869)

* All Qutrit code, including operators, ``default.qutrit``, and ``default.qutrit.mixed`` have been removed.
  [(#9867)](https://github.com/PennyLaneAI/pennylane/pull/9867)