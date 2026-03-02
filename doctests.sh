#! /bin/bash

IGNORE_OPTS_CLOSED="
--ignore=pennylane/labs
"
IGNORE_OPTS_CORE="
--ignore=pennylane/workflow/interfaces/tensorflow.py
--ignore=pennylane/control_flow
--ignore=pennylane/math
--ignore=pennylane/compiler
--ignore=pennylane/measurements
--ignore=pennylane/ops/qubit
--ignore=pennylane/ops/qutrit
--ignore=pennylane/capture
--ignore=pennylane/devices
"
IGNORE_OPTS_AUXILIARY="
--ignore=pennylane/shadows
--ignore=pennylane/gradients
--ignore=pennylane/optimize
--ignore=pennylane/pulse
"
IGNORE_OPTS_TERTIARY="
--ignore=pennylane/io
--ignore=pennylane/logging
--ignore=pennylane/qnn
--ignore=pennylane/qaoa
--ignore=pennylane/data
--ignore=pennylane/debugging
--ignore=pennylane/qcut
--ignore=pennylane/qchem

--ignore=pennylane/_grad
--ignore=pennylane/allocation.py
--ignore=pennylane/boolean_fn.py
--ignore=pennylane/bose
--ignore=pennylane/decomposition
--ignore=pennylane/drawer
--ignore=pennylane/estimator
--ignore=pennylane/fermi
--ignore=pennylane/fourier
--ignore=pennylane/ftqc
--ignore=pennylane/kernels
--ignore=pennylane/liealg
--ignore=pennylane/noise
--ignore=pennylane/numpy
--ignore=pennylane/ops
--ignore=pennylane/operation.py
--ignore=pennylane/pauli
--ignore=pennylane/queuing
--ignore=pennylane/tape
--ignore=pennylane/templates
--ignore=pennylane/spin
--ignore=pennylane/workflow/return_types_spec.rst
"

IGNORE_OPTS="
${IGNORE_OPTS_CLOSED}
${IGNORE_OPTS_CORE}
${IGNORE_OPTS_AUXILIARY}
${IGNORE_OPTS_TERTIARY}
"

pytest pennylane $IGNORE_OPTS --ignore-glob='*tests*' -vv -x
