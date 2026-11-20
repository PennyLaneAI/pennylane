
.. raw:: html

    <style>
    tr {
        border-top: #d3d3d3 1px solid;
        border-bottom: #d3d3d3 1px solid;
    }
    .row-odd {
        background-color: #f7f7f7;
    }
    </style>

Supported devices
=================

Not all PennyLane devices currently work with Catalyst, and for those that do, their supported
feature set may not necessarily match supported features when used without :func:`~.qjit`.

Supported backend devices include:

.. list-table::
  :widths: 20 80

  * - ``lightning.qubit``

    - A fast state-vector qubit simulator written with a C++ backend. See the
      `Lightning-Qubit documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/main/pennylane_lightning/lightning_qubit/lightning_qubit.toml>`__
      for natively supported instructions.

  * - ``lightning.kokkos``

    - A fast state-vector qubit simulator utilizing the Kokkos library for CPU and GPU accelerated
      circuit simulation. See the
      `Lightning-Kokkos documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/main/pennylane_lightning/lightning_kokkos/lightning_kokkos.toml>`__
      for natively supported instructions.

  * - ``lightning.amdgpu``

    - A fast state-vector qubit simulator based on the ``lightning.kokkos`` device to give maximum performance on AMD GPUs. See the
      `Lightning-AMDGPU documentation <https://docs.pennylane.ai/projects/lightning/en/latest/lightning_amdgpu/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/main/pennylane_lightning/lightning_kokkos/lightning_kokkos.toml>`__ 
      (shared with ``lightning.kokkos``) for natively supported instructions.

  * - ``lightning.gpu``

    - A fast state-vector qubit simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`__
      for the GPU-accelerated quantum simulation. See the
      `Lightning-GPU documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
      for more details, as well as its
      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/main/pennylane_lightning/lightning_gpu/lightning_gpu.toml>`__
      for natively supported instructions.

  * - ``braket.aws.qubit``

    - Interact with quantum computing hardware devices and simulators through Amazon Braket. To use
      this device with Catalyst, make sure to install the
      `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.
      See the
      `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/runtime/lib/backend/openqasm/braket_aws_qubit.toml>`__
      for natively supported instructions.

  * - ``braket.local.qubit``

    - Run gate-based circuits on the Braket SDK's local simulator. To use
      this device with Catalyst, make sure to install the
      `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.
      See the
      `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/runtime/lib/backend/openqasm/braket_local_qubit.toml>`__
      for natively supported instructions.

  * - ``qrack.simulator``

    - `Qrack <https://github.com/unitaryfund/qrack>`__ is a GPU-accelerated quantum computer
      simulator with many novel optimizations including hybrid stabilizer simulation. To use this
      device with Catalyst, make sure to install the
      `PennyLane-Qrack plugin <https://pennylane-qrack.readthedocs.io/en/latest/>`__, and check out
      the `QJIT compilation with Qrack and Catalyst tutorial <https://pennylane.ai/qml/demos/qrack/>`__.

      See the `Catalyst configuration file <https://github.com/unitaryfund/pennylane-qrack/blob/master/pennylane_qrack/QrackDeviceConfig.toml>`__
      for natively supported instructions.

  * - ``oqc.cloud``

    - Execute on `Oxford Quantum Circuits (OQC) <https://www.oqc.tech/>`__ superconducting hardware,
      via `OQC Cloud <https://docs.oqc.app>`__. To use OQC Cloud with Catalyst, simply `install the
      client <https://docs.oqc.app/installation.html>`__, ensure your credentials are set as
      environment variables, and use the ``backend`` argument to specify the OQC backend to use:

      .. code-block:: python

          import os
          os.environ["OQC_EMAIL"] = "your_email"
          os.environ["OQC_PASSWORD"] = "your_password"
          os.environ["OQC_URL"] = "oqc_url"

          dev = qp.device("oqc.cloud", backend="lucy", shots=2012, wires=2)

      See the `Catalyst configuration file <https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/third_party/oqc/src/oqc.toml>`__
      for natively supported instructions.

  * - ``null.qubit``

    - A device that performs no numerical operations, useful for resource estimation and benchmarking.
      See the `NullQubit documentation <https://pennylane.ai/devices/null-qubit>`__ for more details.
