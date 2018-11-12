Notebook downloads
==================

In addition to the :ref:`qubit rotation <qubit_rotation>`, :ref:`Gaussian transformation <gaussian_transformation>`, and :ref:`hybrid quantum optimization <plugins_hybrid>` tutorials in the documentation, we also have a selection of `Jupyter notebooks <http://jupyter.org/>`_ available walking through some more advanced optimizations made possible with PennyLane, including supervised learning and quantum generative adversarial networks.


To open the interactive notebooks, launch the Jupyter notebook environment by clicking on the 'Jupyter notebook' shortcut in the start menu (Windows), or by running the following in the Anaconda Prompt/Command Prompt/Terminal:
::

	jupyter notebook

Your web browser should open with the Jupyter notebook home page; simply click the 'Upload' button, browse to the tutorial file you downloaded below, and upload the file. You will now be able to open it and work through the tutorial.

Alternatively, you can also view the notebook contents on GitHub, without interactivity.


Qubit notebooks
---------------

1. **Qubit rotation** (:download:`Q1_qubit-rotation.ipynb <../../examples/Q1_qubit-rotation.ipynb>`/|Q1|)

   Use PennyLane to optimize two rotation gates to flip a single qubit from state :math:`\ket{0}` to state :math:`\ket{1}`. This notebook follows the same process as the :ref:`qubit rotation tutorial <qubit_rotation>`, but with more emphasis on the optimization procedure, exploring the optimization landscape and different optimizers.

2. **Variational quantum eigensolver** (:download:`Q2_variational-quantum-eigensolver.ipynb <../../examples/Q2_variational-quantum-eigensolver.ipynb>`/|Q2|)

   This notebook demonstrates the principle of a variational quantum eigensolver (VQE) :cite:`peruzzo2014variational`. To showcase the hybrid computational capabilities of PennyLane, we train a quantum circuit to find the parameterized state :math:`\ket{\psi_v}` that minimizes the squared energy expectation for a Hamiltonian :math:`H`:

   .. math:: \braketT{\psi_v}{H}{\psi_v}^2  =( 0.1 \braketT{\psi_v}{(\I\otimes \sigma_x)}{\psi_v} + 0.5 \braketT{\psi_v}{(\I\otimes \sigma_y)}{\psi_v} )^2,

   before training a second variational quantum circuit :math:`f(v_1,v_2)` to minimize the the energy expectation of a fixed quantum state :math:`\ket{\psi}`:

   .. math:: \braketT{\psi}{H}{\psi}^2  =( v_1 \braketT{\psi}{(\I\otimes \sigma_x)}{\psi} + v_2\braketT{\psi}{(\I\otimes \sigma_y)}{\psi} )^2.


3. **Variational classifier** (:download:`Q3_variational-classifier.ipynb <../../examples/Q3_variational-classifier.ipynb>`/|Q3|)

   In this notebook we show how to use PennyLane to implement variational quantum classifiers - quantum circuits that can be trained from labeled data to classify new data samples. This optimization example demonstrates how to encode binary inputs into the initial state of the variational circuit, which is simply a computational basis state.

   We then show how to encode real vectors as amplitude vectors (amplitude encoding) and train the model to recognize the first two classes of flowers in the Iris dataset.


4. **Quantum generative adversarial network (QGAN)** (:download:`Q4_quantum-GAN.ipynb <../../examples/Q4_quantum-GAN.ipynb>`/|Q4|)

   This demo constructs a quantum generative adversarial network (QGAN) :cite:`lloyd2018quantum,dallaire2018quantum` using two subcircuits, a generator and a discriminator. The generator attempts to generate synthetic quantum data to match a pattern of "real" data, while the discriminator tries to discern real data from fake data. The gradient of the discriminator's output provides a training signal for the generator to improve its fake generated data.


.. |Q1| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/Q1_qubit-rotation.ipynb" target="_blank">view on GitHub</a>

.. |Q2| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/Q2_variational-quantum-eigensolver.ipynb" target="_blank">view on GitHub</a>

.. |Q3| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/Q3_variational-classifier.ipynb" target="_blank">view on GitHub</a>

.. |Q4| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/Q4_quantum-GAN.ipynb" target="_blank">view on GitHub</a>



Continuous-variable notebooks
-----------------------------

1. **Photon redirection** (:download:`CV1_photon-redirection.ipynb <../../examples/CV1_photon-redirection.ipynb>`/|CV1|)

   Starting with a photon in mode 0 of a variational quantum optical circuit, the goal is to use PennyLane to optimize a beamsplitter to redirect the photon to mode 1. This notebook follows the same process as the :ref:`photon redirection tutorial <photon_redirection>`, but with more emphasis on the optimization procedure, comparing the use of the gradient-descent optimizer with and without momentum.

2. **Quantum neural networks** (:download:`CV2_quantum-neural-net.ipynb <../../examples/CV2_quantum-neural-net.ipynb>`/|CV2|)

   In this notebook, we show how a continuous-variable quantum neural network model :cite:`killoran2018continuous` can be used to learn a fit for a one-dimensional function when being trained with noisy samples from that function. In this case, the variational quantum circuit is trained to fit a one-dimensional sine function from noisy data.


.. |CV1| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/CV1_photon-redirection.ipynb" target="_blank">View on GitHub</a>

.. |CV2| raw:: html

   <a href="https://github.com/XanaduAI/pennylane/blob/master/examples/CV2_quantum-neural-net.ipynb" target="_blank">View on GitHub</a>
