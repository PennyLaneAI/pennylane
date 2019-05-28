.. _Hybrid_Computation:

Hybrid Computation
====================

PennyLane allows us to string together quantum and classical computations in highly-structured ways and the combined hybrid computation can be differentiated and trained end-to-end. 

This opens up the possibility of doing many interesting things. For example:

* Pre-processing (or post-processing) the input (output) of a quantum circuit using a neural network. Both the classical and quantum components can have trainable weights!

* Combining the power of several quantum circuits, either in series or in parallel. These circuits can even be running on different hardware!

* Training a model using both GPUs and QPUs.

This tutorial looks at a couple of hybrid computation examples in PennyLane. 


.. Add the location of your Jupyter notebook below!

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   hybrid_qubit_photon.ipynb
   hybrid_gpu_qpu.ipynb


.. Copy the template below in order to create a link to your notebook, and a thumbnail.

.. _`Hybrid Quantum-Classical Optimization`: hybrid_qubit_photon.html
.. |qco| image:: figures/bloch.gif
   :width: 260px
   :align: middle
   :target: hybrid_qubit_photon.html

.. _`Hybrid GPU-QPU Optimization`: hybrid_gpu_qpu.html
.. |hgq| image:: figures/bloch.gif
   :width: 260px
   :align: middle
   :target: hybrid_gpu_qpu.html


.. Add your thumbnail to the table in the Gallery!

.. rst-class:: gallery-table

+---------------------------------------------+---------------------------------------+
| |qco|                                       | |hgq|                                 |                                       
|                                             |                                       |                  
| `Hybrid Quantum-Classical Optimization`_    | `Hybrid GPU-QPU Optimization`_        | 
+---------------------------------------------+---------------------------------------+
