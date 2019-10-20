.. role:: html(raw)
   :format: html

.. _intro_interfaces:

Interfaces
==========

PennyLane integrates quantum nodes made up of variational circuits with other programming
and machine learning frameworks.
Such frameworks are called *interfaces*. The default interface, implicitly used in the previous section on
how to program :ref:`variational circuits <intro_vcircuits>`, is NumPy.

Currently, there is support for the following three interfaces:

.. raw:: html

    <style>
        #interfaces .card {
            box-shadow: none!important;
        }
        #interfaces .card:hover {
            box-shadow: none!important;
        }
    </style>
    <div id="interfaces" class="container mt-2 mb-2">
        <div class="row mt-3">
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="interfaces/numpy.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                            <img src="../_static/numpy.jpeg" class="card-img-top" style="width:60%;"></img>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="interfaces/torch.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                          <img src="../_static/pytorch.png" class="card-img-top"></img>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="interfaces/tf.html">
                    <div class="card rounded-lg py-2" style="height:100%;">
                        <div class="d-flex justify-content-center align-items-center" style="height:100%;">
                            <img src="../_static/tensorflow.png" class="card-img-top" style="width:90%;"></img>
                        </div>
                    </div>
                </a>
            </div>
        </div>
    </div>


.. toctree::
    :hidden:

    interfaces/numpy
    interfaces/torch
    interfaces/tf
