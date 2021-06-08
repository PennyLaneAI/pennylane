#Dockerfile vars
#
# ARG BASE_IMAGE=18.04
# ARG PYTHON_VERSION=3.7
# ARG TENSORFLOW_VERSION=2.5
# ARG PyTorch_VERSION=1.8.1
# #ARG Jax_VERSION=

#make -f docker.makefile build-base

#vars
IMAGENAME=pennylane-build
BUILDNAME=pennylane

.PHONY: help build-base build-tensorflow build-torch build-jax build-tensorflow-gpu build-torch-gou

help:
	    @echo "Makefile arguments:"
	    @echo ""
	    @echo "BASE_IMAGE - Ubuntu Version"
	    @echo "PYTHON_VERSION - python version"
			@echo "TENSORFLOW_VERSION - tensorflow version"
			@echo "PyTorch_VERSION - pytoch version"
			@echo "JAX_VERSION - jax version"
	    @echo ""
	    @echo "Makefile commands:"
	    @echo "build-base"
	    @echo "build-tensorflow"
	    @echo "build-torch"
			@echo "build-jax"

build-base:
	    @docker build -t pennylane -f docker/pennylane.dockerfile .

build-tensorflow:
	    @docker build -t pennylane -f docker/tensorflow.dockerfile .

build-torch:
	     @docker build -t pennylane -f docker/pytorch.dockerfile .

build-jax:
	     @docker build -t pennylane -f docker/jax.dockerfile .
