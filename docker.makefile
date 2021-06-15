#vars
IMAGENAME=pennylane-build
BUILDNAME=pennylane

.PHONY: help build-base build-tensorflow build-torch build-jax build-jax-gpu build-tensorflow-gpu build-torch-gpu

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
			@echo "build-tensorflow-gpu"
			@echo "build-jax-gpu"
			@echo "build-torch-gpu"

build-base:
	    @docker build -t pennylane/base -f docker/pennylane.dockerfile .

build-qchem:
	    @docker build -t pennylane/qchem -f docker/qchem.dockerfile .

build-tensorflow:
	    @docker build -t pennylane/tensorflow -f docker/tensorflow.dockerfile .

build-torch:
	     @docker build -t pennylane/torch -f docker/torch.dockerfile .

build-jax:
	     @docker build -t pennylane/jax -f docker/jax.dockerfile .

build-tensorflow-gpu:
	@docker build -t pennylane/cuda/base -f docker/gpu-cuda/cuda-base.dockerfile . \
	&& docker build -t pennylane/tensorflow/gpu -f docker/gpu-cuda/tensorflow.dockerfile .

build-jax-gpu:
	@docker build -t pennylane/cuda/base -f docker/gpu-cuda/cuda-base.dockerfile . \
	&& docker build -t pennylane/jax/gpu -f docker/gpu-cuda/jax.dockerfile .

build-torch-gpu:
	@docker build -t pennylane/cuda/base -f docker/gpu-cuda/cuda-base.dockerfile . \
  && docker build -t pennylane/torch/gpu -f docker/gpu-cuda/torch.dockerfile .

build-all:
			 	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
				&& docker build -t pennylane/tensorflow -f docker/tensorflow.dockerfile . \
				&& docker build -t pennylane/torch -f docker/torch.dockerfile . \
				&& docker build -t pennylane/qchem -f docker/qchem.dockerfile . \
				&& docker build -t pennylane/jax -f docker/jax.dockerfile .

build-all-gpu:
	@docker build -t pennylane/cuda/base -f docker/gpu-cuda/cuda-base.dockerfile . \
	&& docker build -t pennylane/tensorflow/gpu -f docker/gpu-cuda/tensorflow.dockerfile . \
	&& docker build -t pennylane/torch/gpu -f docker/gpu-cuda/torch.dockerfile . \
	&& docker build -t pennylane/jax/gpu -f docker/gpu-cuda/jax.dockerfile .
