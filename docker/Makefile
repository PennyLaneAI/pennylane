
.PHONY: help build-base build-interface build-interface-gpu build-plugin

help:
	@echo "Makefile commands:"
	@echo "build-base"
	@echo "build-interface"
	@echo "build-interface-gpu"

available-interfaces:
	@echo "The following interfaces are available for PennyLane:"
	@echo "1.tensorflow 2.torch 3.jax"

available-plugins:
	@echo "The following plugins are available for PennyLane:"
	@echo "1.qiskit 2.amazon-braket 3.sf 4.cirq 5.qulacs 6.aqt"
	@echo "7.honeywell 8.pq 9.qsharp 10.forest 11.orquestra 12.ionq"

build-base:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile .

build-interface:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/$(interface-name) -f docker/interfaces/build_interface.dockerfile \
	--build-arg INTERFACE_NAME=$(interface-name) .

build-interface-gpu:
	@docker build -t pennylane/cuda/base -f docker/interfaces/gpu-cuda/cuda-base.dockerfile . \
	&& docker build -t pennylane/cuda/$(interface-name) -f docker/interfaces/gpu-cuda/build_interface_gpu.dockerfile \
	--build-arg INTERFACE_NAME=$(interface-name) .

build-plugin:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/$(plugin-name) -f docker/plugins/build_plugin.dockerfile \
	--build-arg PLUGIN_NAME=$(plugin-name) .
