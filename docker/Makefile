
.PHONY: help build-base build-interface build-interface-gpu build-plugin

help:
	@echo "Makefile commands:"
	@echo "build-base"
	@echo "build-qchem"
	@echo "build-interface"
	@echo "build-interface-gpu"

build-base:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile .

build-qchem:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/qchem -f docker/qchem.dockerfile .

build-interface:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/$(interface-name) -f docker/build_interface.dockerfile \
	--build-arg INTERFACE_NAME=$(interface-name) .

build-interface-gpu:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/$(interface-name) -f docker/build_interface.dockerfile \
	--build-arg INTERFACE_NAME=$(interface-name)

build-plugin:
	@docker build -t pennylane/base -f docker/pennylane.dockerfile . \
	&& docker build -t pennylane/$(plugin-name) -f docker/plugins/build_plugin.dockerfile \
	--build-arg PLUGIN_NAME=$(plugin-name) .