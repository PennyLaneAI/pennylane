PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=native --no-flaky-report
PLUGIN_TESTRUNNER := -m pytest pennylane/devices/tests --tb=native --no-flaky-report

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install PennyLane"
	@echo "  wheel              to build the PennyLane wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  docs               to build the PennyLane documentation"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply black formatter; use with 'check=1' to check instead of modify (requires black)"
	@echo "  lint               to run pylint on source files"
	@echo "  lint-test          to run pylint on test files"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PennyLane you need to have Python 3 installed"
endif
	uv pip install .

.PHONY: wheel
wheel:
	uv build --wheel

.PHONY: dist
dist:
	uv build --sdist 

.PHONY : clean
clean:
	rm -rf pennylane/__pycache__
	rm -rf pennylane/optimize/__pycache__
	rm -rf pennylane/expectation/__pycache__
	rm -rf pennylane/ops/__pycache__
	rm -rf pennylane/devices/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/new_qnode/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	rm -rf doc/code/api
	make -C doc clean

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)

.PHONY:format
format:
ifdef check
	$(PYTHON) -m isort --py 311 --profile black -l 100 -o autoray -p ./pennylane --skip __init__.py --filter-files ./pennylane ./tests --check
	$(PYTHON) -m black -t py311 -t py312 -t py313 -l 100 ./pennylane ./tests --check
else
	$(PYTHON) -m isort --py 311 --profile black -l 100 -o autoray -p ./pennylane --skip __init__.py --filter-files ./pennylane ./tests
	$(PYTHON) -m black -t py311 -t py312 -t py313 -l 100 ./pennylane ./tests
endif

.PHONY: lint
lint:
	$(PYTHON) -m pylint pennylane --rcfile .pylintrc

.PHONY: lint-test
lint-test:
	$(PYTHON) -m pylint tests pennylane/devices/tests --rcfile tests/.pylintrc
