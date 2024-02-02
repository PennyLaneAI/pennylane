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
	@echo "  lock               to run poetry lock and auto-generate requirements.txt files"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PennyLane you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

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
	$(PYTHON) $(PLUGIN_TESTRUNNER) --device=default.qubit.autograd

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
	$(PYTHON) $(PLUGIN_TESTRUNNER) --device=default.qubit.autograd $(COVERAGE) --cov-append

.PHONY:format
format:
ifdef check
	black -l 100 ./pennylane ./tests --check
else
	black -l 100 ./pennylane ./tests
endif

.PHONY: lint
lint:
	pylint pennylane --rcfile .pylintrc

.PHONY: lint-test
lint-test:
	pylint tests pennylane/devices/tests --rcfile tests/.pylintrc

.PHONY: lock
lock:
	@echo "Calling poetry lock:"
	@poetry lock --no-update
	@echo "Using poetry export to generate all requirements.txt files"
	@poetry export -f requirements.txt -o requirements.txt --without-hashes --without-urls
	@poetry export -f requirements.txt -o requirements-ci.txt --without-hashes --without-urls --only ci
	@poetry export -f requirements.txt -o requirements-dev.txt --without-hashes --without-urls --only dev
	@poetry export -f requirements.txt -o doc/requirements.txt --without-hashes --without-urls --with doc,torch,jax,qchem,data
	@echo "Hacking in un-poetic changes to doc/requirements.txt:"
	@cat doc/.extra-requirements.txt >> doc/requirements.txt
	@grep "torch==2.2.0" doc/requirements.txt > /dev/null && sed -i '' 's/torch==2\.2\.0/torch==1.9.0/g' doc/requirements.txt
	@grep "ml-dtypes==0.3.2" doc/requirements.txt > /dev/null && sed -i '' 's/ml-dtypes==0\.3\.2/ml-dtypes==0.2.0/g' doc/requirements.txt
