PYTHON := python
COVERAGE := --cov=pennylane_qchem --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=native

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install PennyLane-Qchem"
	@echo "  wheel              to build the PennyLane-Qchem wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  coverage           to generate a coverage report"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PennyLane-Qchem you need to have Python 3 installed"
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
	rm -rf qchem/__pycache__
	rm -rf tests/__pycache__
	rm -rf .coverage coverage_html_report/
	rm -rf dist
	rm -rf build
	rm -rf tmp
	rm -rf timer.dat

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
