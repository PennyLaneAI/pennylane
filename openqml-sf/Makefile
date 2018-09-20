PYTHON3 := $(shell which python3 2>/dev/null)
COVERAGE3 := $(shell which coverage3 2>/dev/null)

PYTHON := python3
COVERAGE := coverage3
COPTS := run --append
TESTRUNNER := -m unittest discover tests

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install OpenQML-SF"
	@echo "  wheel              to build the OpenQML-SF wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  coverage           to generate a coverage report"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install OpenQML-SF you need to have Python 3 installed"
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
	rm -rf qmlt/__pycache__
	rm -rf qmlt/numeric/__pycache__
	rm -rf qmlt/tf/__pycache__
	rm -rf tests/__pycache__
	rm -rf logsNUM logsAUTO
	rm -rf tests/logsNUM
	rm -rf tests/logsAUTO
	rm -rf examples/logsNUM
	rm -rf examples/logsAUTO
	rm -rf dist
	rm -rf build
	rm -rf .coverage coverage_html_report/

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean


test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(COVERAGE) $(COPTS) $(TESTRUNNER)
	$(COVERAGE) report
	$(COVERAGE) html
