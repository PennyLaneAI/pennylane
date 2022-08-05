import pennylane as qml


def pytest_sessionstart(session):
    qml.enable_return()


def pytest_sessionfinish(session, exitstatus):
    qml.disable_return()
