class ReturnType:
    activated = False


def enable_return():
    ReturnType.activated = True


def disable_return():
    ReturnType.activated = False


def query_return():
    return ReturnType.activated
