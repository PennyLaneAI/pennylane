return_type = False


def enable_return():
    global return_type
    return_type = True


def disable_return():
    global return_type
    return_type = False


def query_return():
    return return_type
