from pennylane.labs.sample_func import add_num


def test_add_num():
    assert add_num(1, 2) == 3
