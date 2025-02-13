from pennylane.labs.pf.hamiltonians.dabna6state import six_mode_ham

if __name__ == "__main__":
    vham = six_mode_ham()

    for i in range(10):
        print(vham.fragment(i))
