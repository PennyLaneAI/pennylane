"""Classical auxiliary functions"""


def states_vector(wires):
    """prints out list of states in a computational basis"""
    state_vector = list()
    for i in range(2 ** len(wires)):
        state_vector.append("|" + "0" * (len(wires) - len(bin(i)[2:])) + bin(i)[2:] + ">")
    return state_vector


def gcd(a, b):
    """
    Euclid's algorithm - finds greater common devider
    Note: doesn't work correctly for too big numbers (because a%b and int(a/b) do not work)

    """

    ## check order
    if a < b:
        a, b = b, a

    ## algorithm
    r = a % b
    if r == 0:
        return b

    return gcd(b, r)


def diophantine_equation_auxiliary(k_list, alpha, beta, i):
    """
    Auxiliary recursive function for finding alpha and beta in r_n = alpha*a + beta*b, where n: r_n = gcd(a,b)
    before algorithms' execution, array of k should be defined
    i denotes level in Euclid's algorithm
    algorithm should be initialized with alpha = 1, beta = -k_(n-2), i = n-2, where n: r_n = gcd(a,b)
    Note: doesn't work correctly for too big numbers (because a%b and int(a/b) do not work)

    """
    if i != 1:
        return diophantine_equation_auxiliary(
            k_list=k_list, alpha=beta, beta=alpha - beta * k_list[i - 2], i=i - 1
        )

    return [alpha, beta]


def diophantine_equation(a, b):
    """
    solves diophantine equation, i.e. given a,b returns x,y such that ax + by = gcd(a,b)
    Euclid's algorithm produces set of values (k_i,r_i), where r_i = k_i*r_(i+1) + r_(i+2), i goes from 0 to n
    """

    ## check order
    flag = 0
    if a < b:
        a, b = b, a
        flag = 1

    # if b == gcd(a,b)
    if b == gcd(a, b):
        if flag == 0:
            return [0, 1]
        if flag == 1:
            return [1, 0]

    # initialize r and lists of r and k from the equation a = kb + r
    r_list = list([a])
    k_list = list()

    # forward part of the algorithm (Nielsen Chuang p.628)
    while b != 0:
        r_list.append(b)
        k_list.append(a // b)
        a, b = b, a % b

    # backward part of the algorithm (Nielsen Chuang p. 629)
    if flag == 0:
        return diophantine_equation_auxiliary(
            k_list=k_list, alpha=1, beta=-k_list[-2], i=len(r_list) - 2
        )
    if flag == 1:
        return diophantine_equation_auxiliary(
            k_list=k_list, alpha=1, beta=-k_list[-2], i=len(r_list) - 2
        )[::-1]


def modular_multiplicative_inverse(a, N):
    """finds modular multiplicative inverse using diophantine_equation"""
    ## check co-primality
    if gcd(a, N) != 1:
        raise Exception("a and N should be co-prime, i.e. gcd(a,N) should be 1")
    #         if a >= N:
    #             raise Exception('a should be less than N - check that order of arguments of the function is right')

    ## algorithm
    # get inverse
    inverse = diophantine_equation(a=a, b=N)[0]

    # make inverse positive if necessary
    if inverse < 0:
        inverse = inverse % N

    return inverse
