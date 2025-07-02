qubit q0;
qubit q1;

const float pi = 3.14159;

gate custom(θ) a, b
{
    CX a, b;
    CX a, b;
    rx(θ / 2) b;
}

gate custom_two(θ) c, d
{
    y c;
    custom(θ / 2) d, c;
    x d;
}

custom_two(pi) q1, q0;
