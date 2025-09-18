qubit q0;
qubit q1;
qubit[2] p;

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

gate custom_three f
{
    x f[0];
    y f[1];
}

custom_two(pi) q1, q0;
custom_three p;
