qubit q0;
qubit q1;

const float pi = 3.14159;

gate custom(θ) a, b
{
  CX a, b;
  CX a, b;
  rx(θ / 2) b;
}

custom(pi / 2) q0, q1;