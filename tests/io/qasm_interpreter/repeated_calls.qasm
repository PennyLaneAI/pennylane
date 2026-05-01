qubit q0;

def inc(qubit q0, float f) {
  f += 1.0;
  float g = 1.0;
  g -= 0.5;
  rx(f) q0;
  ry(g) q0;
}

inc(q0, 1.0);
inc(q0, 1.0);
inc(q0, 10.0);
