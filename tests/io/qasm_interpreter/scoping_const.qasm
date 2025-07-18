qubit q0;

const float a = 0.5;
float b = 1.0;

def inc(qubit q0, float f) {
  float a = 9.0;
  f += 1.0;
  const float g = f;
  return g;
}

const float c = inc(q0, b);
const float d = inc(q0, b + 0.5);
