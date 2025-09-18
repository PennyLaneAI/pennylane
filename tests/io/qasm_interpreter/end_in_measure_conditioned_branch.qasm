qubit q0;
qubit q1;
qubit q2;

float theta = 0.2;
int power = 2;

ry(theta / 2) q0;
rx(theta) q1;
pow(power) @ x q0;

def random(qubit q) -> bit {
  bit b = "0";
  h q;
  measure q -> b;
  return b;
}

bit m = random(q2);

if (m) {
  int i = 0;
  while (i < 5) {
    i = i + 1;
    rz(i) q1;
    if (m) {
      break;
    }
  }
} else {
  x q0;
  end;
  y q0;
}