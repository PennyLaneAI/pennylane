qubit q0;
qubit q1;
float theta = 0.5;
x q0;
cx q0, q1;
rx(theta) q0;
inv @ rx(theta) q0;
pow(2) @ x q0;
ctrl @ x q1, q0;