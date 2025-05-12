qubit q0;
qubit q1;
float theta = 0.5;
x q0;
rx(theta) q0;
inv @ rx(theta) q0;
ctrl @ x q1, q0;