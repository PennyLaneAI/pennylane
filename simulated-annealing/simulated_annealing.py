# Import the libraries
import pennylane as qml
from pennylane import numpy as np
from math import exp

# Simulated Annealing method
# Due to the periodicity of theta and phi, I have discarded the boundary condition limit
def simulated_annealing(objective, bounds, temperature, iterations, learning_rate):
        # Define initial state
        s0 = np.empty(np.shape(bounds)[0])
        s_eval = 0 
        for i in range(len(s0)):
            s0[i] = bounds[i][0] + np.random.random()* (bounds[i][1] - bounds[i][0])

        # Evaluate current state of the system 
        s = s0
        s_eval = objective(s0)

        for k in range(iterations):
            # Find a candidate element
            s_new = np.empty(np.shape(bounds)[0])
            for i in range(len(s_new)):
                rng = bounds[i][1] - bounds[i][0]     # Range of the variable
                s_new[i] = s[i] + np.random.uniform(bounds[i][0] - rng/2, bounds[i][0] + rng/2)*learning_rate

            # Evaluate the candidate element        
            s_new_eval = objective(s_new)
            # Update the state of the system
            if s_new_eval < s_eval:
                s = s_new
                s_eval = s_new_eval

            delta = s_new_eval - s_eval
            t = temperature/float(k+1) 

            # Metropolis acceptance criterion
            metropolis = exp(-delta/t)
            if metropolis > np.random.uniform():
                s = s_new
                s_eval = s_new_eval
        return s, s_eval
      
# The circuit to be optimized
# The optimization routine has been wrapped in a function
def qft_via_qml(n, m, temperature, steps = 5000, learning_rate = 0.01):
    ''' A function to generate the Quantum Fourier Transform of a number m via a quantum circuit with n-qubits. 
    
    Args:
    n: It is the number of qubits in the variational quantum circuit.
    m: It is the number used to train the variational quantum circuit. Its value ranges from 0 to 2^n - 1
    steps: It is the number of iterations of the optimzation process.
    learning_rate: This is the step size of the optimizer.
    
    Return value:
    circuit: The variational quantum circuit comprising n qubits.
    param_arr: Shape = (steps, n). It consists of the values of the parameters after each optimization step.
    cost_arr: Size = n. It consists of the cost value of the circuit after each optimization step.
    cost: The cost function
    '''
    # Task 2: Load a Quantum Device
    dev = qml.device('default.qubit', wires = n)
    
    # Task 3-6: Create the Quantum Circuit
    @qml.qnode(dev)
    def circuit(theta):
        for i in range(n):
            qml.Hadamard(wires = i)
            qml.RZ(theta[i], wires = i)
        qml.adjoint(qml.QFT)(wires=range(n))
        return qml.probs(wires = range(n))
    
    # Task 7: Create the Cost Function
    def cost(params):
        y = np.zeros(2**n)
        y[m] = 1
        probs = circuit(params)
        return np.sum(np.square((y-probs)))
    
    # Defining the bounds of the angles
    bounds = np.empty((n, 2))
    for bound in bounds:
        bound[0] = 0
        bound[1] = 2*np.pi
    
    angles, cost = simulated_annealing(cost, bounds, temperature, steps, learning_rate)

    # Print the results of optimization
    print("Optimized rotation angles: ", angles)
    print("Cost value at optimized parameters: ",cost)
    
    return circuit, angles, cost
  
  # Run the optimization routine
  circuit, angles, cost = qft_via_qml(2, 3, 5, steps = 6000, learning_rate = 0.01)
  
  # Draw the circuit
  qml.draw_mpl(circuit)(angles)
