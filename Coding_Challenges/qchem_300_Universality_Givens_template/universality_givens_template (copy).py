#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml

def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #

    dev1 = qml.device("default.qubit", wires=6)

    @qml.qnode(dev1)
    def circuit(params):
        t1,t2,t3 = params

        #print(params)
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        qml.DoubleExcitation(t1, wires=[0, 1, 2, 3])
        qml.DoubleExcitation(t2, wires=[2, 3, 4, 5])

        qml.ctrl(qml.SingleExcitation,control = 0)(t3,wires=[1,3])
        
        return qml.state()

    def cost(x):

        state = circuit(x) 
        #print(state[48])

        #co = ((qml.math.toarray(state[48]).real) - a) ** 2 + (qml.math.toarray(state[12]).real - b) ** 2 + (qml.math.toarray(state[3]).real - c) ** 2 + (qml.math.toarray(state[36]).real - d) ** 2

        co = ((state[48]) - a) ** 2 + ((state[12]) - b) ** 2 + ((state[3]) - c) ** 2 + ((state[36]) - d) ** 2

        

        #print(type(co))
        return abs(co)

    init_params = np.array([0.011, 0.012,0.013], requires_grad=True)
    #print(cost(init_params))

    # initialise the optimizer
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    
    # set the number of steps
    steps = 100
    # set the initial parameter values
    params = init_params
    
    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)
        
        params = np.clip(params,-np.pi,np.pi)
        #if (i + 1) % 5 == 0:
        #    print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

    #print("Optimized rotation angles: {}".format(params))

    return params
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
