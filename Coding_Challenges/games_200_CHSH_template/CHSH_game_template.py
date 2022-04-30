#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    qml.MottonenStatePreparation((1/(np.sqrt(alpha**2+beta**2))) * np.array([alpha,0,0,beta]),wires=[0,1])
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #

    

    if x == 0:
        vA = np.array([[np.cos(theta_A0),-np.sin(theta_A0)],[np.sin(theta_A0) ,np.cos(theta_A0)]])
#       v1A = np.array([[-np.sin(theta_A0),0],[0,np.cos(theta_A0)]])

    if x == 1:
        vA = np.array([[np.cos(theta_A1),-np.sin(theta_A1)],[np.sin(theta_A1),np.cos(theta_A1)]])
#       v1A = np.array([[-np.sin(theta_A1),0],[0,np.cos(theta_A1)]])

    if y == 0:
#       v0B = np.array([[np.cos(theta_B0),0],[0,np.sin(theta_B0)]])
        vB = np.array([[np.cos(theta_B0),-np.sin(theta_B0)],[np.sin(theta_B0) ,np.cos(theta_B0)]])

    if y == 1:
#       v0B = np.array([[np.cos(theta_B1),0],[0,np.sin(theta_B1)]])
        vB = np.array([[np.cos(theta_B1),-np.sin(theta_B1)],[np.sin(theta_B1) ,np.cos(theta_B1)]])
    # QHACK #

    qml.QubitUnitary(vA,wires=0)
    qml.QubitUnitary(vB,wires=1)
    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    win_probs = []
    for x,y in [(0,0),(0,1),(1,0),(1,1)]:
        probs = chsh_circuit(params[0],params[1],params[2],params[3], x, y, alpha, beta)
        #print(probs)
        if x == 0 or y == 0:
            win_probs.append(probs[0] + probs[3])
        if x==1 and y==1:
            win_probs.append(probs[1] + probs[2])

    return 0.25 * sum(win_probs)
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return 1 - winning_prob(params, alpha, beta)

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([0.5,0.5,0.5,0.5],requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=0.1)
    steps = 500


    # QHACK #
    
    # set the initial parameter values
    params = init_params


    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost,params)
        #print(winning_prob(params, alpha, beta))
        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
