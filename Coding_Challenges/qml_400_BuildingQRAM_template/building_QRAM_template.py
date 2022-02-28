#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        for i in range(3):
            qml.Hadamard(wires=i)
        
        for i in range(8):
            i_bin = np.binary_repr(i,width=3)

            for j in range(3):
                if i_bin[j] == '0':
                    qml.PauliX(wires=j)

            qml.ctrl(qml.RY,control=range(3))(thetas[i],wires=3)

            for j in range(3):
                if i_bin[j] == '0':
                    qml.PauliX(wires=j)
            

        # QHACK #

        return qml.state()

    #print(thetas)
    #print(qml.draw(circuit)())
    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
