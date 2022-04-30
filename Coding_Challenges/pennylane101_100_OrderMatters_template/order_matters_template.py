#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def compare_circuits(angles):
    """Given two angles, compare two circuit outputs that have their order of operations flipped: RX then RY VERSUS RY then RX.

    Args:
        - angles (np.ndarray): Two angles

    Returns:
        - (float): | < \sigma^x >_1 - < \sigma^x >_2 |
    """

    # QHACK #

    # define a device and quantum functions/circuits here

    dev1 = qml.device("default.qubit", wires=2)

    @qml.qnode(dev1)
    def cir1(t1,t2,wire=0):
        qml.RX(t1,wires=wire)
        qml.RY(t2,wires=wire)
        return qml.expval(qml.PauliX(wire))
    
    @qml.qnode(dev1)
    def cir2(t1,t2,wire=1):
        qml.RY(t2,wires=wire)
        qml.RX(t1,wires=wire)
        return qml.expval(qml.PauliX(wire))
    
    return abs(cir1(angles[0],angles[1])-cir2(angles[0],angles[1]))

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    angles = np.array(sys.stdin.read().split(","), dtype=float)
    output = compare_circuits(angles)
    print(f"{output:.6f}")
