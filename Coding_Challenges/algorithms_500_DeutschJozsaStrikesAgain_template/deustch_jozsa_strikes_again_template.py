#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #

    dev = qml.device("default.qubit", wires=6, shots=1)

    @qml.qnode(dev)
    def circuit(fn):
        qml.Hadamard(wires = 0)
        qml.Hadamard(wires = 1)

        qml.PauliX(wires = 2)
        qml.Hadamard(wires = 2)
        
        fn([0,1,2])

        qml.Hadamard(wires = 0)
        qml.Hadamard(wires = 1)

        return (qml.sample(wires=[0,1]))

    cnt1 = 0
    cnt2 = 0
    for fn in fs:
        
        sample  = circuit(fn)
        #print(sample)
        #print(qml.draw(circuit)())
        if sample[0] == 0 and sample[1] == 0:
            cnt1+=1
        else:
            cnt2 += 1

    if cnt1 == 4 or cnt2 == 4:
        return "4 same"
    else:
        return "2 and 2"

    return(type_)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
