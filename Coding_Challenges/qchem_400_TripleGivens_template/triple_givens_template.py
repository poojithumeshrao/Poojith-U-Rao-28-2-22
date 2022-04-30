import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #

    
    # QHACK #

    mat = np.identity(2**6)

    mat[7][7] = np.cos(gamma/2)
    mat[7][56] = -np.sin(gamma/2)
    mat[56][7] = np.sin(gamma/2)
    mat[56][56] = np.cos(gamma/2)

    return mat


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #

    alpha,beta,gamma = angles
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.PauliX(wires=2)
    
    qml.SingleExcitation(alpha,wires=[0,5])
    qml.DoubleExcitation(beta,wires=[0,1,4,5])
    qml.QubitUnitary(triple_excitation_matrix(gamma),wires=range(6))
    

    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
