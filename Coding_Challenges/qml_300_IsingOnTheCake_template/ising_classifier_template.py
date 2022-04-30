import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires) 

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(data,angles):
        #print(data)
        for l in range(n_layers):
            for i in range(num_wires):
                if data[i] == 0:
                    qml.PauliX(wires=i)
    
                qml.RX(angles[l][i][0],wires=i)
                qml.RY(angles[l][i][1],wires=i)
                qml.RZ(angles[l][i][2],wires=i)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])
        return qml.expval(qml.PauliZ(0))

    # Define a cost function below with your needed arguments
    def cost(angles,b,X,Y):

        # QHACK #
        
        # Insert an expression for your model predictions here
        predictions = []
        for data in X:
            predictions.append(circuit(data,angles)+b)

        # QHACK #

        
        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here

    n_layers = 3
    
    weights_init = np.random.randn(n_layers,num_wires,3,requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(0.2)
    bias_init = np.array(0.0, requires_grad=True)

    batch_size = 10
    
    weights = weights_init
    bias = bias_init

    acc = 0
    while acc < 0.9:

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(ising_configs), (batch_size,))
        X_batch = ising_configs[batch_index]
        Y_batch = labels[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        # Compute accuracy
        #print(ising_configs[0])
        predictions = [int(np.sign(circuit(x,weights)+bias)) for x in ising_configs]
        acc = accuracy(labels, predictions)
        
        #print("Iter: | Cost: {:0.7f} | Accuracy: {:0.7f} ".format( cost(weights, bias,ising_configs , labels), acc))
    
    # QHACK #
    

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
