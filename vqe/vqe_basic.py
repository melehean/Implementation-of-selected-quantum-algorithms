import pennylane as qml
from pennylane import numpy as np
from numpy import linalg as LA
# Hamiltonian Z+2*X+3*Y
# Ansatz RY

QUBITS_NUMBER = 3

dev = qml.device("default.qubit",shots=1000, wires=QUBITS_NUMBER)

def ansatz(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RY(params[2], wires=2)

@qml.qnode(dev)
def circuit_Z(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit_Y(params):
    ansatz(params)
    return qml.expval(qml.PauliY(2))

@qml.qnode(dev)
def circuit_X(params):
    ansatz(params)
    return qml.expval(qml.PauliX(1))

def cost(params):
    z_expectation = circuit_Z(params)
    y_expectation = circuit_Y(params)
    x_expectation = circuit_X(params)
    return z_expectation + 5*x_expectation + 2*y_expectation

def getExactValue():
    X = np.array([[0,1],
                  [1,0]])
    Y = np.array([[0,-1j],
                  [1j,0]])
    Z = np.array([[1,0],
                  [0,-1]])
    H = Z + 2*X + 3*Y
    eigenValues, eigenVectors = LA.eig(H)
    return np.amin(eigenValues).real


if __name__ == '__main__':
    params = np.random.normal(0, np.pi, (3)) 

    optimizer = qml.NesterovMomentumOptimizer(stepsize=0.4)
    stepNumber = 100

    for i in range(stepNumber):
        params = optimizer.step(cost,params)
        print("Iteration " + str(i) + " Value: " + str(cost(params)))

    print("Final value: " + str(cost(params)))
    print("Exact value: " + str(getExactValue()))

    drawer = qml.draw(circuit_X)
    print(drawer(params))
    drawer = qml.draw(circuit_Y)
    print(drawer(params))
    drawer = qml.draw(circuit_Z)
    print(drawer(params))

