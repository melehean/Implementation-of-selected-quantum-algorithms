import pennylane as qml
from pennylane import numpy as np

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

#data = datasets.load_iris()
data = datasets.load_wine()
X = data.data
Y = data.target
qubits_number = len(X[0])
device = qml.device("default.qubit", wires=qubits_number)


def prepare_layer(weights, r):
    for i in range(qubits_number):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
    
    for i in range(qubits_number):
        qml.CNOT(wires=[i, (i+r)%qubits_number])


def prepare_angles(_input):
    qml.BasisState(np.zeros(qubits_number), wires=range(qubits_number))
    for index, value in enumerate(_input):
        qml.RX(value, wires=index)


@qml.qnode(device)
def circuit(weights, shift_array, batch):

    prepare_angles(batch)

    for index, layer in enumerate(weights):
        prepare_layer(layer, shift_array[index])

    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]


def qnn(variables, shift_array, batch):
    weights = variables[0]
    bias = variables[1]
    return circuit(weights, shift_array, batch) + bias


def count_loss(predictions, expected_results):
    loss_sum = 0
    for prediction, expected_result in zip(predictions, expected_results):
        single_loss = 0
        prediction = (prediction + 1.0)/2.0
        single_loss += (prediction[expected_result] - 1) ** 2
        single_loss += (prediction[(expected_result+1)%3]) ** 2
        single_loss += (prediction[(expected_result-1)%3]) ** 2
        loss_sum += single_loss
    return loss_sum/len(predictions)


def loss(variables, shift_array, batch, expected_results):
    predictions = np.array([qnn(variables, shift_array, x) for x in batch])
    return count_loss(predictions, expected_results)


def devide_into_batches(array, batch_size):
    batch_array= []
    help_array = []
    batch_index = 0
    for element in array:
       help_array.append(element) 
       batch_index+=1
       if batch_index == batch_size:
           batch_array.append(help_array)
           help_array = []
           batch_index = 0
    if help_array:
        batch_array.append(help_array)
    return batch_array

def change_predictions_to_classes(predictions, threshold):
    classes = []
    for predictions_vector in predictions:
        max_prediction_index = np.argmax(predictions_vector)
        classes.append(max_prediction_index)
    return classes

def accuracy(variables, shift_array, _input, expected_results, threshold):
    predictions = np.array([qnn(variables, shift_array, x) for x in _input])

    predictions = (predictions + 1.0)/2.0
    predictions = change_predictions_to_classes(predictions, threshold)

    accuracy_count = 0
    for prediction, expected_result in zip(predictions, expected_results):
        if prediction == expected_result:
            accuracy_count += 1
    
    return accuracy_count/len(expected_results)

def main():
    global X, Y, qubits_number
    # get only first two classes
    #X = X[Y<=1]
    #Y = Y[Y<=1]

    seed = 0
    np.random.seed(0)

    X = minmax_scale(X, feature_range=(0, np.pi))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, shuffle=True, stratify=Y)
    Y_train_scaled = np.empty_like(Y_train)
    layers_number = 4
    shift_array = [1,2,3,1]
    batch_size = 140
    batches_number = len(X_train)//batch_size
    epochs_number = 10
    weights = 0.01 * np.random.randn(layers_number, qubits_number, 3)
    bias = np.zeros(3)
    variables = [weights,bias]

    optimizer = qml.AdamOptimizer()

    

    for epoch in range(epochs_number):
        print("Epoch:", epoch)
            
        X_shuffled, Y_shuffled = shuffle(X_train, Y_train, random_state=seed)
        X_batches = devide_into_batches(X_shuffled, batch_size)
        Y_batches = devide_into_batches(Y_shuffled, batch_size)
        
        for iteration in range(batches_number):
            batch_cost = lambda help_variables: loss(help_variables, shift_array, X_batches[iteration], Y_batches[iteration])
            variables = optimizer.step(batch_cost, variables)
        
        print("Test cost", loss(variables, shift_array, X_test, Y_test))
        print("Train cost", loss(variables, shift_array, X_train, Y_train))

        accuracy_test_value = accuracy(variables, shift_array, X_test, Y_test, threshold=0.5)
        accuracy_train_value = accuracy(variables, shift_array, X_train, Y_train, threshold=0.5)

        print("Accuracy in train set:", accuracy_train_value)
        print("Accuracy in test set:", accuracy_test_value)

if __name__ == '__main__':
    main()