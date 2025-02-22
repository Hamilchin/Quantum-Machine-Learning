import pennylane as pl
import pandas as pd
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
#Importing required packages


device = pl.device("default.qubit", wires = 6)
#Creating 6 wire quantum circuit


def stateinit(bitlist):
    '''
    Parameters
    ----------
    bitlist : a list of 6 binary values, corresponding to the presence of covid symptoms.
    ["cough", "fever","sore_throat","shortness_of_breath","head_ache","gender","corona_result"]
    
    Function
    ----------
    Initializes starting qubit state: all 1s encoded as |1> and vice versa
    '''
    pl.BasisState(bitlist, wires=[0, 1, 2, 3, 4, 5])


def layer(weights):
    '''
    Parameters
    ----------
    weights : a 4x6 numpy matrix representing the weights in the layer of the quantum circuit

    Function
    -------
    Applies a layer of the neural network given a set of that layer's weights, entangles with CNOT gates
    '''
    
    pl.Rot(weights[0, 0], weights[0, 1], weights[0, 2], wires=0)
    pl.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=1)
    pl.Rot(weights[2, 0], weights[2, 1], weights[2, 2], wires=2)
    pl.Rot(weights[3, 0], weights[3, 1], weights[3, 2], wires=3)
    pl.Rot(weights[4, 0], weights[4, 1], weights[4, 2], wires=4)
    pl.Rot(weights[5, 0], weights[5, 1], weights[5, 2], wires=5)
    #Performs a series of rotations (Z, Y, Z) on each wire, defined by the weight values
    
    pl.CNOT(wires=[0, 1])
    pl.CNOT(wires=[1, 2])
    pl.CNOT(wires=[2, 3])
    pl.CNOT(wires=[3, 4])
    pl.CNOT(wires=[4, 5])
    pl.CNOT(wires=[5, 0])
    #Entangles ajacent qubtis by applying a CNOT gate
    

@pl.qnode(device)
def nnsimulation(inputs, weightlist):
    '''
    Parameters
    ----------
    inputs : the inputs (bitlist) to the neural network - corresponding to the symptom presence
    weightlist : list of numpy arrays each corresponding to the weights of a single layer of the neural network: 
    length defines number of layers

    Function
    -------
    Initializes then simulates one runthrough of the quantum neural network

    Returns
    -------
    expval : the expectation value for qubit 0 in the Z basis - somewhere between -1 (|0>) and 1 (|1>) 

    '''
    stateinit(inputs)

    for weights in weightlist:
        #Applies a layer for each set of weights in the weightlist
        layer(weights)

    expval = pl.expval(pl.PauliZ(0))
    return expval
    #Returns the expectation value for qubit 0 in the Z basis - somewhere between -1 (|0>) and 1 (|1>)


def neural_network(inputs, weightlist, bias):
    '''
    Parameters
    ----------
    inputs : the inputs (bitlist) to the neural network - corresponding to the symptom presence
    weightlist : list of numpy arrays each corresponding to the weights of a single layer of the neural network: 
    length defines number of layers
    bias : float postprocessing bias added to output

    Function
    -------
    Initializes then simulates one runthrough of the quantum neural network then adds bias to result

    Returns
    -------
    pred_result : the prediction result: somewhere between -1 (|0>) and 1 (|1>) with bias added
    '''
    pred_result = nnsimulation(inputs, weightlist) + bias
    return pred_result


def avg_loss(labels, predictions):
    '''
    Parameters
    ----------
    labels : list of authroritative labels
    predictions : list of neural network predictions for given data

    Returns
    -------
    loss : average loss (mean square) over all sets of labels and predictions

    '''
    totalloss = 0
    
    for label, prediction in zip(labels, predictions):
        totalloss += (label - prediction) ** 2
    #Iterates over all sets of labels and predictions, calculates loss squared, adds to total loss
    
    loss = totalloss / len(labels)
    #calculates average
    
    return loss


def avg_accuracy(labels, predictions):
    '''
    Parameters
    ----------
    labels : list of authroritative labels
    predictions : list of neural network predictions for given data

    Returns
    -------
    accuracy : average accuracy (counts as accurate if prediction is within threshold value) over all sets of labels and predictions
    '''
    
    totalacc = 0
    threshold = 0.2
    for label, prediction in zip(labels, predictions):
        if abs(label-prediction) <= threshold:
            totalacc += 1
    #Iterates over all sets of labels and predictions
    
    accuracy = totalacc / len(labels)
    #calculates average
    
    return accuracy


def cost(weights, bias, inputs, labels):
    '''
    Parameters
    ----------
    weights : list of numpy array represented weights for each layer
    bias : postprocessing bias
    inputs : a list of bitlists that encodes input data
    labels : list of labels for those input data: same length as inputs

    Returns
    -------
    cost : total average cost over all training samples for this set of weights and bias

    '''
    predictions = [neural_network(inp, weights, bias) for inp in inputs]
    #Generates a set of neural network predictions for the set of inputs
    
    cost = avg_loss(labels, predictions)
    #Calculates average loss of neural network predictions
    return cost


def train_eval(weights, bias, inputs, labels):
    '''
    Parameters
    ----------
    weights : list of numpy array represented weights for each layer
    bias : postprocessing bias
    inputs : a list of bitlists that encodes input data
    labels : list of labels for those input data: same length as inputs

    Returns
    -------
    cost : total average cost over all training samples for this set of weights and bias
    accuracy : total average accuracy

    '''
    predictions = [neural_network(inp, weights, bias) for inp in inputs]
    #Generates a set of neural network predictions for the set of inputs
    
    accuracy = avg_accuracy(labels, predictions)
    cost = avg_loss(labels, predictions)
    #Calculates average loss of neural network predictions
    return (cost, accuracy)



def extract_karnaugh_pickle(file_name):
    '''
    Parameters
    ----------
    file_name : string name of the pkl file encoded with six variable karnaugh map

    Returns
    -------
    inputs : numpy array of every set of inputs
    labels : numpy array of their respective labeled outputs

    '''
    
    #Reads DataFrame
    karnaugh = pd.read_pickle(file_name)

    gray = ['000', '001', '011', '010', '110', '111', '101', '100']
    data = []
    
    #Creates a numpy array of input data and output (labeled) data from karnaugh map
    for f3 in gray:
        for l3 in gray:
            lst = [int(bit) for bit in (f3+l3)]
            lst.append(karnaugh.at[f3, l3])
            data.append(lst)
    
    data = np.array(data)
    
    #Separates into inputs and labels
    inputs = np.array(data[:, :-1], requires_grad=False)
    labels = np.array(data[:, -1], requires_grad=False)
    #Shifts labels from {0, 1} to {-1, 1}
    labels = labels * 2 - 1
    
    return (inputs, labels)


def prediction_karnaugh(weights, bias, inputs, labels):
    '''
    Parameters
    ----------
    weights : list of numpy array represented weights for each layer
    bias : postprocessing bias
    inputs : a list of bitlists that encodes input data
    labels : list of labels for those input data: same length as inputs
    
    Returns
    -------
    inputs : numpy array of every set of inputs
    labels : numpy array of their respective labeled outputs
    '''
    
    gray = ['000', '001', '011', '010', '110', '111', '101', '100']
    predictiondf = pd.DataFrame(0, columns = gray, index = gray)
    
    
    predictions = [neural_network(inp, weights, bias) for inp in inputs]
    #Generates a set of neural network predictions for the set of inputs
    
    indx = 0
    for f3 in gray:
        for l3 in gray:
            #Aterates through all predictions and adds them to the df map
            predictiondf.loc[f3, l3] = float(predictions[indx])
            indx += 1
    
    return predictiondf



def weightinit(layers, qubits, seed=0):
    '''
    Parameters
    ----------
    layers : integer number of layers
    qubits : integer number of qubits
    seed : Starting RNG seed: the default is 0.

    Returns
    -------
    weights : a randomized three dimensional numpy array: a list of weight matrices containing the weights for each layer.
    '''
    
    np.random.seed(seed)
    #Creating a seed for reproducibility (given the same seed, we get the same pseudo-random numbers)
    
    rotation_parameters = 3
    #Number of rotation parameters pl.Rot takes
    
    weights = 0.01 * np.random.randn(layers, qubits, rotation_parameters, requires_grad=True)
    #Numbers are chosen from a Gaussian standard normal distribution then multiplied by 0.01 to reduce magnitude shift
    #Depends on number of layers and qubits
    
    return weights


def biasinit(init=0.0):
    '''
    Parameters
    ----------
    init : starting bias: the default is 0.0.

    Returns
    -------
    bias : numpy array of bias value

    '''
    
    bias = np.array(init, requires_grad=True)
    #Initializes bias to arbitrary numpy array: default 0
    
    return bias



def main():
    optimizer = AdamOptimizer(0.01)
    batch_size = 10
    iterations = 200
    
    weights = weightinit(10,6)
    bias = biasinit()
    
    inputs, labels = extract_karnaugh_pickle("ratios.pkl")    
    
    cost_graph = []
    acc_graph = []
    
    for iteration in range(iterations):
    
        # Update the weights by one optimizer step
        batch_indices = np.random.randint(0, len(labels), batch_size)
        curr_inputs = inputs[batch_indices]
        curr_labels = labels[batch_indices]


        weights, bias, _, _ = optimizer.step(cost, weights, bias, curr_inputs, curr_labels)
    
        eval_data = train_eval(weights, bias, inputs, labels)
        print(f"Iteration: {iteration+1} | Cost: {eval_data[0]} | Accuracy : {eval_data[1]} | Bias: {bias}")
        cost_graph.append(eval_data[0])
        acc_graph.append(eval_data[1])
        
    print(f"Weights: {weights}")
    
    plt.figure()
    plt.plot([x for x in range(iterations)], cost_graph)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Mean Squared Cost")
    plt.title("Cost over 200 Iterations")
    
    plt.figure()
    plt.plot([x for x in range(iterations)], acc_graph, color="purple")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Accuracy")
    plt.title("Accuracy over 200 Iterations")
    
    predictiondf = prediction_karnaugh(weights, bias, inputs, labels)
    return predictiondf



def datatester():
    predictionstorage = []
    maxdata = {}
    for layers in [1,2,3,4,5,6,7,8,9,10]:
        optimizer = AdamOptimizer(0.01)
        batch_size = 10
        iterations = 200
        
        weights = weightinit(layers,6)
        bias = biasinit()
        
        inputs, labels = extract_karnaugh_pickle("ratios.pkl")    
        
        cost_graph = []
        acc_graph = []
        
        for iteration in range(iterations):
        
            # Update the weights by one optimizer step
            batch_indices = np.random.randint(0, len(labels), batch_size)
            curr_inputs = inputs[batch_indices]
            curr_labels = labels[batch_indices]
    
    
            weights, bias, _, _ = optimizer.step(cost, weights, bias, curr_inputs, curr_labels)
        
            eval_data = train_eval(weights, bias, inputs, labels)
            print(f"Layer: {layers} | Iteration: {iteration+1} | Cost: {eval_data[0]} | Accuracy : {eval_data[1]} | Bias: {bias}")
            cost_graph.append(eval_data[0])
            acc_graph.append(eval_data[1])
            
        print(f"Weights: {weights}")
        
        plt.figure()
        plt.plot([x for x in range(iterations)], cost_graph)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Mean Squared Cost")
        plt.title("Cost Function Over Time for {layer} Layer QNN")
    
        plt.figure()
        plt.plot([x for x in range(iterations)], acc_graph, color="purple")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Accuracy")
        plt.title(f"Accuracy Over Time for {layer} Layer QNN")
    
        
        predictiondf = prediction_karnaugh(weights, bias, inputs, labels)
        predictionstorage.append(predictiondf.to_numpy())
        print("MAX:", max(cost_graph))
        maxdata[layers] = (float(min(cost_graph)), max(acc_graph))
        
    return (predictionstorage, maxdata)



if __name__ == "__main__":
    
    predictiondf = np.around((main().to_numpy()+1)/2, decimals = 2)
    labeldf = np.around(pd.read_pickle("ratios.pkl").to_numpy(), decimals=2)
    diffdf = abs(predictiondf - labeldf)

    '''
    x = datatester()
    predictions = x[0]
    maxdata = x[1]
    p1 = predictions[0]
    p2 = predictions[1]
    p3 = predictions[2]
    p4 = predictions[3]
    p5 = predictions[4]
    p6 = predictions[5]
    p7 = predictions[6]
    p3 = predictions[7]
    p9 = predictions[8]
    p10 = predictions[9]
    '''
    




