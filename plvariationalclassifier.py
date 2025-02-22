import pennylane as pl
import pandas as pd
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
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
def nniteration(inputs, weightlist, bias):
    '''

    Parameters
    ----------
    inputs : the inputs (bitlist) to the neural network - corresponding to the symptom presence
    weightlist : list of numpy arrays each corresponding to the weights of a single layer of the neural network: 
    length defines number of layers

    Returns
    -------

    '''
    stateinit(inputs)

    for weights in weightlist:
        #Applies a layer for each set of weights in the weightlist
        layer(weights)

    return pl.expval(pl.PauliZ(0)) + bias
    #Returns the expectation value for qubit 0 in the Z basis - somewhere between -1 (|0>) and 1 (|1>) and adds bias
    



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


def cost(weights, bias, inputs, labels):
    '''
    

    Parameters
    ----------
    weights : set of weights for each layer
    bias : postprocessing bias
    inputs : the set of bitlists that encodes input data
    labels : labels for those input data: same length as inputs

    Returns
    -------
    cost : total average cost over all training samples for this set of weights and bias

    '''
    predictions = [nniteration(inp, weights, bias) for inp in inputs]
    #Generates a set of neural network predictions for the set of inputs
    
    cost = avg_loss(labels, predictions)
    #Calculates average loss of neural network predictions
    return cost


#DATA PREPROCESSING

karnaugh = pd.read_pickle("ratios.pkl")
gray = ['000', '001', '011', '010', '110', '111', '101', '100']
data = []

#Creates a numpy array of input data and output (labeled) data
for f3 in gray:
    for l3 in gray:
        lst = [int(bit) for bit in (f3+l3)]
        lst.append(karnaugh.at[f3, l3])
        data.append(lst)

data = np.array(data)

#Seperates into inputs and outputs
inputs = np.array(data[:, :-1], requires_grad=False)
labels = np.array(data[:, -1], requires_grad=False)






