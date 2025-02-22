
from pennylane import *
import numpy as np
from math import sqrt
from IPython.display import display
import pandas as pd


df = pd.read_csv("smalldata.csv")
df = df[["cough", "fever","sore_throat","shortness_of_breath","head_ache","gender","corona_result"]]
df["gender"] = [0 if x == "female" else 1 for x in df["gender"]]
df["corona_result"] = [0 if x == "negative" else 1 for x in df["corona_result"]]
display(df.head)





dev = device("default.qubit", wires = 2)

@qnode(dev)
def circuit(datum):
    if datum["cough"]:
        PauliX(wires = [0])
    if datum["fever"]:
        PauliY(wires = [0])
    if datum["sore_throat"]:
        PauliZ(wires = [1])
    if datum["shortness_of_breath"]:
        PauliX(wires = [1])
    if datum["head_ache"]:
        PauliY(wires = [1])
    if datum["gender"]:
        PauliZ(wires = [1])

    return probs(wires = [1,0])

for i in range(5):
    print(circuit(df.iloc[i]))
    print(dev.state)
    
