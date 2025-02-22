from IPython.display import display
import pandas as pd


df = pd.read_csv("testing.csv")
df = df[["cough", "fever","sore_throat","shortness_of_breath","head_ache","gender","corona_result"]]
df["gender"] = [0 if x == "female" else 1 for x in df["gender"]]
df["corona_result"] = [0 if x == "negative" else 1 for x in df["corona_result"]]

for column in df.columns:
    df = df[df[f"{column}"] != "None"]
    df[f"{column}"] = df.astype({f"{column}":str}, errors = "raise")[f"{column}"]

gray = ['000', '001', '011', '010', '110', '111', '101', '100']
karnaugh = pd.DataFrame(0, columns = gray, index = gray)
karnaugh2 = pd.DataFrame(0, columns = gray, index = gray)
df = df.reset_index(drop = True)

for indx in df.index:
    #print(indx/df.index[-1] * 100)
    datum = df.iloc[indx]
    if int(datum["corona_result"]):
        karnaugh.at["".join(datum[:3]), "".join(datum[3:6])] += 1
    else:
        karnaugh2.at["".join(datum[:3]), "".join(datum[3:6])] += 1

totals = karnaugh + karnaugh2
ratios = karnaugh/totals

for column in ratios.columns:
    ratios[f"{column}"] = [0 if ratio <= 0.95 else 1 for ratio in ratios[f"{column}"]]


'''
ratios.to_pickle("ratios.pkl")
karnaugh.to_pickle("positives.pkl")
totals.to_pickle("totals.pkl")
'''