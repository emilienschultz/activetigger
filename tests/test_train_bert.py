import pandas as pd
from pathlib import Path
from activetigger.functions import train_bert

data =  pd.read_csv("./data_annotation.csv")
data = data[["text","labels"]].dropna()
params_default = {
            "batchsize": 4,
            "gradacc": 1,
            "epochs": 3,
            "lrate": 5e-05,
            "wdecay": 0.01,
            "best": True,
            "eval": 10,
            "gpu": True,
            "adapt": True,
        }
train_bert(Path("../test"),"test",data,"text","labels","camembert/camembert-base",params_default,0.2)