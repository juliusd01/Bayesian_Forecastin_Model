import pandas as pd

file = "/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/Resources_all/Economist_2021/polls_btw_wide.dta"

df = pd.read_stata(file)
df.to_csv("/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/Resources_all/Economist_2021/polls_btw_wide.csv", index=False)