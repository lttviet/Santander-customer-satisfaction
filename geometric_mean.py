import numpy as np
import pandas as pd
from scipy.stats import gmean

df1 = pd.read_csv("1.csv", header=0, index_col="ID")
df2 = pd.read_csv("2.csv", header=0, index_col="ID")
df3 = pd.read_csv("3.csv", header=0, index_col="ID")
#df4 = pd.read_csv("test_logmodel.csv", header=0, index_col="ID")
#df5 = pd.read_csv("test_rfc.csv", header=0, index_col="ID")
#df6 = pd.read_csv("test_xgbc.csv", header=0, index_col="ID")
df7 = pd.read_csv("test_xgbc2.csv", header=0, index_col="ID")

df = df1.rename(columns={"TARGET":"TARGET1"})
df["TARGET2"] = df2
df["TARGET3"] = df3
#df["TARGET4"] = df4
#df["TARGET5"] = df5
#df["TARGET6"] = df6
df["TARGET7"] = df7

geo_mean = pd.DataFrame(gmean(df, axis=1), index=df.index)

submission = pd.DataFrame(index=df.index)
submission["TARGET"] = geo_mean[0]
submission.to_csv("geo_mean.csv")
