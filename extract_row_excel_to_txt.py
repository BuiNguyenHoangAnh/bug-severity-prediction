
import pandas as pd 

df=pd.read_csv("Book1.csv",sep=",")

for index in range(len(df)):
     with open(df["valueID"][index] +  '.txt', 'w') as output:
        output.write(df["note"][index])