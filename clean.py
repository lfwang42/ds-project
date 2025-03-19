import datefinder
import numpy as np
import json
import pandas

df = pandas.read_csv('new_data.csv')
print(df.columns)
df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
print(len(df))

df = df[df['File Number'].str.len().gt(5)]
# df.drop(df[df['File Number'].map(lambda x: len(str(x)) < 20)], inplace=True, axis=0)
print(len(df))

# for i, case in df.iterrows():

#     print(df.loc[i, 'File Number'])
print(df.columns)