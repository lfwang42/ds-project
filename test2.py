from openai import OpenAI
import numpy as np
import json
import pandas
import time
import os
from dotenv import load_dotenv
df = pandas.read_csv('ruling.csv')
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0.2'])
# print(df)
print(len(df))
df['Ruling'].replace('', np.nan, inplace=True)
df.dropna(subset=['Ruling'], inplace=True)
df['Application Type'].replace('', np.nan, inplace=True)
df.dropna(subset=['Application Type'], inplace=True)
df['File Number'].replace('', np.nan, inplace=True)
df.dropna(subset=['File Number'], inplace=True)
df['Tenant-Caused Damage'].replace('', np.nan, inplace=True)
df.dropna(subset=['Tenant-Caused Damage'], inplace=True)
print(len(df))
df.to_csv('ruling.csv')
# print(len(res))
