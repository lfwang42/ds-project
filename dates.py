import datefinder
import numpy as np
import json
import pandas
from datetime import datetime

cases = pandas.read_csv('new_out.csv')
new_cases = cases.copy(deep=True)
k = 0
cases['File Number'] = [''] * 847
df = pandas.read_excel('2010.xlsx')
file_numbers = set(df['File Number'].values)
for i, case in cases.iterrows():

    print(i)
    # k += 1
    # if k > 5:
    #     break

    text =  cases.loc[i, 'Text'].split('Date Issued')[0]
    dates = datefinder.find_dates(text)
    temp = [date for date in dates]
    new_cases.loc[i, 'Decision Date'] = temp[-1].strftime('%Y-%m-%d')
    # print(temp[-1].strftime('%Y-%m-%d'))
    for line in text.split("\n"):
        if 'File Number: ' in line:
            file_num = line.split(':')[-1].strip()
            if file_num[0] != 'L' and file_num[0] != 'E':
                file_num = line.split(':')[1].strip()
            new_cases.loc[i, 'File Number'] = file_num


    # print(new_cases.loc[i, 'File Number'])
    # if new_cases.loc[i, 'File Number'] in file_numbers:
    #     print('found')
    # print(new_cases.loc[i])
    # cases.loc[i, "Test1"] = 'hi'
    # print(cases.loc[i])
# for text in cases.Text.values:
#     for line in text.split("\n"):
#         if 'File Number:' in line:
#             print(line.split(':')[1].strip())
d1 = datetime(2009, 1, 1)
d2 = datetime(2025, 5, 1)
print(len(new_cases))
new_cases = new_cases[(new_cases['Decision Date'] > '2009-01-01') & (new_cases['Decision Date'] < '2025-05-01')]
print('after: ')
print(len(new_cases))
new_cases.to_csv('new_data_2.csv')
print(df.columns)