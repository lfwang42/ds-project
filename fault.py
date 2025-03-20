from openai import OpenAI
import numpy as np
import json
import pandas
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
client = OpenAI(api_key=API_KEY)
res = []
j = 0
cases = pandas.read_csv('new_data.csv')
for i, row  in cases.iterrows():
    # print(k)
    # j += 1
    # print(j)
    # if j > 100:
    #     break
    text = row['Text']
    m = "Out of 'Tenant', 'Landlord', or 'Neither', can you tell me who the judge ruled in favor of in the following text?  The judge rules in favor of the tenant if they dismiss the Landlord's application (if the Landlord filed one) or grant the tenant significantly more money/damages than the landlord, rules in favor of the landlord if they dismiss the tenant's application (if the tenant filed one), grant the Landlord significantly more money/damages than the landlord, or evict the tenant.  The judge rules in favour of neither if the balance of money/damages granted to both is roughly equivalent and neither party has their application dismissed.  Do not attempt to explain your answer.  Only include one of the previously mentioned answers.  Here is the text: " + text + '.'
    # print(m)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": m
        }]
    )

    print(str(i) + ': ' + completion.choices[0].message.content)
    res.append(completion.choices[0].message.content)
    # time.sleep(1)

    #'Tenant-Caused Damage', 'Tenant-Nonpayment', 'Landlord's maintainenance', 'Tenant serious problems', 'Tenant Illegal Activity', 'Landlord Familyn', 'Landlord Bad Faith Term', and 'Other'
    # text_labels = ['Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord's failure to maintain", 'Tenant caused serious problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', 'Other']
    # labels = [0]*10
    # for i in range(len(text_labels)):
    #     if text_labels[i] in completion.choices[0].message.content:
    #         # print('true')
    #         labels[i] = 1


    # # time.sleep(1)
    # print('to app:' + to_app[-1])
    # res.append(to_app)

cases['Ruling'] = res

cases.to_csv('ruling.csv')