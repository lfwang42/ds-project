from openai import OpenAI
import numpy as np
import json
import pandas
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
textmap = {}
with open('newdata.json', encoding='utf-8') as f:
    textmap = json.load(f)
f.close()
client = OpenAI(api_key=API_KEY)
old = pandas.read_csv('out.csv')
keys = textmap.keys()
res = []
j = 0
cases = pandas.read_csv('new_data.csv')
for k in textmap:
    # print(k)
    # j += 1
    # print(j)
    # if j > 10:
    #     break
    t = textmap[k]

    text = t[0] if isinstance(t, list) else t
    file_num = ""
    for line in text.split("\n"):
        if 'File Number: ' in line:
            file_num = line.split(':')[-1].strip()
            if len(file_num) == 0 or (file_num[0] != 'L' and file_num[0] != 'E'):
                file_num = line.split(':')[1].strip()
    if not file_num in cases['File Number'].values:
        # print(text)
        # keywords = t[1]
        m = "Between 'Tenant-Caused Damage', 'Tenant-Nonpayment', 'Landlord's failure to maintain', 'Tenant caused serious problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', and 'Other', can you pick the main dispute between the tenant and the landlord in the following text?  You must pick at least one of the reasons, and you can pick multiple if they apply.  Do not attempt to explain your answer and do not include anything other than the previously mentioned disputes.  Here is the text: " + text + '.'
        # print(m)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": m
            }]
        )

        print('first completion: ' + completion.choices[0].message.content)
        # time.sleep(1)

        #'Tenant-Caused Damage', 'Tenant-Nonpayment', 'Landlord's maintainenance', 'Tenant serious problems', 'Tenant Illegal Activity', 'Landlord Familyn', 'Landlord Bad Faith Term', and 'Other'
        text_labels = ['Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord's failure to maintain", 'Tenant caused serious problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', 'Other']
        labels = [0]*10
        for i in range(len(text_labels)):
            if text_labels[i] in completion.choices[0].message.content:
                # print('true')
                labels[i] = 1['Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord's failure to maintain", 'Tenant caused serious problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', 'Other']


        # time.sleep(1)

        m2 = """Between N1: Notice of Rent Increase, N2: Notice of Rent Increase (Unit Partially Exempt), 
        N3: Notice to Increase the Rent and/or Charges for Care Services and Meals, 
        N10: Agreement to Increase the Rent Above the Guideline, 
        N4: Notice to End your Tenancy Early for Non-payment of Rent 
        N5: Notice to End your Tenancy for Interfering with Others, Damage or Overcrowding 
        N6: Notice to End your Tenancy for Illegal Acts or Misrepresenting Income in a Rent-Geared-to-Income Rental Unit 	
        N7: Notice to End your Tenancy for Causing Serious Problems in the Rental Unit or Residential Complex 
        N8: Notice to End your Tenancy at the End of the Term 	
        N11: Agreement to End the Tenancy     
        N12: Notice to End your Tenancy Because the Landlord, a Purchaser or a Family Member Requires the Rental Unit 	
        N13: Notice to End your Tenancy Because the Landlord Wants to Demolish the Rental Unit, Repair it or Convert it to Another Use 	
        L1: Application to evict a tenant for non-payment of rent and to collect rent the tenant owes
        L2: Application to End a Tenancy and Evict a Tenant or Collect Money 	
        L3: Application to End a Tenancy: Tenant Gave Notice or Agreed to Terminate the Tenancy 
        L4: Application to End a Tenancy and Evict a Tenant: Tenant Failed to Meet Conditions of a Settlement or Order 	
        L5: Application for an Above Guideline Increase 	
        L6: Application for Review of a Work Order about Provincial Maintenance Standards 	
        L7: Application to Transfer a Care Home Tenant 	
        L8: Application Because the Tenant Changed the Locks 	
        L9: Application to Collect Rent the Tenant Owes 	
        L10: Application to Collect Money a Former Tenant Owes, which is the application or notice found in the following text?  Do not attempt to explain your answer and reply only with in format [Letter][Number], e.g. N1.  Here is the text: """ + text + '.  '
        # print(m)
        application_completion = client.chat.completions.create(    model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": m2
            }]
        )

        print(application_completion.choices[0].message.content)
        to_app = [text] + labels + [application_completion.choices[0].message.content]
        print('to app:' + to_app[-1])
        res.append(to_app)
    else:
        print("not in")

df = pandas.DataFrame(data=res)
print(len(df))
df.columns = ['Text', 'Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord Maintenance Failure", 'Tenant caused problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', 'Other', 'Application Type']
new_df = pandas.concat([old, df])
df.to_csv('new_out.csv')
# print(df)



p = 0
for k in textmap:
    print(k)
    p += 1
    print(p)
    if p > 10:
        break
    t = textmap[k]

    text = t[0] if isinstance(t, list) else t
    # print(text)
    # keywords = t[1]
    m = """Between N1: Notice of Rent Increase, N2: Notice of Rent Increase (Unit Partially Exempt), 
N3: Notice to Increase the Rent and/or Charges for Care Services and Meals, 
N10: Agreement to Increase the Rent Above the Guideline, 
N4: Notice to End your Tenancy Early for Non-payment of Rent 
N5: Notice to End your Tenancy for Interfering with Others, Damage or Overcrowding 
N6: Notice to End your Tenancy for Illegal Acts or Misrepresenting Income in a Rent-Geared-to-Income Rental Unit 	
N7: Notice to End your Tenancy for Causing Serious Problems in the Rental Unit or Residential Complex 
N8: Notice to End your Tenancy at the End of the Term 	
N11: Agreement to End the Tenancy     
N12: Notice to End your Tenancy Because the Landlord, a Purchaser or a Family Member Requires the Rental Unit 	
N13: Notice to End your Tenancy Because the Landlord Wants to Demolish the Rental Unit, Repair it or Convert it to Another Use 	
L1: Application to evict a tenant for non-payment of rent and to collect rent the tenant owes
L2: Application to End a Tenancy and Evict a Tenant or Collect Money 	
L3: Application to End a Tenancy: Tenant Gave Notice or Agreed to Terminate the Tenancy 
L4: Application to End a Tenancy and Evict a Tenant: Tenant Failed to Meet Conditions of a Settlement or Order 	
L5: Application for an Above Guideline Increase 	
L6: Application for Review of a Work Order about Provincial Maintenance Standards 	
L7: Application to Transfer a Care Home Tenant 	
L8: Application Because the Tenant Changed the Locks 	
L9: Application to Collect Rent the Tenant Owes 	
L10: Application to Collect Money a Former Tenant Owes, which is the application or notice found in the following text?  Do not attempt to explain your answer and reply only with in format [Letter][Number], e.g. N1.  Here is the text: """ + text + '.  '
    # print(m)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": m
        }]
    )
    # print(completion.choices[0].message.content)
    #'Tenant-Caused Damage', 'Tenant-Nonpayment', 'Landlord's maintainenance', 'Tenant serious problems', 'Tenant Illegal Activity', 'Landlord Familyn', 'Landlord Bad Faith Term', and 'Other'
    print(completion.choices[0].message.content)

"""N1: Notice of Rent Increase, N2: Notice of Rent Increase (Unit Partially Exempt), 
N3: Notice to Increase the Rent and/or Charges for Care Services and Meals, 
N10: Agreement to Increase the Rent Above the Guideline, 
N4: Notice to End your Tenancy Early for Non-payment of Rent 
N5: Notice to End your Tenancy for Interfering with Others, Damage or Overcrowding 
N6: Notice to End your Tenancy for Illegal Acts or Misrepresenting Income in a Rent-Geared-to-Income Rental Unit 	
N7: Notice to End your Tenancy for Causing Serious Problems in the Rental Unit or Residential Complex 
N8: Notice to End your Tenancy at the End of the Term 	
N11: Agreement to End the Tenancy     
N12: Notice to End your Tenancy Because the Landlord, a Purchaser or a Family Member Requires the Rental Unit 	
N13: Notice to End your Tenancy Because the Landlord Wants to Demolish the Rental Unit, Repair it or Convert it to Another Use 	
L1: Application to evict a tenant for non-payment of rent and to collect rent the tenant owes
L2: Application to End a Tenancy and Evict a Tenant or Collect Money 	
L3: Application to End a Tenancy: Tenant Gave Notice or Agreed to Terminate the Tenancy 
L4: Application to End a Tenancy and Evict a Tenant: Tenant Failed to Meet Conditions of a Settlement or Order 	
L5: Application for an Above Guideline Increase 	
L6: Application for Review of a Work Order about Provincial Maintenance Standards 	
L7: Application to Transfer a Care Home Tenant 	
L8: Application Because the Tenant Changed the Locks 	
L9: Application to Collect Rent the Tenant Owes 	
L10: Application to Collect Money a Former Tenant Owes"""