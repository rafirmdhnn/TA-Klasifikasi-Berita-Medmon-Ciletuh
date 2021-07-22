# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:04:51 2021

@author: rafir
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from pandas import DataFrame


file = open(r'C:\\Skripsi\Skripsi\Sidang\Dataset\single-dataset.csv','r')
df = pd.read_csv(file, delimiter = ',',  encoding='cp1252')
X = df.iloc[:, 1:30]
y = df.iloc[:, 30]

query = df.iloc[:, 1:30].columns

chi_score = chi2(X, y)
print(chi_score)

result = []
for i in range(len(query)):
    pVal = chi_score[1][i]
    fitur = query[i]
    result.append((fitur, pVal))

hasil = DataFrame(result)
hasil.columns = ['Fitur', 'P_Value']
hasil.index = hasil.index + 1
hasil.sort_values("P_Value", ascending = False , inplace = True)
hasil.to_csv(r'C:\Skripsi\Skripsi\Sidang\Result\outputChisqr.csv', index=None)

# set nilai threshold
threshold =  0.05
# seleksi berdasarkan nilai p.value sesuai threshold yg ditentukan
waste = hasil[(hasil.P_Value <= threshold) | (hasil.P_Value.isnull())]
wastedFeature = list(waste['Fitur'])
# delete column atau fitur dari data sesuai hasil yg telah diseleksi
goodFeature = df.drop(wastedFeature, axis=1)
# save data to csv
pathCsv = r'C:\Skripsi\Skripsi\Sidang\Dataset\chiSqr({})-dataset.csv'.format(threshold)
goodFeature.to_csv(pathCsv, index = None)

p_values = pd.Series(chi_score[1], index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()



