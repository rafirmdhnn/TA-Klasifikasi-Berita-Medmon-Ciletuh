# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:04:51 2021

@author: rafir
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from pandas import DataFrame


file = open(r'C:\Sidang Akhir\Sidang\Dataset\single-dataset.csv','r')
df = pd.read_csv(file, delimiter = ';',  encoding='cp1252')
X = df.iloc[:, 1:30]
y = df.iloc[:, 30]

query = df.iloc[:, 1:30].columns

chi_score = chi2(X, y)
print(chi_score)

result = []
for i in range(len(query)):
    chi = chi_score[1][i]
    fitur = query[i]
    result.append((fitur, chi))

hasil = DataFrame(result)
hasil.columns = ['Fitur', 'chi_sqr']
hasil.index = hasil.index + 1
hasil.sort_values("chi_sqr", ascending = False , inplace = True)
hasil.to_csv(r'C:\Sidang Akhir\Sidang\Hasil\outputChisqr.csv', index=None)

# set nilai threshold
threshold =  0.01
# seleksi berdasarkan nilai p.value sesuai threshold yg ditentukan
waste = hasil[(hasil.chi_sqr <= threshold) | (hasil.chi_sqr.isnull())]
wastedFeature = list(waste['Fitur'])
print(wastedFeature)
# delete column atau fitur dari data sesuai hasil yg telah diseleksi
goodFeature = df.drop(wastedFeature, axis=1)
# save data to csv
pathCsv = r'C:\Sidang Akhir\Sidang\Dataset\DtchiSqr({})-dataset.csv'.format(threshold)
goodFeature.to_csv(pathCsv, index = None)

p_values = pd.Series(chi_score[1], index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()



