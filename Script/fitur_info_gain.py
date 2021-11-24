# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 21:09:26 2021

@author: rafir
"""

import pandas as pd
from pandas import DataFrame
from collections import Counter
import numpy as np


class infoGain(object):
    
    def __init__(self, data):
        self.data = data
        
    def countEntropyP(self):
        #mengumpulkan seluruh label/kelas dari dataset kedalam list
        self.label = self.data['label'].tolist()
        #menghitung jumlah label/kelas
        self.numClass = Counter(self.label)
        self.total = len(self.label)
        
        #menghitung nilai entropi parent
        self.entropiParent = -sum(count/self.total * np.log2(count/self.total) for count in self.numClass.values())
        print('Parent entropy:', self.entropiParent)

    def countIG(self):
        self.fitur = self.data.iloc[:, 1:30].columns.tolist()
        self.fiturIG = []
        
        #melakukan perhitungan nilai entropi setiap fitur dan nilai IG setiap fitur
        for i in self.fitur:
            self.classEntropy = 0
            self.entropiAtt = []
            self.totalAtt = []
            self.getAtt = []
            #kondisi looping untuk memperoleh seluruh atribut berdasarkan masing2 label pada fitur(i)
            for c, att in self.data.groupby([i]):
                #menyimpan atribut(c) pada fitur[i] pada list getAtt
                self.getAtt.append(c)
                #menghitung jumlah nilai/atribut(c) dari keseluruhan label
                self.attTotal = att.count().values[0]
                #menghitung jumlah nilai/atribut(c) dari tiap label 0 dan 1
                self.p = Counter(att['label'].tolist())
                #menyimpan jumlah nilai/atribut(c) dari tiap label 0 dan 1 kedalam list
                self.totalAtt.append(self.p)
                #list untuk menampung entropi dari seluruh nilai/atribut pada fitur[i]
                self.value = []
                #menghitung entropi masing2 nilai/atribut dari kedua label 0 dan 1 dari fitur[i]
                for v in self.p.values():
                    self.value.append(-v/self.attTotal * np.log2(v/self.attTotal))
                #menghitung nilai entropi kelas dari setiap fitur
                self.classEntropy += self.attTotal/self.data.count().values[0] * sum(self.value)
                self.entropiAtt.append(self.value)
            #menghitung nilai information gain fitur[i]
            self.infoGain = (self.entropiParent - self.classEntropy)
            self.fiturIG.append([i, self.entropiParent, self.classEntropy, self.infoGain])
    
    #fungsi untuk menyimpan hasil kedalam file csv
    def saveCsv(self, threshold, outIg, dtIg):
        self.outIg = outIg
        self.dtIg = dtIg
        
        # membuat dataframe
        self.df = DataFrame(self.fiturIG)
        self.df.columns = ['Fitur', 'Parent Entropy', 'Class Entropy', 'IG']
        self.df.index = self.df.index + 1
        self.df.sort_values("IG", ascending = False , inplace = True)
        
        self.df.to_csv(self.outIg, index=None)
        
        # get nilai threshold
        self.threshold = threshold
        # seleksi berdasarkan nilai IG sesuai threshold yg ditentukan
        self.result = self.df[self.df.IG <= self.threshold]
        # memisahkan fitur yang memiliki nilai IG <= nilai threshold
        self.wastedFeature = list(self.result['Fitur'])
        # delete column atau fitur dari data sesuai hasil yg telah diseleksi
        self.goodFeature = self.data.drop(self.wastedFeature, axis=1)
        # save data to csv
        self.goodFeature.to_csv(self.dtIg, index = None)
        print(self.wastedFeature)
        
#%%

#import dataset
file = open(r'C:\Sidang Akhir\Sidang\Dataset\single-dataset.csv','r')
dataset = pd.read_csv(file, delimiter = ';',  encoding='cp1252')

#set nilai threshold, dan path untuk menyimpan file csv output
threshold = 0.01
#path untuk menyimpan file hasil perhitungan IG keseluruhan
#pathOutIg = r'C:\Sidang Akhir\TA-Klasifikasi-Berita-Medmon-Ciletuh\Dataset\OutIG({})-result.csv'.format(threshold)

pathOutIg = r'C:\Sidang Akhir\Sidang\Hasil\OutIG({})-result.csv'.format(threshold)

#path untuk menyimpan file dataset setelah dilakukan seleksi fitur yang kurang relevan
#pathDtIg = r'C:\Sidang Akhir\TA-Klasifikasi-Berita-Medmon-Ciletuh\Dataset\DtIG({})-dataset.csv'.format(threshold)

pathDtIg = r'C:\Sidang Akhir\Sidang\Dataset\DtIG({})-dataset.csv'.format(threshold)

#initiallize object yang akan digunakan
ig = infoGain(dataset)
#jalankan fungsi untuk menghitung entropi Parent
entropi = ig.countEntropyP()
#jalankan fungsi untuk menghitung entropi class dan information gain dari setiao fitur
out_ig = ig.countIG()
#jalankan fungsi untuk menyimpan hasil pemilihan fitur kedalam csv
ig.saveCsv(threshold,pathOutIg, pathDtIg)