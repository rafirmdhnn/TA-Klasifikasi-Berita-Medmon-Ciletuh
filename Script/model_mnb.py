# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:33:21 2021

@author: rafir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultiNayes:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.fitted = False

#TRAIN MODEL KLASIFIKASI MNB
    #fungsi untuk mengubah label/kelas kedalam bentuk biner
    def label_array(self, y, classes=None, label_bin=None):
        if classes is None:
            classes = np.unique(y)
            label_bin = np.zeros((y.shape[0], classes.shape[0]))
            self.classes = classes
            self.label_bin = label_bin
            
        if label_bin.shape[0] < 1:
            return None

        x = np.where(classes == y[0])
        label_bin[0][x] = 1

        return self.label_array(y[1:], classes, label_bin[1:])

    def fit(self, X, y):
        #membentuk array biner untuk menyimpan label setiap data
        self.label_array(y)
        
        #menghitung jumlah label/kelas yang terdapat pada dataset
        self.n_classes = self.classes.shape[0]
        #menghitung jumlah fitur yang terdapat pada dataset
        self.n_features = X.shape[1]
        
        #array kosong untuk menyimpan hasil perhitungan jumlah label 
        self.class_count = np.zeros(self.n_classes)
        #array kosong untuk menyimpan hasil perhitungan jumlah fitur/term pada setiap dokumen data train
        self.feature_count = np.zeros((self.n_classes, self.n_features))
        
        #menghitung jumlah kemunculan (frekuensi) fitur pada kedua label dari seluruh data train (count(tn,a))
        self.feature_count += np.dot(self.label_bin.T, X)
        #menghitung jumlah dari tiap label (N(a))
        self.class_count += self.label_bin.sum(axis=0)

        # add smoothing
        self.feature_count += self.alpha
        # untuk menghitung nilai (Count(a) + |V|)
        self.smoothed_class_count = self.feature_count.sum(axis=1)

        # hitung cond prob  P(tn|a) = Count(tn,a) + 1 / Count(a) + |V| <=> P(tn|a) = log(Count(tn,a) + 1) - log(Count(a) + |V|)
        # perhitungan dilakukan dengan menggunakan log untuk menghindari terjadinya 
        # floating point underflow kondisi angka dibelakang koma yang melebihi batas yg ditampung atau diproses komputer
        self.feat_log_probs = (np.log(self.feature_count) -
                           np.log(self.smoothed_class_count.reshape(-1, 1)))

        # hitung prob prior  P(a) = N(a)/N <=> P log2(N(a)) - log2(N)
        self.class_log_priors = (np.log(self.class_count) - np.log(self.class_count.sum(axis=0).reshape(-1, 1)))
        
        #set classifier fitted = true
        self.fitted = True

##TESTING MODEL KLASIFIKASI 
    def predict(self, X):
        #hitung MAP (Maximum a Posteriori) dengan input data test, Cmap = P(a) x P(tn|a) <=> Cmap = log(P(a)) + log(P(tn|a))
        #fungsi dot dilakukan untuk mempermudah perhitungan cond prob dengan input data testing
        self.scores = np.dot(X, self.feat_log_probs.T) + self.class_log_priors
        #menentukan prediksi berdasarkan nilai MAP tertinggi
        self.predictions = self.classes[np.argmax(self.scores, axis=1)]

        return self.predictions

    def accuracy(self, y_pred, y):
        #menghitung nilai akurasi dari hasil prediksi yang diperoleh
        self.points = (y_pred == y).astype(int)
        self.akurasi = self.points.sum() / self.points.shape[0]
        return self.akurasi
    
    # Confusion Matrix
    def confusion_matrix(self, y_actual, y_pred):
        self.cm_mn = np.zeros([2,2])
        
        for i in range (len(y_pred)):
            #TN
            if y_actual[i] == y_pred[i] == 1:
                self.cm_mn[1][1] += 1
            #FN
            if y_pred[i] == 1 and y_actual[i]!= y_pred[i]:
                self.cm_mn[0][1] += 1
            #TP
            if y_actual[i] == y_pred[i] == 0:
                self.cm_mn[0][0] += 1
            #FP
            if y_pred[i] == 0 and y_actual[i]!= y_pred[i]:
                self.cm_mn[1][0] += 1
        
        #Buat Plot untuk hasil confusion matrix
        plt.clf()
        plt.imshow(self.cm_mn)
        classNames = ['Hard News','Soft News']
        plt.title('Hard or Soft News Confusion Matrix - MNB')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TP','FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(self.cm_mn[i][j]), color='r')
        plt.show()
    
        return(self.cm_mn)
    
    def evaluasi(self):
        #menentukan TP,TN,FP,FN berdasarkan nilai confussion matrix
        self.TP = self.cm_mn[0][0]
        self.TN = self.cm_mn[1][1]
        self.FP = self.cm_mn[1][0]
        self.FN = self.cm_mn[0][1]
        #Menghitung nilai recall, precision, f1-score
        self.recall = self.TP / (self.TP + self.FN)
        self.precision = self.TP / (self.TP + self.FP)
        self.F1score = 2 * (self.recall * self.precision) / (self.recall + self.precision)
        print('Recall: ', "%.2f" % self.recall)
        print('Precision: ', "%.2f" % self.precision)
        print('F1-Score: ', "%.2f" % self.F1score)

#%%
#Run Model Multinomial Naive Bayes

#list dataset yg digunakan:
#single dimensional tanpa pemilihan fitur -> single-dataset.csv
#single dimensional dengan IG (0.05) -> DtIG(0.05)-dataset.csv
#single dimensional dengan IG (0.01) -> DtIG(0.01)-dataset.csv
#single dimensional dengan CHI (0.01) -> chiSqr(0.01)-dataset.csv
#single dimensional dengan CHI (0.01) -> chiSqr(0.05)-dataset.csv
#multidimensional -> multidimensional-dataset.csv

#Mempersiapkan dataset yang akan digunakan
file = open(r'C:\Sidang Akhir\Sidang\Dataset\DtIG(0.01)-dataset.csv','r')
#untuk dataset multidimensional ubah delimiter menjadi ';'
df = pd.read_csv(file, delimiter=',')

#Nilai untuk memperoleh fitur sebagai variabel X dan label sebagai variabel Y:
#noFeature: X = [:, 1:30] , Y = [:, 30]
#infogain (0.01): X = [;, 1:26] , Y = [:, 26]
#infogain (0.05): X = [:, 1:17] , Y = [:, 17]
#chisquare (0.05): X = [:, 1:17] , Y = [:, 17]
#chisquare (0.01): X = [:, 1:24] , Y = [:, 24]
#multidimension: X = [:, 1:6] , Y = [:, 6]
X = df.iloc[:, 1:26]
y = df.iloc[:, 26].values

# Membagi dataset ke dalam Training dan Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.90, random_state = 0)

#training model klasifikasi MNB
clf = MultiNayes()
clf.fit(X_train, y_train)
#jalankan model untuk melakukan prediksi
#prediksi data test
y_pred_test = clf.predict(X_test)
#jalankan fungsi untuk memperoleh nilai akurasi
akurasi_test = clf.accuracy(y_pred_test, y_test)
print("Nilai akurasi sebesar :", "%.2f" % (akurasi_test * 100), "pada test set")
#jalankan fungsi confusion matrix dengan input y_test dan y_pred
cm_train = clf.confusion_matrix(y_test, y_pred_test)
#jalankan fungsi untuk melakukan evaluasi berdasarkan nilai precision, recall, dan f1-score
metricScore_test = clf.evaluasi()

#save hasil prediksi kedalam file csv
resPred = []
for i in range(len(y_pred_test)):
    # if(X_test.index[i] == raw['dokumen'])
    resPred.append([X_test.index[i] + 1,y_pred_test[i], y_test[i]])
pred_df = pd.DataFrame(resPred, index=None) 
pred_df.columns = ['dokumen','Prediksi', 'Actual']

pred_df.to_csv(r'C:\Sidang Akhir\Sidang\Hasil\MNB\IG(0.01)-90.csv', index=None)