# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:47:46 2021

@author: rafir
"""

import random
import pandas as pd
import numpy as np

class RF():
    #constructor untuk mengambil input setiap nilai hyperparameter
    def __init__(self, X_train, y_train, n_tree, max_fitur, max_depth, min_samples_split):
        self.X_train = X_train
        self.y_train = y_train
        self.n_tree = n_tree
        self.max_fitur = max_fitur
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

##TRAIN MODEL KLASIFIKASI RF   
    #fungsi untuk menghitung entropi dari parent node, left_child node, dan right_child node
    def entropi(self, l):
        if l == 0:
            return 0
        elif l == 1:
            return 0
        else:
            return - (l * np.log2(l) + (1 - l) * np.log2(1-l))
    
    #fungsi untuk menghitung nilai IG untuk menentukan parent(root) node dan child(inner) node
    def info_gain(self, left_child, right_child):
        parent = left_child + right_child
        #perhitungan probabilitas dari setiap node untuk digunakan pada perhitungan entropi
        p_parent = parent.count(0) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(0) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(0) / len(right_child) if len(right_child) > 0 else 0
        #pemanggilan fungsi untuk menghitung nilai entropi
        IG_p = self.entropi(p_parent)
        IG_l = self.entropi(p_left)
        IG_r = self.entropi(p_right)
        #perhitungan nilai information gain untuk setiap split
        return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r
    
    def sampling_bootstrap(self, X_train, y_train):
        #pengambilan sampel bootstrap dengan pengembalian (replacement)
        idx_bootstrap = list(np.random.choice(range(len(X_train)), len(y_train), replace = True))
        idx_oob = [i for i in range(len(X_train)) if i not in idx_bootstrap]
        #X_bootstrap berisi seluruh nilai/atribut dari setiap fitur pada datatrain
        X_bootstrap = X_train.iloc[idx_bootstrap].values
        #y_bootstrap berisi seluruh label 
        y_bootstrap = y_train[idx_bootstrap]
        #X_oob dan Y_oob merupakan data yang tidak terpilih saat proses pengambilan sampel bootstrap
        X_oob = X_train.iloc[idx_oob].values
        y_oob = y_train[idx_oob]
        return X_bootstrap, y_bootstrap, X_oob, y_oob
    
    #fungsi untuk mendapatkan nilai estimasi error yang diperoleh sampel OOB
    def estimate_oob(self, tree, X_test, y_test):
        missed_label = 0
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            #cek jika prediksi tidak sesuai dengan aktual class
            if pred != y_test[i]:
                missed_label += 1
        return missed_label / len(X_test)
    
    #fungsi untuk menentukan parent(root) node dan child(inner) node
    def get_split_point(self, X_bootstrap, y_bootstrap, max_fitur):
        ls_fitur = list()
        num_fitur = len(X_bootstrap[0])
        #cek jika kondisi jumlah fitur yg dipilih kurang dari max_fitur maka jalankan
        while len(ls_fitur) <= max_fitur:
            # pengambilan fitur secara acak 
            idx_fitur = random.sample(range(num_fitur), 1)
            #cek apabila fitur belum digunakan dalam tree, maka masukan fitur kedalam ls_fitur
            if idx_fitur not in ls_fitur:
                ls_fitur.extend(idx_fitur)
            
            best_info_gain = -999
            node = None
            
            #pada setiap fitur terpilih, iterasi dilakukan untuk memperoleh setiap nilai/atribut 
            #dalam sampel bootstrap untuk memperoleh nilai IG
            for idx_fitur in ls_fitur:
                for split_point in X_bootstrap[:,idx_fitur]:
                    #set list kosong untuk menampung setiap child node (left/right)
                    left_child = {'X_bootstrap': [], 'y_bootstrap': []}
                    right_child = {'X_bootstrap': [], 'y_bootstrap': []}
                
                # split child node (left/right) untuk variable yang bersifat numeric/kontinu
                # cek jika split_point berupa angka/float
                if type(split_point) in [int, float]:
                    for i, value in enumerate(X_bootstrap[:,idx_fitur]):
                        if value <= split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])
                # split child node (left/right) untuk variable yang bersifat kategori/string
                # cek jika split_point berupa string
                else:
                    for i, value in enumerate(X_bootstrap[:,idx_fitur]):
                        if value == split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])
                
                #hitung nilai informatiion gain dari fitur
                split_info_gain = self.info_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
                #cek jika nilai info gain yg diperoleh lebih besar dari nilai info gain 
                #terbaik sebelumnya untuk memperoleh parent/root node 
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    #set dictionary untuk kedua child node
                    #untuk proses pembentukan cabang 
                    left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                    right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                    #satu split/tree yang telah terbentuk disimpan kedalam dictionary node
                    #untuk digunakan pada tahap pembentukan cabang/split selanjutnya
                    node = {'information_gain': split_info_gain,
                            'left_child': left_child,
                            'right_child': right_child,
                            'split_point': split_point,
                            'feature_idx': idx_fitur}
        return node
    
    
    #fungsi untuk menampung leaf node atau prediksi dari setiap tree
    def terminal_node(self, node):
        y_bootstrap = node['y_bootstrap']
        self.pred = max(y_bootstrap, key = y_bootstrap.count)
        return self.pred
    
    #fungsi untuk menentukan pembentukan cabang/split 
    #dapat dilakukan atau dihentikan dalam sebuah tree
    def split_node(self, node, max_fitur, min_samples_split, max_depth, depth):
        #simpan child node dari proses pembentukan cabang/split sebelumnya kedalam variabel baru
        left_child = node['left_child']
        right_child = node['right_child']
        
        #hapus child node yang diperoleh dari proses pembentukan cabang sebelumnya
        del(node['left_child'])
        del(node['right_child'])
        
        #cek jika salah satu child node memiliki dictionary kosong 
        #jika benar maka jalankan fungsi terminal node untuk memperoleh predikisi(leaf node)
        #dan simpan kedalam node left dan right split
        if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
            empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
            node['left_split'] = self.terminal_node(empty_child)
            node['right_split'] = self.terminal_node(empty_child)
            return
        
        #cek jika kondisi kedalaman tree sudah mencapai batas maksimum
        #jika sudah, maka jalankan fungsi terminal node untuk memperoleh predikisi(leaf node)
        #dan simpan kedalam node left dan right_split 
        if depth >= max_depth:
            node['left_split'] = self.terminal_node(left_child)
            node['right_split'] = self.terminal_node(right_child)
            return node
        
        #cek jika kondisi sample bootstrap pada left_child node untuk pembentukan cabang/split baru
        #sudah lebih rendah dari batas minimal maka jalankan fungsi term node 
        #dan simpan prediksi (leaf node) kedalam node left_split
        if len(left_child['X_bootstrap']) <= min_samples_split:
            node['left_split'] = node['right_split'] = self.terminal_node(left_child)
        #jika kondisi belum terpenuhi, jalankan fungsi get_split_point
        #untuk membentuk split/cabang baru sampai kondisi terpenuhi
        else:
            node['left_split'] = self.get_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_fitur)
            self.split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
        if len(right_child['X_bootstrap']) <= min_samples_split:
            node['right_split'] = node['left_split'] = self.terminal_node(right_child)
        else:
            node['right_split'] = self.get_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_fitur)
            self.split_node(node['right_split'], max_fitur, min_samples_split, max_depth, depth + 1)
            
    #fungsi untuk membentuk sebuah decision tree 
    #dengan menjalankan fungsi get_split_point untuk memperoleh parent/root node
    #jalankan fungsi split_node secare rekursif dengan input node hasil dari fungsi get_split_point
    #sampai setiap internal/child node memiliki predikisi (leaf node)
    def build_tree(self, X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_fitur):
        root_node = self.get_split_point(X_bootstrap, y_bootstrap, max_fitur)
        self.split_node(root_node, max_fitur, min_samples_split, max_depth, 1)
        return root_node
    
    #fungsi untuk membentuk sekumpulan decision tree 
    def random_forest(self):
        #set list untuk menyimpan setiap tree yg terbentuk
        self.ls_tree = list()
        #set list untuk menyimpan setiap oob error yang diperoleh pada setiap pembentukan tree
        self.ls_oob = list()
        #bentuk tree sebanyak n_tree
        for i in range(self.n_tree):
            #membentuk sample bootstrap
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.sampling_bootstrap(self.X_train, self.y_train)
            #membentuk satu buah decision tree 
            tree = self.build_tree(X_bootstrap, y_bootstrap, self.max_depth, self.min_samples_split, self.max_fitur)
            #simpan tree yang terbentuk ke dalam ls_tree
            self.ls_tree.append(tree)
            #menghitung nilai error dari sampel oob
            oob_error = self.estimate_oob(tree, X_oob, y_oob)
            #simpan nilai kedalam list
            self.ls_oob.append(oob_error)
        return self.ls_tree
    
    #fungsi untuk memperoleh nilai rata-rata dari keseluruhan nilai estimasi oob error
    def get_mean_OOB(self):
        mean = np.mean(self.ls_oob)
        return mean
    
##TESTING MODEL KLASIFIKASI RF  
    #fungsi untuk memprediksi/mengetest sebuah decision tree dengan input data testing/unseen data
    def predict_tree(self, tree, X_test):
        #get fitur dari train model
        self.idx_fitur = tree['feature_idx']
        
        #cek apabila input berupa numeric/continou
        if X_test[self.idx_fitur] <= tree['split_point']:
            #cek jika left_child/split berupa dictionary
            #maka lakukan proses testing/prediksi lagi 
            if type(tree['left_split']) == dict:
                return self.predict_tree(tree['left_split'], X_test)
            #jika left_child/split bukan berupa dictionary
            #maka return hasil prediksi 
            else:
                self.value = tree['left_split']
                return self.value
        #cek apabila input berupa string/kategori
        else:
            if type(tree['right_split']) == dict:
                return self.predict_tree(tree['right_split'], X_test)
            else:
                return tree['right_split']
            
    
    def predict_rf(self, ls_tree, X_test):
        #set list untuk menyimpan hasil prediksi dari setiap tree
        self.pred_ls = list()
        #lakukan testing/prediksi untuk keseluruhan dokumen pada testing data/unseen data
        for i in range(len(X_test)):
            #lakukan testing/prediksi pada setiap tree yang terbentuk dalam model 
            self.ensemble_preds = [self.predict_tree(tree, X_test.values[i]) for tree in ls_tree]
            #majority voting untuk memperoleh kelas/label prediksi terbanyak dari keseluruhan tree
            self.final_pred = max(self.ensemble_preds, key = self.ensemble_preds.count)
            #simpan hasil mahority voting kedalam list
            self.pred_ls.append(self.final_pred)
        return np.array(self.pred_ls)

#%%
#dataset preparation

#list dataset yg digunakan:
#single dimensional tanpa pemilihan fitur -> single-dataset.csv
#single dimensional dengan IG (0.05) -> DtIG(0.05)-dataset.csv
#single dimensional dengan IG (0.01) -> DtIG(0.01)-dataset.csv
#single dimensional dengan CHI (0.01) -> chiSqr(0.01)-dataset.csv
#single dimensional dengan CHI (0.01) -> chiSqr(0.05)-dataset.csv
#multidimensional -> multidimensional-dataset.csv

file = open(r'C:\Sidang Akhir\Sidang\Dataset\DtIG(0.01)-dataset.csv','r')
#untuk dataset multidimensional ubah delimiter ',' menjadi ';'
df = pd.read_csv(file, delimiter=',')

#Nilai untuk memperoleh fitur sebagai variabel X dan label sebagai variabel Y:
#noFeature: X = [:, 1:30] , Y = [:, 30], fitur = [:, 1:30]
#infogain (0.01): X = [;, 1:26] , Y = [:, 26], fitur = [:, 1:26]
#infogain (0.05): X = [:, 1:17] , Y = [:, 17], fitur = [:, 1:17]
#chisquare (0.05): X = [:, 1:17] , Y = [:, 17], fitur = [:, 1:17]
#chisquare (0.01): X = [:, 1:24] , Y = [:, 24], fitur = [:, 1:24]
#multidimension: X = [:, 1:6] , Y = [:, 6], fitur = [:, 1:6]
X = df.iloc[:, 1:26]
y = df.iloc[:, 26]
fitur = df.iloc[:, 1:26].columns

#set persentase train size yg akan digunakan
nb_train = int(np.floor(0.9 * len(df)))
df = df.sample(frac=1, random_state= 1)
#pembentukan data train dan data test dari input dataset yang digunakan
X_train = df[fitur][:nb_train]
y_train = df['label'][:nb_train].values
X_test = df[fitur][nb_train:]
y_test = df['label'][nb_train:].values


#%%

#run with grid search


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#fungsi untuk membuat confusion matrix dan memperoleh nilai TN, FN, TP, FP
def confusion_matrix(y_actual, y_pred):
    cm_rf = np.zeros([2,2])
    
    for i in range (len(y_pred)):
        #TN
        if y_actual[i] == y_pred[i] == 1:
            cm_rf[1][1] += 1
        #FN
        if y_pred[i] == 1 and y_actual[i]!= y_pred[i]:
            cm_rf[0][1] += 1
        #TP
        if y_actual[i] == y_pred[i] == 0:
            cm_rf[0][0] += 1
        #FP
        if y_pred[i] == 0 and y_actual[i]!= y_pred[i]:
            cm_rf[1][0] += 1

    return(cm_rf)

def metricsEval(cm):
    #get metrics value (TP,TN,FP,FN)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    #hitung nilai recall
    recall = TP / (TP + FN)
    #hitung nilai precision
    precision = TP / (TP + FP)
    #hitung nilai f1-score
    F1score = 2 * (recall * precision) / (recall + precision)
    
    print('Recall: ', recall)
    # precision: tp / (tp + fp)
    print('Precision: ', precision)
    # f1-Score: 2 tp / (2 tp + fp + fn)
    print('F1-Score: ', F1score)


#Hypermarameter Tuning
#set nilai yang akan digunakan
n_tree = [25, 50,100]
max_depth = [5, 10, 15]
min_samples_split = [2, 3, 4]
max_fitur = [2, 3, 5] 

max_acc = 0
#set list untuk menyimpan komnbinasi parameter berdasarkan grid search
parameters = []
#perulangan untuk memperoleh kombinasi dari masing-masing nilai hyperparameter
for i in n_tree:
    for j in max_depth:
        for k in max_fitur:
            for l in min_samples_split:
                #simpan kombinasi hyperparameter kedalam list
                parameters.append((i, j, k, l))
print("Kombinasi yang terbentuk : ",  parameters )

#set list untuk menyimpan hasil dari setiap kombinasi yang digunakan
result = {'n_tree': [], 
          'max_depth': [],
          'max_fitur': [],
          'min_samples_split': [],
          'accuracy': [],
          'precision': [],
          'recall': [],
          'f1-score': [],
          'confusion_matrix': [],
          'mean_oob': [],
          'prediction':[]}
#jalankan train dan test model pada setiap kombinasi hyperparameter yang tersedia
for k in range( len( parameters ) ) :    
    clf = RF(X_train, y_train, n_tree = parameters[k][0], max_fitur = parameters[k][2], max_depth = parameters[k][1], min_samples_split = parameters[k][3])
    model = clf.random_forest()
    oob = clf.get_mean_OOB()
    y_pred = clf.predict_rf(model, X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    
    #Menghitung nilai recall, precision, f1-score
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1score = 2 * (recall * precision) / (recall + precision)
  
    correctly_classified = 0
  
    # counter    
    count = 0
    
    #menghitung nilai akurasi dari kombinasi yang digunakan
    for count in range( np.size( y_pred ) ) :            
        if y_test[count] == y_pred[count] :                
            correctly_classified = correctly_classified + 1   
    curr_accuracy = ( correctly_classified / len(y_test) ) * 100
    
    #untuk memperoleh nilai akurasi terbaik dari seluruh kombinasi hyperparameter yg digunakan
    if max_acc < curr_accuracy :            
        max_acc = curr_accuracy
    #simpan seluruh nilai hyperparameter yg digunakan, nilai oob, akurasi, recall, precision, 
    #f1-score dan confussion matrix yg diperoleh
    result['n_tree'].append(parameters[k][0])
    result['max_depth'].append(parameters[k][1])
    result['max_fitur'].append(parameters[k][2])
    result['min_samples_split'].append(parameters[k][3])
    result['accuracy'].append(curr_accuracy)
    result['precision'].append(precision)
    result['recall'].append(recall)
    result['f1-score'].append(F1score)
    result['confusion_matrix'].append(cm)
    result['mean_oob'].append(oob)
    result['prediction'].append(y_pred)

print( "Akurasi terbaik yang diperoleh : ", max_acc )

#untuk menampilkan hasil dari kombinasi yang memperoleh nilai akurasi terbaik
for i in range(len(result['n_tree'])):
    if result['accuracy'][i] == max_acc:
        best_n_tree = (result['n_tree'][i])
        best_max_fitur = (result['max_fitur'][i])
        best_max_depth = (result['max_depth'][i])
        best_min_samples_split = (result['min_samples_split'][i])
        best_recall = (result['recall'][i])
        best_precision = (result['precision'][i])
        best_f1_score = (result['f1-score'][i])
        best_cm_score = (result['confusion_matrix'][i])
        best_mean_oob = (result['mean_oob'][i])
        best_pred = (result['prediction'][i])
        print('Best number of ntree: ',best_n_tree)
        print('Best number of max fitur: ',best_max_fitur)
        print('Best number of max depth: ',best_max_depth)
        print('Best number of min split: ',best_min_samples_split)
        print('Best recall: ', "%.2f" % best_recall)
        print('Best precision: ', "%.2f" % best_precision)
        print('Best f1-score: ', "%.2f" % best_f1_score)
        print('Best confusion matrix: ',  best_cm_score)
        print('OOB Estimate: ', "%.2f" % best_mean_oob)

        #Buat Plot untuk hasil confusion matrix
        plt.clf()
        plt.imshow(best_cm_score)
        classNames = ['Hard News','Soft News']
        plt.title('Hard or Soft News Confusion Matrix - RF')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TP','FN'], ['FP', 'TN']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(best_cm_score[i][j]), color='r')
        plt.show()

# save hasil prediksi kedalam file csv
from pandas import DataFrame
resPred = []
for i in range(len(best_pred)):
    resPred.append([X_test.index[i] + 1, best_pred[i], y_test[i]])
pred_df = DataFrame(resPred)
pred_df.columns = ['dokumen','Prediksi', 'Actual']

pred_df.to_csv(r'C:\Sidang Akhir\Sidang\Hasil\RF\IG(0.01)-90.csv', index = None)


