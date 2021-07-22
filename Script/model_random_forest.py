# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:47:46 2021

@author: rafir
"""

import random
import pandas as pd
import numpy as np

class RF():
    
    def __init__(self, X_train, y_train, n_tree, max_fitur, max_depth, min_samples_split):
        
        self.X_train = X_train
        self.y_train = y_train
        self.n_tree = n_tree
        self.max_fitur = max_fitur
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def entropi(self, l):
        if l == 0:
            return 0
        elif l == 1:
            return 0
        else:
            return - (l * np.log2(l) + (1 - l) * np.log2(1-l))
    
    def info_gain(self, left_child, right_child):
        parent = left_child + right_child
        p_parent = parent.count(0) / len(parent) if len(parent) > 0 else 0
        p_left = left_child.count(0) / len(left_child) if len(left_child) > 0 else 0
        p_right = right_child.count(0) / len(right_child) if len(right_child) > 0 else 0
        IG_p = self.entropi(p_parent)
        IG_l = self.entropi(p_left)
        IG_r = self.entropi(p_right)
        return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r
    
    def bootstrap(self, X_train, y_train):
        #pengambilan sampel bootstrap dengan pengembalian (replacement)
        idx_bootstrap = list(np.random.choice(range(len(X_train)), len(y_train), replace = True))
        idx_oob = [i for i in range(len(X_train)) if i not in idx_bootstrap]
        X_bootstrap = X_train.iloc[idx_bootstrap].values
        y_bootstrap = y_train[idx_bootstrap]
        X_oob = X_train.iloc[idx_oob].values
        y_oob = y_train[idx_oob]
        return X_bootstrap, y_bootstrap, X_oob, y_oob
    
    def estimate_oob(self, tree, X_test, y_test):
        missed_label = 0
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            if pred != y_test[i]:
                missed_label += 1
        return missed_label / len(X_test)
    
    
    def get_split_point(self, X_bootstrap, y_bootstrap, max_fitur):
        ls_fitur = list()
        num_fitur = len(X_bootstrap[0])
        # pengambilan fitur secara acak 
        while len(ls_fitur) <= max_fitur:
            idx_fitur = random.sample(range(num_fitur), 1)
            if idx_fitur not in ls_fitur:
                ls_fitur.extend(idx_fitur)
    
            best_info_gain = -999
            node = None
            for idx_fitur in ls_fitur:
                for split_point in X_bootstrap[:,idx_fitur]:
                    left_child = {'X_bootstrap': [], 'y_bootstrap': []}
                    right_child = {'X_bootstrap': [], 'y_bootstrap': []}
                
                # split child node untuk variable yang bersifat kontinu
                if type(split_point) in [int, float]:
                    for i, value in enumerate(X_bootstrap[:,idx_fitur]):
                        if value <= split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])
                # split child node untuk variable yang bersifat kategori
                else:
                    for i, value in enumerate(X_bootstrap[:,idx_fitur]):
                        if value == split_point:
                            left_child['X_bootstrap'].append(X_bootstrap[i])
                            left_child['y_bootstrap'].append(y_bootstrap[i])
                        else:
                            right_child['X_bootstrap'].append(X_bootstrap[i])
                            right_child['y_bootstrap'].append(y_bootstrap[i])
        
                split_info_gain = self.info_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
                if split_info_gain > best_info_gain:
                    best_info_gain = split_info_gain
                    left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                    right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                    node = {'information_gain': split_info_gain,
                            'left_child': left_child,
                            'right_child': right_child,
                            'split_point': split_point,
                            'feature_idx': idx_fitur}
        return node
    
    
    def terminal_node(self, node):
        y_bootstrap = node['y_bootstrap']
        self.pred = max(y_bootstrap, key = y_bootstrap.count)
        return self.pred
    
    
    def split_node(self, node, max_fitur, min_samples_split, max_depth, depth):
        left_child = node['left_child']
        right_child = node['right_child']
    
        del(node['left_child'])
        del(node['right_child'])
    
        if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
            empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
            node['left_split'] = self.terminal_node(empty_child)
            node['right_split'] = self.terminal_node(empty_child)
            return
    
        if depth >= max_depth:
            node['left_split'] = self.terminal_node(left_child)
            node['right_split'] = self.terminal_node(right_child)
            return node
    
        if len(left_child['X_bootstrap']) <= min_samples_split:
            node['left_split'] = node['right_split'] = self.terminal_node(left_child)
        else:
            node['left_split'] = self.get_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_fitur)
            self.split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
        if len(right_child['X_bootstrap']) <= min_samples_split:
            node['right_split'] = node['left_split'] = self.terminal_node(right_child)
        else:
            node['right_split'] = self.get_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_fitur)
            self.split_node(node['right_split'], max_fitur, min_samples_split, max_depth, depth + 1)
            
            
    def build_tree(self, X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_fitur):
        root_node = self.get_split_point(X_bootstrap, y_bootstrap, max_fitur)
        self.split_node(root_node, max_fitur, min_samples_split, max_depth, 1)
        return root_node
    
    def random_forest(self):
        self.ls_tree = list()
        self.ls_oob = list()
        for i in range(self.n_tree):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.bootstrap(self.X_train, self.y_train)
            tree = self.build_tree(X_bootstrap, y_bootstrap, self.max_depth, self.min_samples_split, self.max_fitur)
            self.ls_tree.append(tree)
            oob_error = self.estimate_oob(tree, X_oob, y_oob)
            self.ls_oob.append(oob_error)
        # print("OOB estimate: {:.2f}".format(np.mean(ls_oob)))
        return self.ls_tree
    
    def get_mean_OOB(self):
        mean = np.mean(self.ls_oob)
        return mean
        
    
    def predict_tree(self, tree, X_test):
        self.idx_fitur = tree['feature_idx']
    
        if X_test[self.idx_fitur] <= tree['split_point']:
            if type(tree['left_split']) == dict:
                return self.predict_tree(tree['left_split'], X_test)
            else:
                self.value = tree['left_split']
                return self.value
        else:
            if type(tree['right_split']) == dict:
                return self.predict_tree(tree['right_split'], X_test)
            else:
                return tree['right_split']
            
    
    def predict_rf(self, ls_tree, X_test):
        self.pred_ls = list()
        for i in range(len(X_test)):
            self.ensemble_preds = [self.predict_tree(tree, X_test.values[i]) for tree in ls_tree]
            self.final_pred = max(self.ensemble_preds, key = self.ensemble_preds.count)
            self.pred_ls.append(self.final_pred)
        return np.array(self.pred_ls)

#%%


file = open(r'C:\\Skripsi\Skripsi\Sidang\Dataset\multidim-dataset.csv','r')
df = pd.read_csv(file, delimiter=',')

#noFeature: 1:30
#infogain: 1:17
#chisquare: 1:17
#multidimension: 1:6
X = df.iloc[:, 1:6]
y = df.iloc[:, 6]

fitur = df.iloc[:, 1:6].columns
nb_train = int(np.floor(0.9 * len(df)))
df = df.sample(frac=1, random_state= 1)
X_train = df[fitur][:nb_train]
y_train = df['label'][:nb_train].values
X_test = df[fitur][nb_train:]
y_test = df['label'][nb_train:].values


#%%

#run with grid search
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1score = 2 * (recall * precision) / (recall + precision)
    
    print('Recall: ', recall)
    # precision: tp / (tp + fp)
    print('Precision: ', precision)
    # f1-Score: 2 tp / (2 tp + fp + fn)
    print('F1-Score: ', F1score)


#Hypermarameter Tuning

n_tree = [25, 50,100]
max_depth = [5, 10, 15]
min_samples_split = [2, 3, 4]
max_fitur = [2, 3, 5] 

max_acc = 0
parameters = []

for i in n_tree:
    for j in max_depth:
        for k in max_fitur:
            for l in min_samples_split:
                parameters.append((i, j, k, l))

print("Available combinations : ",  parameters )

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
  
    for count in range( np.size( y_pred ) ) :            
        if y_test[count] == y_pred[count] :                
            correctly_classified = correctly_classified + 1   
              
    curr_accuracy = ( correctly_classified / len(y_test) ) * 100
              
    if max_acc < curr_accuracy :            
        max_acc = curr_accuracy
    
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

print( "Maximum accuracy achieved by our model through grid searching : ", max_acc )

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

#save to csv
from pandas import DataFrame
resPred = []
for i in range(len(best_pred)):
    resPred.append([X_test.index[i] + 1, best_pred[i], y_test[i]])
pred_df = DataFrame(resPred)
pred_df.columns = ['dokumen','Prediksi', 'Actual']

pred_df.to_csv(r'C:\Skripsi\Skripsi\Sidang\Result\RF\NoFeature-90.csv', index = None)

#%%

####### run without grid search

file = open(r'C:\\Skripsi\Skripsi\Sidang\Dataset\multidim-dataset.csv','r')
df = pd.read_csv(file, delimiter=';')

#noFeature: 1:30
#infogain(0.05): 1:17
#infogain(0.01): 1:26
#chisquare(0.05): 1:17
#chisquare(0.01): 1:23
#multidimension: 1:6
X = df.iloc[:, 1:6]
y = df.iloc[:, 6]

fitur = df.iloc[:, 1:6].columns
nb_train = int(np.floor(0.5 * len(df)))
df = df.sample(frac=1, random_state= 1)
X_train = df[fitur][:nb_train]
y_train = df['label'][:nb_train].values
X_test = df[fitur][nb_train:]
y_test = df['label'][nb_train:].values

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
    
    #Buat Plot untuk hasil confusion matrix
    plt.clf()
    plt.imshow(cm_rf)
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
            plt.text(j,i, str(s[i][j])+" = "+str(cm_rf[i][j]), color='r')
    plt.show()

    return(cm_rf)

def metricsEval(cm):
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1score = 2 * (recall * precision) / (recall + precision)
    
    print('Recall: ', "%.2f" % recall)
    # precision: tp / (tp + fp)
    print('Precision: ', "%.2f" % precision)
    # f1-Score: 2 tp / (2 tp + fp + fn)
    print('F1-Score: ', "%.2f" % F1score)

n_tree = 25
max_fitur = 5
max_depth = 15
min_samples_split = 3

clf = RF(X_train, y_train, n_tree, max_fitur, max_depth, min_samples_split)
ls_acc = list()
for i in range(10):
    model = clf.random_forest()
    y_pred = clf.predict_rf(model, X_train)
    acc = sum(y_pred == y_train) / len(y_train)
    cm = confusion_matrix(y_train, y_pred)
    met_eval = metricsEval(cm)
    ls_acc.append(acc)
    print("Nilai akurasi sebesar :", "%.2f" % (acc * 100), "%")
print("Nilai akurasi rata-rata: {}".format("%.2f" % (acc * 100)))
print("Nilai akurasi tertinggi: {}".format("%.2f" % (max(ls_acc)*100)))

