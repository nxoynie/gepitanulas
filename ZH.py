# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:07:57 2021

@author: xszbe
"""

import numpy as np;
import pandas as pd;
from sklearn.metrics import confusion_matrix, roc_curve, auc;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.neural_network import MLPClassifier;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.cluster import KMeans;
from sklearn.metrics import davies_bouldin_score;
from sklearn.decomposition import PCA;
from matplotlib import pyplot as plt;
from sklearn.datasets import load_wine;


# 1. feladat
wine = load_wine();
print(f'Rekordok száma: {wine.data.shape[0]}');
print(f'Attribútumok száma: {wine.data.shape[1]}');
print(f'Osztályok száma: {len(wine.target_names)}');


# 2. feladat
df = pd.DataFrame(data=wine.data, columns=wine.feature_names);
df['Target'] = wine.target;

mean_by_target = df.groupby(by="Target").mean();
print(mean_by_target);

std_by_target = df.groupby(by="Target").std();
print(std_by_target);


#3. feladat
plt.figure(1);
pd.plotting.parallel_coordinates(df,class_column='Target');
plt.show();

plt.figure(2);
pd.plotting.scatter_matrix(df);
plt.show();

# 4. feladat
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
                                                    test_size=0.2,
                                                    shuffle = True,
                                                    random_state=2021);


# 5. feladat
class_tree = DecisionTreeClassifier(criterion = 'gini',
                                    max_depth = 4);
class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train);
score_test_tree = class_tree.score(X_test, y_test);


logreg_classifier = LogisticRegression(solver = 'newton-cg');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);
ypred_logreg = logreg_classifier.predict(X_test);
yprobab_logreg = logreg_classifier.predict_proba(X_test);


neural_classifier = MLPClassifier(hidden_layer_sizes=(3),
                                  activation='logistic',
                                  max_iter=1000);
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);
score_test_neural = neural_classifier.score(X_test,y_test);
ypred_neural = neural_classifier.predict(X_test);
yprobab_neural = neural_classifier.predict_proba(X_test);

print(f'Test score of tree in %: {score_test_tree*100}');
print(f'Test score of logreg in %: {score_test_logreg*100}'); # A logreg volt a legtöbbször jó
print(f'Test score of neural in %: {score_test_neural*100}');


# 6. feladat
cm_logreg_test = confusion_matrix(y_test, ypred_logreg);

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,0], pos_label=0);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

plt.figure(3);
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=2, label='Logreg (area = %0.2f)' % roc_auc_logreg);
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();


# 7. feladat
Max_K = 32;
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2021);
    kmeans.fit(wine.data);
    labels = kmeans.labels_;
    centers = kmeans.cluster_centers_;
    DB[i] = davies_bouldin_score(wine.data,labels);

optimal_n_c, = np.where(np.isclose(DB, DB.min()))[0] + 2;

print(f'Minimum Davies-Bouldin index: {DB.min()}');
print(f'Optimal number of cluster: {optimal_n_c}');

kmeans = KMeans(n_clusters=optimal_n_c, random_state=2021);
kmeans.fit(wine.data);
labels = kmeans.labels_;
centers = kmeans.cluster_centers_;
score = kmeans.score(wine.data);

pca = PCA(n_components=2);
pca.fit(wine.data);
data_pc = pca.transform(wine.data);
centers_pc = pca.transform(centers);

fig = plt.figure(4);
plt.title('Clustering of the Wine data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(data_pc[:,0],data_pc[:,1],s=50,c=labels);
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');
plt.show();