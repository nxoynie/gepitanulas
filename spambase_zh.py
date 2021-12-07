# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:39:55 2021

@author: iivan
"""
import numpy as np;
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
import matplotlib.colors as col;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression;
from sklearn.tree import DecisionTreeClassifier, plot_tree;    # importing decision tree tools
from sklearn.neural_network import MLPClassifier; # importing neural network classifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve; #  importing performance metrics

url="https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spamdata.csv"
df=pd.read_csv(url, sep = ";")
print(df.head())
print(df.describe())
names = df.columns.values.tolist()
data = df.values
X = data[:,0:56]
y = data[:,57]
input_names = names[0:56];
target_names = ['not spam','spam'];
#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal="kde");
#plt.figure();
andrews_curves(df, "spam");
plt.figure();
#parallel_coordinates(df, "spam");

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2020);
crit = 'gini';
depth =4;
# Instance of decision tree class
class_tree = DecisionTreeClassifier(criterion=crit,max_depth=depth);

# Fitting decision tree on training dataset
class_tree.fit(X_train, y_train);
score_train = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test_tree = class_tree.score(X_test, y_test); # Goodness of tree on test dataset
y_pred_gini = class_tree.predict(X_test); # Predicting spam for test data

# Plot of decision tree
fig = plt.figure(2,figsize = (16,10),dpi=100);
plot_tree(class_tree, feature_names = input_names, 
               class_names = target_names,
               filled = True, fontsize = 6);
# Writing to local repository as C:\\Users\user_name
fig.savefig('spambase_tree_gini.png');  

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);   # spam prediction for train
score_logreg_train = logreg_classifier.score(X_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities
score_test_logreg = logreg_classifier.score(X_test,y_test);

#Plot of logreg
#x_axis = 5;  # x axis attribute (0..56)
#y_axis = 22;  # y axis attribute (0..56)
# Enter axis from consol
#user_input = input('X axis [0..56, default:5]: ');
#if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
#    x_axis = np.int8(user_input);
#user_input = input('Y axis [0..56, default:22]: ');
#if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
#    y_axis = np.int8(user_input);    
#colors = ['blue','red'];  #  colors of spam and not spam
#fig = plt.figure(3);
#plt.title('Scatterplot for test digits dataset');
#plt.xlabel(input_names[x_axis]);
#plt.ylabel(input_names[y_axis]);
#plt.scatter(X_test[:,x_axis],X_test[:,y_axis],
#            c=ypred_logreg,cmap=col.ListedColormap(colors));
#plt.show();


# Fitting neural network classifier
neural_classifier = MLPClassifier(hidden_layer_sizes=(5),activation='logistic',max_iter=500);  #  number of hidden neurons: 5
neural_classifier.fit(X_train,y_train);
ypred_neural = neural_classifier.predict(X_train); # spam prediction for train
score_train_neural = neural_classifier.score(X_train,y_train); #  goodness of fit
cm_neural_train = confusion_matrix(y_train, ypred_neural); # train confusion matrix
ypred_neural = neural_classifier.predict(X_test); # spam prediction for test
cm_neural_test = confusion_matrix(y_test, ypred_neural); # test confusion matrix
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
yprobab_neural = neural_classifier.predict_proba(X_test); #  prediction probabilities

#Plot of neural
#x_axis = 5;  # x axis attribute (0..56)
#y_axis = 22;  # y axis attribute (0..56)
# Enter axis from consol
#user_input = input('X axis [0..56, default:5]: ');
#if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
#    x_axis = np.int8(user_input);
#user_input = input('Y axis [0..56, default:22]: ');
#if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=56 :
#    y_axis = np.int8(user_input);    
#colors = ['blue','red'];  #  colors of spam and not spam
#fig = plt.figure(4);
#plt.title('Scatterplot for test digits dataset');
#plt.xlabel(input_names[x_axis]);
#plt.ylabel(input_names[y_axis]);
#plt.scatter(X_test[:,x_axis],X_test[:,y_axis],
#            c=ypred_neural,cmap=col.ListedColormap(colors));
#plt.show();

# Printing the results
print(f'Test score of tree in %: {score_test_tree*100}');
print(f'Test score of logreg in %: {score_test_logreg*100}');
print(f'Test score of neural in %: {score_test_neural*100}');

print(f'Confusion matrix for neural train: {cm_neural_train}')
print(f'Confusion matrix for neural test: {cm_neural_test}')

plot_roc_curve(neural_classifier, X_test, y_test);

fpr_neural, tpr_neural, _ = roc_curve(y_test, yprobab_neural[:,1]);
roc_auc_neural = auc(fpr_neural, tpr_neural);

plt.figure(6);
lw = 2;
plt.plot(fpr_neural, tpr_neural, color='red',
         lw=lw, label='Neural (AUC = %0.2f)' % roc_auc_neural);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();