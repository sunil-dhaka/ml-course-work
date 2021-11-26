import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from statistics import mean,stdev
# import warnings
# warnings.filterwarnings("ignore")

if len(sys.argv)>0 and sys.argv[1]!='':
    test_file_path=sys.argv[1]
    print('Test file path is >> ',test_file_path)
else:
    print('Please give a input test file path.')
    exit()

data=pd.read_table('training-s110.csv',header=None,sep=',')
test_data=pd.read_table(test_file_path,header=None,sep=',')

X_train=np.array(data.iloc[:,1:])
y_train=np.array(data.iloc[:,0])

X_test=np.array(test_data.iloc[:,1:])
y_test=np.array(test_data.iloc[:,0])

# Feature Scaling for input features.
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(X_train)

cv=10
model=DecisionTreeClassifier(random_state=1,criterion='gini',class_weight='balanced')
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
lst_accu_stratified = []

for train_index, test_index in skf.split(X_train, y_train):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    model.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(model.score(x_test_fold, y_test_fold))

# Print the output in terminal
print('Accuracy-Info for Stratified CV'.center(50,'='))
print('\nMaximum Accuracy:',
max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:',
min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:',
mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation:', stdev(lst_accu_stratified))

# train on whole dataset
classifier = DecisionTreeClassifier(random_state=1,criterion='gini',class_weight='balanced')
clf=classifier.fit(X_train, y_train)

# predict and save classes
test_predict=clf.predict(X_test)
with open('answer-s110.csv','w') as file:
    for class_label in list(test_predict):
        file.write(f'{class_label}\n')

# test_data.to_csv(test_file_path,sep=',')
# print(test_data.head())

if 'C0' in y_test:
    print('Model predicted classes have been save in answer-s110.csv')
else:
    print('Accuracy-Info for Test Data'.center(50,'='))
    print('Accuracy on test data supplied is: ',clf.score(X_test,y_test))