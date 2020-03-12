from warnings import simplefilter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(r"C:\Users\g-boz\Desktop\Trabalho IA\datasets\cleveland.csv", header=None)
df.columns = ['idade', 'sexo', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['thal'] = df.thal.fillna(df.thal.mean())#SUBSITUI OS VALORES NULOS PELA MEDIA
df['ca'] = df.ca.fillna(df.ca.mean())#SUBSITUI OS VALORES NULOS PELA MEDIA
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

X = df.iloc[:, :-1].values#X tem todas as colunas menos a ultima
y = df.iloc[:, -1].values#Y tem apenas a ultima coluna (target)

sc = ss()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ENCONTRANDO OS PARAMETROS POR CROSS-VALIDATION
######################################################SVM####################################################################################
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(), tuned_parameters, cv=3)
clf.fit(X_train, y_train)
print(clf.cv_results_['params'][clf.best_index_])#RETORNA OS PARAMETROS QUE MAXIMIZAM A MEDIA DAS ACURACIAS

######################################################KNN####################################################################################
tuned_parameters1 = [{'n_neighbors':[1,2,3,4,5,6,7,8,9,10,15,20,25],'weights':['uniform','distance'],
                      'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]
clf1 = GridSearchCV(KNeighborsClassifier(), tuned_parameters1,cv=3)
clf1.fit(X_train, y_train)
print(clf1.cv_results_['params'][clf1.best_index_])

######################################################NAIVE-BAYES####################################################################################
tuned_parameters2 = [{'var_smoothing':[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]}]
clf2 = GridSearchCV(GaussianNB(), tuned_parameters2,cv=3)
clf2.fit(X_train, y_train)
print(clf2.cv_results_['params'][clf2.best_index_])

######################################################DECISIONTREE####################################################################################
tuned_parameters3 = [{'criterion':['gini','entropy'], 'splitter':['best','random']}]
clf3 = GridSearchCV(DecisionTreeClassifier(), tuned_parameters3,cv=3)
clf3.fit(X_train, y_train)
print(clf3.cv_results_['params'][clf3.best_index_])

######################################################MLP####################################################################################
tuned_parameters4 = [{'hidden_layer_sizes':[50,100,200,300], 'activation':['identity', 'logistic', 'tanh', 'relu'],
                      'solver':['lbfgs', 'sgd', 'adam'], 'alpha':[1e-3,1e-4,1e-5]}]
clf4 = GridSearchCV(MLPClassifier(), tuned_parameters4,cv=3)
clf4.fit(X_train, y_train)
print(clf4.cv_results_['params'][clf4.best_index_])
