import pandas as pd
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(r"C:\Users\g-boz\Desktop\Trabalho IA\datasets\cleveland.csv", header=None)

df.columns = ['idade', 'sexo', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

####################################PRE-PROCESSAMENTO#######################################################################
df['thal'] = df.thal.fillna(df.thal.mean())#SUBSITUI OS VALORES NULOS PELA MEDIA
df['ca'] = df.ca.fillna(df.ca.mean())
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})#CLASSIFICACAO EM RISCO(1,2,3 OU 4) OU NAO RISCO(0)

sc = ss()
df = pd.get_dummies(df, columns = ['sexo', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])#ATRIBUTOS CATEGORICOS
columns_to_scale = ['idade', 'trestbps', 'chol', 'thalach', 'oldpeak']#ATRIBUTOS NAO CATEGORICOS
df[columns_to_scale] = sc.fit_transform(df[columns_to_scale])#NORMALIZA ATRIBUTOS NAO CATEGORICOS
###########################################################################################################################

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#########################################   SVM   ########################################################################

classifier = SVC(C= 1000, kernel= 'rbf',gamma=0.001)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)
y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Acuracia no conjunto de treino SVM = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Acuracia no conjunto de teste SVM = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
print("Matriz de confusao do conjunto de treino SVM:")
print(confusion_matrix(y_pred_train, y_train))
print("Matriz de confusao do conjunto de teste SVM:")
print(confusion_matrix(y_pred, y_test))

print("Acuracias do cross-validation com 10-fold SVM:")
k=cross_val_score(SVC(C= 1000, kernel= 'rbf',gamma=0.001),X,y,cv=10)
print(k)
print("Media das acuracias SVM:")
print(k.mean())

#########################################   Naive Bayes  ##################################################################

classifier = GaussianNB(var_smoothing=0.00001)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)
y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Acuracia no conjunto de treino NB = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Acuracia no conjunto de teste NB = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
print("Matriz de confusao do conjunto de treino NB:")
print(confusion_matrix(y_pred_train, y_train))
print("Matriz de confusao do conjunto de teste NB:")
print(confusion_matrix(y_pred, y_test))

print("Acuracias do cross-validation com 10-fold NB:")
k=cross_val_score(GaussianNB(var_smoothing=0.00001),X,y,cv=10)
print(k)
print("Media das acuracias NB:")
print(k.mean())

#########################################   KNN  ##########################################################################

classifier = KNeighborsClassifier(n_neighbors=5,weights='distance')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)
y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Acuracia no conjunto de treino KNN = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Acuracia no conjunto de teste KNN = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
print("Matriz de confusao do conjunto de treino KNN:")
print(confusion_matrix(y_pred_train, y_train))
print("Matriz de confusao do conjunto de teste KNN:")
print(confusion_matrix(y_pred, y_test))

print("Acuracias do cross-validation com 10-fold KNN:")
k=cross_val_score(KNeighborsClassifier(n_neighbors=5,weights='distance'),X,y,cv=10)
print(k)
print("Media das acuracias KNN:")
print(k.mean())

#########################################   Decision Tree  #################################################################

classifier = DecisionTreeClassifier(criterion='gini',splitter='best')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)
y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Acuracia no conjunto de treino DecisionTree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Acuracia no conjunto de teste DecisionTree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
print("Matriz de confusao do conjunto de treino DecisionTree:")
print(confusion_matrix(y_pred_train, y_train))
print("Matriz de confusao do conjunto de teste DecisionTree:")
print(confusion_matrix(y_pred, y_test))

print("Acuracias do cross-validation com 10-fold DecisionTree:")
k=cross_val_score(DecisionTreeClassifier(criterion='gini',splitter='best'),X,y,cv=10)
print(k)
print("Media das acuracias DecisionTree:")
print(k.mean())

#########################################  MLP #############################################################################

classifier = MLPClassifier(activation= 'identity', alpha= 0.0001, hidden_layer_sizes=200, solver= 'adam')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_pred, y_test)
y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Acuracia no conjunto de treino MLP = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Acuracia no conjunto de teste MLP = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
print("Matriz de confusao do conjunto de treino MLP:")
print(confusion_matrix(y_pred_train, y_train))
print("Matriz de confusao do conjunto de teste MLP:")
print(confusion_matrix(y_pred, y_test))

print("Acuracias do cross-validation com 10-fold MLP:")
k=cross_val_score(MLPClassifier(activation= 'identity', alpha= 0.0001, hidden_layer_sizes=200, solver= 'adam'),X,y,cv=10)
print(k)
print("Media das acuracias MLP:")
print(k.mean())