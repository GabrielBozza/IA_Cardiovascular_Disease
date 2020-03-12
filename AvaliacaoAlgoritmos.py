import pandas as pd
from warnings import simplefilter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(r"C:\Users\g-boz\Desktop\Trabalho IA\datasets\cleveland.csv", header=None)

df.columns = ['idade', 'sexo', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

####################################PRE-PROCESSAMENTO#######################################################################

df['thal'] = df.thal.fillna(df.thal.mean())#SUBSITUI OS VALORES NULOS PELA MEDIA
df['ca'] = df.ca.fillna(df.ca.mean())
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})#CLASSIFICACAO EM RISCO(1,2,3 OU 4) OU NAO RISCO(0)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

sc = ss()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
###########################################################################################################################

#########################################   SVM   ########################################################################

classifier = SVC(C= 1000, kernel= 'rbf',gamma=0.0001)
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

#########################################   Naive Bayes  ##################################################################

classifier = GaussianNB(var_smoothing=0.001)
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

#########################################   KNN  ##########################################################################

classifier = KNeighborsClassifier(n_neighbors=6,weights='distance')
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

#########################################   Decision Tree  #################################################################

classifier = DecisionTreeClassifier(criterion='entropy',splitter='random')
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

#########################################  MLP #############################################################################

classifier = MLPClassifier(activation= 'identity', alpha= 0.0001, hidden_layer_sizes=300, solver= 'sgd')
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



