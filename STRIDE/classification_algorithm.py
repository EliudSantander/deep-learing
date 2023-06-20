# Importando librerias
import pandas as pd
import numpy as np
from io import open
from keras.utils import np_utils

# Leyendo los datasets

# Conjunto de testing original ---------------------------------------------
fichero = open('testX.txt', 'r')
lineas = fichero.readlines()
fichero.close()
X = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    X.append(linea[:41])

fichero = open('testy.txt', 'r')
lineas = fichero.readlines()
fichero.close()
y = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    y.append([linea[0]])
X = tuple(X)
y = tuple(y)

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

# Conjunto de training original -------------------------------------------
fichero = open('trainX.txt', 'r')
lineas = fichero.readlines()
fichero.close()
X_train = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    X_train.append(linea[:41])

fichero = open('trainy.txt', 'r')
lineas = fichero.readlines()
fichero.close()
y_train = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    y_train.append([linea[0]])
X_train = tuple(X_train)
y_train = tuple(y_train)

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float)


# Convirtiendo los datasets a un conjunto de pandas------------------------

# df_train = pd.read_excel(io="train_test1500.xlsx", sheet_name = 1)

df_trainX = pd.DataFrame(X_train)
df_trainy = pd.DataFrame(y_train)

df_testX = pd.DataFrame(X)
df_testy = pd.DataFrame(y)

df_trainX.insert(loc=41, column=41, value=df_trainy)

df_testX.insert(loc=41, column=41, value=df_testy)

df = pd.concat([df_trainX, df_testX])

df.describe()

X = df.iloc[:, :-1].values
y = df.iloc[:, 41].values


# Escalado de variables ---------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X = sc_X.transform(X)

# Conversion de los valores de salida y
y_list = list()

for i in range(len(y)):
    if y[i] == 0 or y[i] == 1:
        y_list.append("normal")
    if y[i] == 4 or y[i] == 5 or y[i] == 7 or y[i] == 8:
        y_list.append("spoofing")
    if y[i] == 12 or y[i] == 13 or y[i] == 17:
        y_list.append("tampering")
    if y[i] == 3 or y[i] == 18:
        y_list.append("repudiation")
    if y[i] == 14 or y[i] == 19 or y[i] == 22:
        y_list.append("disclosure")
    if y[i] == 2 or y[i] == 6 or y[i] == 9 or y[i] == 10 or y[i] == 11 or y[i] == 20:
        y_list.append("denial")
    if y[i] == 15 or y[i] == 16 or y[i] == 21 or y[i] == 23:
        y_list.append("elevation")
        
y = np.array(y_list)

# Conversion de los valores de salida y_train
y_list = list()

for i in range(len(y_train)):
    if y_train[i] == 0 or y_train[i] == 1:
        y_list.append("normal")
    if y_train[i] == 4 or y_train[i] == 5 or y_train[i] == 7 or y_train[i] == 8:
        y_list.append("spoofing")
    if y_train[i] == 12 or y_train[i] == 13 or y_train[i] == 17:
        y_list.append("tampering")
    if y_train[i] == 3 or y_train[i] == 18:
        y_list.append("repudiation")
    if y_train[i] == 14 or y_train[i] == 19 or y_train[i] == 22:
        y_list.append("disclosure")
    if y_train[i] == 2 or y_train[i] == 6 or y_train[i] == 9 or y_train[i] == 10 or y_train[i] == 11 or y_train[i] == 20:
        y_list.append("denial")
    if y_train[i] == 15 or y_train[i] == 16 or y_train[i] == 21 or y_train[i] == 23:
        y_list.append("elevation")
        
y_train = np.array(y_list)


# Se divide el dataset en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Codificando datos de y_train
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y_train = encoder.transform(y_train)

y = np_utils.to_categorical(y)
y_train = np_utils.to_categorical(y_train)


# Se crea el modelo
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dropout

def baseline_model():
    model = Sequential()
    model.add(Dense(units = 24, input_dim = 41, activation='relu'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 24, activation='relu'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(units = 7, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)

print("CV Mean: " + str(results.mean()))
print("CV Std: " + str(results.std()))

estimator.fit(X_train, y_train)

y_pred = estimator.predict(X)

estimator.get_params()


# Preparando el y_test para comparar en la matriz de confusion
# length, width = y_test.shape

# y_list = list()

# for i in range(length):
#     for j in range(width):
#         if y_test[i][j] == 1:
#             y_list.append(j)

# y_test = np.array(y_list)


# Creando la matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

length, width = cm.shape

suma_aciertos = 0

for i in range(length):
    suma_aciertos += cm[i][i]
    
cm_acierto = suma_aciertos*100/y.size
print("Porcentaje de acierto: " + str(cm_acierto))
