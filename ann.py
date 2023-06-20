# Redes Neuronales Artificiales

# Instalar Theano 
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Importando las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando el dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Codifica datos categoricos. No siempre es necesario
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
## CREA TANTOS LABEL ENCODERS COMO NECESITES Y LUEGO HAZ UN COLUMN TRANSFORMER
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'),[1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float) #Convierte los paises a numeros

# Quito la primera columna de los paises porque si un individuo no es español ni aleman obviamente es frances
# Asi se evita el problema de la colinealidad
X = X[:, 1:]

# Dividiendo el dataset en conjunto de entrenamiento y conjunto de testeo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
# El escalado se ajusta con el conjunto de training, por eso el de test solo tiene transform y no el fit

# Parte 2 - Construir el RNA

# Importando Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Iniciar la RNA
classifier = Sequential()

# Añadir capa de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer= "uniform", 
                     activation = "relu", input_dim = 11))
#classifier.add(Dropout(p = 0.1))

# Segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer= "uniform", 
                     activation = "relu")) # No es necesario indicar la dimension de entrada 
                                            # porque la red ya la sabe
#classifier.add(Dropout(p = 0.1))
                                            
# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer= "uniform", 
                     activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Ajustar el RNA al conjunto de entrenamiento
classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)

# Prediciendo los resultados
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #Esto se hizo para obtener un valor binario 1 y 0

# Creando matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
