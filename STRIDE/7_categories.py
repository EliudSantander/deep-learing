# -*- coding: utf-8 -*-
import numpy as np
from redNeuronal import Neural_Network, trainer #computeNumericalGradient
from io import open
import pandas as pd

#Revisar que pasa si no existe el archivo
fichero = open('pesosEntrenadosW1.txt', 'w')
fichero.close()
fichero = open('pesosEntrenadosW1.txt', 'r')
entrada1 = fichero.read()
fichero.close()
fichero = open('pesosEntrenadosW2.txt', 'w')
fichero.close()
fichero = open('pesosEntrenadosW2.txt', 'r')
entrada2 = fichero.read()
fichero.close()

RN = Neural_Network()
#RN.randWeight()  # Activar esta funcion para generar nuevos pesos
RN.dfltWeight()  # Activar esta funcion para usar pesos ya guardados

## Recibir datos de entrenamiento
fichero = open('trainX.txt', 'r')
lineas = fichero.readlines()
fichero.close()
X_train = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    X_train.append(linea[:])

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
    
X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float)
y_train = y_train/23

# Recibir datos de entrada y avanzar en la red
fichero = open('testX.txt', 'r')
lineas = fichero.readlines()
fichero.close()
X_test = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    X_test.append(linea[:41])

fichero = open('testy.txt', 'r')
lineas = fichero.readlines()
fichero.close()
y_test = []
for l in range(len(lineas)):
    linea = lineas[l] 
    linea = linea.replace("\n", "").split(",")
    for n in range(len(linea)):
        linea[n] = float(linea[n])
    y_test.append([linea[0]])
X_test = tuple(X_test)
y_test = tuple(y_test)

X_test = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=float)

# Escalado de los datos
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# While ¿no cuente con previo pesos /entrenamiento?
while entrada1 == "" or entrada2 == "":
    
    ## Entrenamiento y guarda pesos
    T = trainer(RN)
    T.train(X_train,y_train)
    
    fichero = open('pesosEntrenadosW1.txt', 'r')
    entrada1 = fichero.read()
    fichero.close()
    fichero = open('pesosEntrenadosW2.txt', 'r')
    entrada2 = fichero.read()
    fichero.close()
    

yHat_train = RN.forward(X_train)
# Generar y guardar resultados
rYHat_train = np.around(yHat_train*23)
norm, spoof, tamp, repu, info, deni, elev = 0,0,0,0,0,0,0
for i in range(len(rYHat_train)):
    au = int(rYHat_train[i])
    if au == 0 or au == 1:
        norm += 1
    if au == 4 or au == 5 or au == 7 or au == 8:
        spoof += 1
    if au == 12 or au == 13 or au == 17:
        tamp += 1
    if au == 3 or au == 18:
        repu += 1
    if au == 14 or au == 19 or au == 22:
        info += 1
    if au == 2 or au == 6 or au == 9 or au == 10 or au == 11 or au == 20:
        deni += 1    
    if au == 15 or au == 16 or au == 21 or au == 23:
        elev += 1

y_train = np.around(y_train*23)
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, rYHat_train)

length, width = cm_train.shape

suma_train = 0

for i in range(length):
    suma_train += cm_train[i][i]
    
cm_acierto_train = suma_train*100/rYHat_train.size
# ---------------------------------------------------------------------------------------------

y_list = list()

# Imprimir los valores esperados
norm1, spoof1, tamp1, repu1, info1, deni1, elev1 = 0,0,0,0,0,0,0
for i in range(len(y_test)):
    au = int(y_test[i])
    if au == 0 or au == 1:
        norm1 += 1
        y_list.append("normal")
    if au == 4 or au == 5 or au == 7 or au == 8:
        spoof1 += 1
        y_list.append("spoofing")
    if au == 12 or au == 13 or au == 17:
        tamp1 += 1
        y_list.append("tampering")
    if au == 3 or au == 18:
        repu1 += 1
        y_list.append("repudiation")
    if au == 14 or au == 19 or au == 22:
        info1 += 1
        y_list.append("disclosure")
    if au == 2 or au == 6 or au == 9 or au == 10 or au == 11 or au == 20:
        deni1 += 1
        y_list.append("denial")
    if au == 15 or au == 16 or au == 21 or au == 23:
        elev1 += 1
        y_list.append("elevation")
    
y_test = np.array(y_list)

print("\nPaquetes esperados:\n{} normal\n{} spoofing\n \
{} tampering\n{} repudiation\n{} disclosure\n{} denial\n \
{} elevation".format(norm1, spoof1, tamp1, repu1, info1, deni1, elev1))

# y = y/23

y_list = list()

yHat = RN.forward(X_test)
# Generar y guardar resultados
rYHat = np.around(yHat*23)
norm, spoof, tamp, repu, info, deni, elev = 0,0,0,0,0,0,0
for i in range(len(rYHat)):
    au = int(rYHat[i])
    if au == 0:
        norm += 1
        y_list.append("normal")
        rYHat[i] = 1
    if au == 1:
        norm += 1
        y_list.append("normal")
    if au == 4 or au == 5 or au == 7 or au == 8:
        spoof += 1
        y_list.append("spoofing")
    if au == 12 or au == 13 or au == 17:
        tamp += 1
        y_list.append("tampering")
    if au == 3 or au == 18:
        repu += 1
        y_list.append("repudiation")
    if au == 14 or au == 19 or au == 22:
        info += 1
        y_list.append("disclosure")
    if au == 2 or au == 6 or au == 9 or au == 10 or au == 11 or au == 20:
        deni += 1   
        y_list.append("denial")
    if au == 15 or au == 16 or au == 21 or au == 23:
        elev += 1
        y_list.append("elevation")

rYHat = np.array(y_list)

print("\nPaquetes obtenidos:\n{} normal\n{} spoofing\n \
{} tampering\n{} repudiation\n{} disclosure\n{} denial\n \
{} elevation".format(norm, spoof, tamp, repu, info, deni, elev))


e_norm = np.abs(100 - (norm/norm1*100))
e_spoof = np.abs(100 - (spoof/spoof1*100))
e_tamp = np.abs(100 - (tamp/tamp1*100))
e_repu = np.abs(100 - (repu/repu1*100))
e_info = np.abs(100 - (info/info1*100))
e_deni = np.abs(100 - (deni/deni1*100))
e_elev = np.abs(100 - (elev/elev1*100))

p_acierto = 100 - (e_norm + e_spoof + e_tamp + e_repu + e_info + e_deni + e_elev)/7

y_test = pd.DataFrame(y_test)
y_test = y_test.iloc[:, 0].values

rYHat = pd.DataFrame(rYHat)
rYHat = rYHat.iloc[:, 0].values

y_true = pd.Series(y_test, name="Esperados")
y_pred = pd.Series(rYHat, name="Obtenidos")
df_confusion = pd.crosstab(y_true, y_pred)
df_confusion.to_csv('cm_7_categories.csv')
df_confusion.to_html('cm_7_categories.html')
# print(df_confusion)

cm = confusion_matrix(y_test, rYHat)

#y = y/23

length, width = cm.shape

suma_aciertos = 0

for i in range(length):
    suma_aciertos += cm[i][i]
    
cm_acierto = suma_aciertos*100/rYHat.size
    
# Imprimiendo todos los resultados a considerar
print("\nNumero de registros utilizados: 1500")
print("Numero de iteraciones: 200")
print("Porcentaje de acierto(documento original): " + str(p_acierto))
print("Porcentaje de acierto(matriz de confusion): " + str(cm_acierto))
print("% de acierto con el conjunto de training: " + str(cm_acierto_train))