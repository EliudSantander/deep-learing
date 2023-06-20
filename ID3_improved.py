#coding:utf-8
# Algoritmo ID3 para construir un árbol de decisiones
import numpy as np
import math
import pandas as pd

#Crea un conjunto de datos
def createDataSet():
    
    features = ['srace','sethnici','sgender','vrace','vethnici','vgender']
    arreglo = pd.read_csv("SanFranciscoHomicides.csv")
    dataSet = np.array(arreglo)
    return dataSet, features

#Calcular la entropía del conjunto de datos
def calcEntropy(dataSet):
    #Probabilidad primero
    labels = list(dataSet[:,-1])
    
    prob = {}
    entropy = 0.0
    for label in labels:
        prob[label] = (labels.count(label) / float(len(labels)))
    for v in prob.values():
        entropy = entropy + (-v * math.log(v,2))
    return entropy

#Aqui se calcula la importancia de cada categoria
def calcImportance(dataSet):
    X = dataSet[:, 0:-1] # Se seleccionan los atributos para calcular importancia
    y = dataSet[:, 3] # Se selecciona la categoria objetivo del arbol
    
    X = X.T # Transponiendo la X por comodidad
    
    valor_objetivo = [y[0]]
    # for value in y:
    #     if value not in valor_objetivo:
    #         valor_objetivo.append(value)
            
    #print(valor_objetivo)
    
    importancias = []
    
    # Recorre cada atributo del dataset para calcular la funcion de correlacion
    AFs = [] # Esta lista guardara los valores de la correlacion de cada atributo
    for atribute in X:
        lista = [atribute[0]]
        for value in atribute:
            if value not in lista:
                lista.append(value)
        #print(lista)
        
        contador = 0
        suma_value = []
        for element in lista: #Se recorren los elementos de la lista de los valores de la categoria
            indice = np.array(np.where(atribute == element)) # Se identifican los indices del valor estudiado
            # Lo siguiente aplica la funcion de correlacion entre el atributo y el objetivo
            for i in range(indice.size):
                    if y[indice[0][i]] == valor_objetivo[0]:
                        contador+=1
                    else:
                        contador-=1
            if contador < 0:
                contador = contador*-1 # Obtiene el valor absoluto (osea positivo)
            suma_value.append(contador)
        # Funcion de correlacion del atributo
        AF = 0
        for suma in suma_value:
            # print("suma: " + str(suma))
            AF = AF + suma
        # print("Total: " + str(AF))
        # print("Valores: " + str(len(lista)))
        AF = AF/len(lista)
        AFs.append(AF)
    # Se utilizan las funciones de correclacion para calcular la importancia
    sumaAF  = 0
    # Se suman las correlaciones
    for valor in AFs:
        sumaAF = sumaAF + valor
    importancias = [] # Arreglo donde se guardan las importancias
    # Se guardan las importancias
    for valor in AFs:
        importancias.append(valor/sumaAF)
    return importancias

#Conjunto de datos de partición
def splitDataSet(dataSet, i, fc):
    subDataSet = []
    for j in range(len(dataSet)):
        if dataSet[j, i] == str(fc):
            sbs = []
            sbs.append(dataSet[j, :])
            subDataSet.extend(sbs)
    subDataSet = np.array(subDataSet)
    return np.delete(subDataSet,[i],1)

#Calcule la ganancia de información
def chooseBestFeatureToSplit(dataSet, importance):
    bestInfoGain = 0.0   #Ganancia máxima de información
    bestFeature = -1   
    #Extraiga la columna de características y la columna de etiquetas
    for i in range(dataSet.shape[1]-1):     #Columna
        #Calculo de la importancia de cada atribtuo de ctaegoria
        new_importance = importance[i] #Importancia de cada categoria
        #Calcule la probabilidad de cada categoría
        prob = {}
        featureCoulmnL = list(dataSet[:,i])
        for fcl in featureCoulmnL:
            prob[fcl] = featureCoulmnL.count(fcl) / float(len(featureCoulmnL))
        #Calcule la entropía de cada categoría
        featureCoulmn = set(dataSet[:,i])   #Columna de funciones
        
        for fc in featureCoulmn:
            subDataSet = splitDataSet(dataSet, i, fc)
            new_entropy = calcEntropy(subDataSet)   #Entropía de cada categoría
        infoGain = calcEntropy(dataSet) - new_entropy*new_importance    #Calcular la ganancia de información
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#Si las características del conjunto de características están vacías, entonces T es un solo nodo, y la etiqueta de clase con el árbol de instancia más grande en el conjunto de datos D se usa como la etiqueta de clase del nodo, y se devuelve T
def majorityLabelCount(labels):
    labelCount = {}
    for label in labels:
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    return max(labelCount)

#Construir árbol de decisión T
def createDecisionTree(dataSet, features, importance):
    labels = list(dataSet[:,-1])
    #Si todas las instancias en el conjunto de datos pertenecen a la misma etiqueta de clase, T es un árbol de un solo nodo, y la etiqueta de clase se usa como etiqueta de clase del nodo, y se devuelve T
    if len(set(labels)) == 1:
        return labels[0]
    #Si las características del conjunto de características están vacías, entonces T es un solo nodo, y la etiqueta de clase con el árbol de instancia más grande en el conjunto de datos D se usa como la etiqueta de clase del nodo, y se devuelve T
    if len(dataSet[0]) == 1:
        return majorityLabelCount(labels)
    #De lo contrario, calcule la ganancia de información de cada característica en el conjunto de características para el conjunto de datos D de acuerdo con el algoritmo ID3, y seleccione la característica con la mayor ganancia de información, beatFeature
    bestFeatureI = chooseBestFeatureToSplit(dataSet, importance)  #Subíndice de la mejor característica
    bestFeature = features[bestFeatureI]    #Mejor característica
    decisionTree = {bestFeature:{}} #Construya un árbol con la característica bestFeature con la mayor ganancia de información como nodo hijo
    del(features[bestFeatureI])    #Esta función se ha utilizado como un nodo secundario, elimínela para que pueda continuar construyendo el subárbol
    bestFeatureColumn = set(dataSet[:,bestFeatureI])
    for bfc in bestFeatureColumn:
        subFeatures = features[:]
        decisionTree[bestFeature][bfc] = createDecisionTree(splitDataSet(dataSet, bestFeatureI, bfc), subFeatures, importance)
    return decisionTree

#Categorizar datos de prueba
def classify(testData, features, decisionTree):
    for key in decisionTree:
        index = features.index(key)
        testData_value = testData[index]
        subTree = decisionTree[key][testData_value]
        if type(subTree) == dict:
            result = classify(testData,features,subTree)
            return result
        else:
            return subTree


if __name__ == '__main__':
    dataSet, features = createDataSet()     #Crea un conjunto de datos
    importance = calcImportance(dataSet)
    #importance = [0.2,0.4,0.5] # Este se va a eliminar cuando el calculo se haga corectamente
    decisionTree = createDecisionTree(dataSet, features, importance)   #Construya un árbol de decisiones
    print(decisionTree)

    dataSet, features = createDataSet()
    testData = ['srace1', 'seth1', 'sm', 'vrace1', 'veth1', 'vm']
    result = classify(testData, features, decisionTree)  #Categorizar datos de prueba