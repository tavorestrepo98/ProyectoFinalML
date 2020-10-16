"""
EL SIGUIENTE EJERCICIO TRATA DE MODELAR EL COMPORTAMIENTO DE LAS VARIABLES QUE DICTAN SI UN TUMOR DE SENO LLEGA A SER MALIGNO O BENIGNO,
PARA ESTO SE IMPLEMENTARAN DIFERENTES TECNICAS DE MACHINE LEARNING QUE NOS PERMITIRAN PREDECIR EL COMPORTAMIENTO DE ESTAS VARIABLES, DADO
QUE LOS RESULTADOS SE DICTAN A OBTENER SI ES O NO MALIGNO UN TUMOR, LAS TECNICAS QUE SE UTILIZARAN SERAN DE CLASIFICACION Y SE OPTARA POR
TRABAJAR DIFERENTES MODELOS QUE LOGREN PRECISAR LA OBTENCION DE FUTUROS RESULTADOS (Regresion Logistica, Redes neuronales, Clustering)
"""

# IMPORTO LOS MODULOS REQUERIDOS
# Manejo de DataSets
from pandas import DataFrame
import pandas as pd
# Numpy
import numpy as np
# Ploteo de datos
import matplotlib.pyplot as plt
# Normalizador de columnas, donde estandarizara valores a terminos binarios. Estandariza muestras de las columnas con valores distantes.
from sklearn.preprocessing import StandardScaler
# Datasets De prubea, donde obtendremos el que se trabajara
from sklearn import datasets
# Herramienta para dividir el conjunto de datos totales en grupos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Modelo para Regresion logistica
from sklearn.linear_model import LogisticRegression
# Modelo para KNN
from sklearn.neighbors import KNeighborsClassifier
# Modelo para Redes Neuronales
import tensorflow as tf
# Metricas para medir el comportamiento de los modelos
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, r2_score, f1_score

# Comenzamos cargando el dataSet
breastC = datasets.load_breast_cancer()
#Observamos la informacion a manejar
print(breastC.DESCR,"\n")

#Guardamos los datos como variables
x = breastC.data
y = breastC.target

clase = breastC.feature_names
resultado = breastC.target_names

# L O G I S T I C  R E G R E S S I O N
def logistic():
	# Hiperparametros
	testSize = 0.25
	trainValSize = 0.30
	logSolver = "liblinear"

	print("\nTest DataSet Size: ",testSize)
	print("Train&Validation DataSet Size: ", trainValSize)
	print("Solver: ", logSolver)

	X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=testSize)
	#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=trainValSize)

	#escalamos los datos
	escalar = StandardScaler()
	X_train = escalar.fit_transform(X_train)
	X_test = escalar.transform(X_test)
	#X_val = escalar.transform(X_val)

	logRegressor = LogisticRegression(solver='liblinear')

	logRegressor.fit(X_train, Y_train)

	y_predict = logRegressor.predict(X_test)
	print("\n",y_predict)

	print('\n\nPrecision: ', precision_score(Y_test, y_predict))
	print('Accuracy; ', accuracy_score(Y_test, y_predict))
	print('sensibility: ', recall_score(Y_test, y_predict))
	print('puntaje f1 del modelo: ', f1_score(Y_test, y_predict))	

	#print("\nTest Data Set\n",Y_test)
	#print("\n",y_predict)
	tnts, fpts, fnts, tpts = confusion_matrix(Y_test, y_predict).ravel()
	print("\nVerdaderos Positivos: ",tpts)
	print("Falsos Positivos: ",fpts)
	print("Verdaderos Negativos: ",tnts)
	print("Falsos Negativos: ",fnts)

# K N E A R E S T  N E I G H B O R S
def KNN():
	# Hiperparametros
	testSize = 0.25
	neighbors = 3

	print("\nTest Size: ",testSize)
	print("Neighbors quantity: ", neighbors)

	# Divido el dataSet para prueba y testeo
	x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(x, y, test_size=testSize)

	escalar = StandardScaler()
	x_train_k = escalar.fit_transform(x_train_k)
	x_test_k = escalar.transform(x_test_k)

	print("\n\nX_train shape: ", x_train_k.shape)
	print("y_train shape: ", y_train_k.shape)
	print("X_test shape: ", x_test_k.shape)
	print("X_test shape: ", y_test_k.shape)

	# Cargo el modelo y encajo el hiperparametro de vecinos
	knn = KNeighborsClassifier(n_neighbors=neighbors)
	# Lo entreno
	knn.fit(x_train_k, y_train_k)
	# Realizo la prediccion
	y_pred_k = knn.predict(x_test_k)

	# Obtengo las metricas obtenidas
	print('\nPrecision: ', precision_score(y_test_k, y_pred_k))
	print('Accuracy; ', accuracy_score(y_test_k, y_pred_k))
	print('sensibility: ', recall_score(y_test_k, y_pred_k))
	print('puntaje f1 del modelo: ', f1_score(y_test_k, y_pred_k))

	# Revisar la eficacia para predecir, las probabilidades de que sea 1 o 0, a mas alto el porcentaje mas posible que sea 1
	""" print("\nTrain Data Set\n",y_train_k)
	train_predict_k = knn.predict(x_train_k)
	print("\n",train_predict_k)
	tntrk, fptrk, fntrk, tptrk = confusion_matrix(y_train_k, train_predict_k).ravel()
	print("\nVerdaderos Positivos: ",tptrk)
	print("Falsos Positivos: ",fptrk)
	print("Verdaderos Negativos: ",tntrk)	
	print("Falsos Negativos: ",fntrk)	 """

	#print("\nTest Data Set\n",y_test_k)
	test_predict_k = knn.predict(x_test_k)
	#print("\n",test_predict_k)
	tntsk, fptsk, fntsk, tptsk = confusion_matrix(y_test_k, test_predict_k).ravel()
	print("\nVerdaderos Positivos: ",tptsk)
	print("Falsos Positivos: ",fptsk)
	print("Verdaderos Negativos: ",tntsk)
	print("Falsos Negativos: ",fntsk)

# N E U R A L  N E T W O R K S
def neural():
	# Hiperparametros
	neuronasOcultas = 8
	learningRate = 0.01
	epocas = 50
	batchSize = 50
	validationSplit = 0.20

	x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(x, y, test_size=0.25)

	print("\nEntrenamiento x: ",x_train_n.shape)
	print("Testeo x: ",x_test_n.shape)
	print("Entrenamiento y: ",y_train_n.shape)
	print("Testeo y: ",y_test_n.shape)

	print("\nNeuronas en la capa oculta: ",neuronasOcultas)
	print("Nivel de aprendizaje: ",learningRate)
	print("Iteraciones totales: ",epocas,"\n")

	scaler = StandardScaler()
	x_train_n = scaler.fit_transform(x_train_n)
	x_test_n = scaler.transform(x_test_n)

	print("\n",x_train_n[3])
	print(y_train_n[3])

	model = tf.keras.models.Sequential()

	"""Capa oculta
	Donde tendremos neuronas de validacion inicialm y una funcion de activacion, en nuestro caso relu que se acopla
	bastante bien al comportamiento que presentara el modelo (donde es lineal de cero en adelante y para los negativos sera cero),
	que se puede acercar al comportamiento de las variables de salida
	"""

	model.add(tf.keras.layers.Dense(units=neuronasOcultas, activation=tf.keras.activations.relu))

	"""Capa de Salida
	Activacion=softmax y loss=categorical_crossentropy cuando hay clasificacion de mas de dos clases
	activacion=sigmoid y loss=binary_crossentropy cuando hay una clasificacion con 1 clase y puede tomar 
	valores binarios (ej: lanzar una moneda, da un resultado (el lado de caida, qeu bien sera cara o sello))
	la estructura de la capa se basa en units(cantidad de neuronas, en este caso sera 1 porque solo hay una salida
	la cual tomara el valor de 0 o 1, y la activacion que sera la funcion con la que trabajaremos, en nuestro caso
	sigmoide es la que mas se acerca)
	"""
	model.add(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid))
	"""La funcion de perdida o loss sera crossentropy para el caso de clasificacion
	el optimizer es la funcion 
	"""
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss='binary_crossentropy',
		metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Accuracy(),tf.keras.metrics.Recall()])
	"""La parte de entrenamiento o fit tomara los valores respectivos de los entrenamientos obtenidos en el split,
	y recibira epochs que son las iteraciones que se realizaran, batch_size como los pedazos del conjunto total de 
	training para hacer pruebas y validation_split como el trozo que sacaremos de esos dataSets de entrenamiento para 
	realizar la evaluacion del modelo
	"""
	model.fit(x_train_n, y_train_n, epochs=epocas, batch_size=batchSize, validation_split=validationSplit)

	# Revisar la eficacia para predecir, las probabilidades de que sea 1 o 0, a mas alto el porcentaje mas posible que sea 1
	print("\nTrain Data Set\n",y_train_n)
	train_predict_n = model.predict(x_train_n)
	print("\n",train_predict_n[0:10,:])

	print("\nTest Data Set\n",y_test_n)
	test_predict_n = model.predict(x_test_n)
	print("\n",test_predict_n[0:10,:])

#Menu de seleccion
ans=True
while ans:
    print("""
----------------------------------------------------------------------------------------------

    Machine Learning Classifier Selector

    1.Logistic
    2.KNN
    3.neuralNetworks
    4.Exit/Quit
    """)
    ans=input("What would you like to do? ")
    if ans=="1":
      logistic()
    elif ans=="2":
      KNN()
    elif ans=="3":
      neural()
    elif ans=="4":
      ans = None
    else:
       print("\n Opcion no valida")