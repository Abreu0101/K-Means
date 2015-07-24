import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet():
	path = "data_iris.csv"
	df = pd.read_csv(path,header=None)[::][[0,1]]
	return df

def plotDataFrame(dataFrame):
	ax = dataFrame.plot(kind='scatter',x=0,y=1,title="Iris Dataset")
	ax.set_xlabel("Sepal length in cm")
	ax.set_ylabel("Sepal width in cm")
	#plt.scatter(dataFrame[::][[0]],dataFrame[::][[1]])
	plt.show()

def plotClusterData(dataFrame,clusters):
	ax = None
	for cluster in clusters:
		if cluster:
			color = tuple(np.random.rand(4))
			if ax is None:
				ax = dataFrame[dataFrame.index.isin(cluster)].plot(kind='scatter',x=0,y=1,ax=ax,color=color)
			else:
				dataFrame[dataFrame.index.isin(cluster)].plot(kind='scatter',x=0,y=1,ax=ax,color=color)
	plt.show()
	

def distance(observation,centroide):
	return np.sqrt(np.sum((observation - centroide)**2))

def kMeans(dataFrame,k=3):
	dataset = dataFrame.as_matrix()
	#Crear k cantidad de centroides randoms, cada uno con la misma dimension que el dataset (columnas o features)
	centroides = [[np.random.randint(np.min(dataset[:,feature_index]),np.max(dataset[:,feature_index])) for feature_index in range(len(dataset[0]))] for _ in range(k)]
	print(centroides)
	#Crear k Cluster vacios
	cluster = [[] for _ in range(k)]

	#Pos_tmp para luego verifica si hubo cambio de cluster y hasChanged Boolean
	pos_tmp = dict()
	hasChanged = True

	while hasChanged:
		hasChanged = False
		for pos in range(len(dataset)):
			observation = dataset[pos]

			#Calcular distancia entre la observacion y el centroide
			distances = [distance(observation,centroide) for centroide in centroides]
			min_distance = min(distances)
			min_index = distances.index(min_distance)
			#print("Indice Menor : %d"%min_index)
			
			#Asignar observacion a cluster
			if pos in pos_tmp:
				if pos_tmp[pos] != min_index:
					hasChanged = True
					pos_prev = pos_tmp[pos]
					cluster[pos_prev].remove(pos)
					cluster[min_index].append(pos)
					pos_tmp[pos] = min_index 
			else:
				pos_tmp[pos] = min_index
				cluster[min_index].append(pos)
				hasChanged = True

		#Reorganizar Clusters
		if hasChanged:
			for centroide_index in range(k):
				avg_pos = np.zeros((1,len(dataset[0])))
				for observation_index in cluster[centroide_index]:
					avg_pos += dataset[observation_index]
				avg_pos /= len(cluster[centroide_index])
				centroides[centroide_index] = avg_pos

	#plot color cluster
	plotClusterData(dataFrame,cluster)

	#Return Cluster de posiciones de observaciones
	return cluster
