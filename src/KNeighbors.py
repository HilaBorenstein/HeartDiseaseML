import pandas as pd  
#import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
from sklearn import metrics
import os

'''
The principle behind nearest neighbor methods is to find a predefined number of training samples
 closest in distance to the new point and predict the label from these. 
 The number of samples can be a user-defined constant (k-nearest neighbor learning) or vary based
 on the local density of points (radius-based neighbor learning). The distance can, in general, 
 be any metric measure: standard Euclidean distance is the most common choice. 

The parameters to changed in order to achieve good performance were:
•	N_neighbors- Number of neighbors to use.
•	weights: weight function used in prediction. 
	Possible values:
	o	‘uniform’: uniform weights. All points in each neighborhood are weighted equally.
	o	‘distance’: weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
	o	[callable]: a user-defined function which accepts an array of distances and returns an array of the same shape containing the weights.
'''

##rgument that we can change in order to increase model performance- n_neighbors,weights, random_state,leaf_size

class K_NearestNeighbors:

	def __init__(self, log_file,n_neighbors,weights ,dataset,test_size):
		self.log_file=log_file
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.dataset=dataset
		self.test_size=test_size


	####################################################################################
	#Data Preprocessing-reading files ,dividing the dataset into attributes and labels
	####################################################################################
	def preProcessing (self):
		#print("\nDataset Dimension is: "+ str(self.dataset.shape) +"\n") 
		X = self.dataset.drop('HeartDiseaseorAttack', axis=1)  
		y = self.dataset['HeartDiseaseorAttack']
		
		#split
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size =self.test_size)

	############################
	#  Training the Algorithm
	############################

	def train(self):
		#Creating the model
		self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors,weights=self.weights)
		#Fit Model on training data
		self.model.fit(self.X_train, self.y_train)


	############################
	#Predictions
	############################

	def predict(self):
		self.y_pred=self.model.predict(self.X_test)
		#print(self.model.score(self.X_test, self.y_test))

	######################################################################################################
	#	create csv file with the original observations file the real class and the predicted class
	######################################################################################################
	def predict_to_file(self,csv_file):
		#new_dataset= pd.read_csv(csv_file)
		new_dataset=csv_file
		csv_file_X = new_dataset.drop('HeartDiseaseorAttack', axis=1)  
		csv_file_y = new_dataset['HeartDiseaseorAttack']
		predicted_y=self.model.predict(csv_file_X)
		#creating the new column
		new_dataset['Predicted_Class'] = predicted_y
		#creating the csv file with the new column
		if not os.path.exists("results"):
			os.mkdir("results")
		output_filename="results/Predicted_Results_K_Nearest_Neighbors_for_heart_disease_prediction.csv"
		if os.path.exists(output_filename):
			os.remove(output_filename)
		new_dataset.to_csv(output_filename, index=False)
		#calculating accuracy for the file
		accuracy = accuracy_score(csv_file_y, predicted_y)
		print("Accuracy for dataset: %.2f%%" % (accuracy * 100.0))


	####################################################################################################################
	#Evaluating the Algorithm (calculating classification_report,confusion_matrix,model Accuracy,)
	####################################################################################################################
	def evaluate(self,label1, label2):
		#writing to screen
		print("\nclassification_report:\n")
		print(classification_report(self.y_test,self.y_pred))  
		#cm=confusion_matrix(self.y_test,self.y_pred, labels=[label1, label2]), index=["true:"+label1, "true:"+label2], columns=["pred:"+label1, "pred:"+label2])
		cm=confusion_matrix(self.y_test,self.y_pred)
		cmtx = pd.DataFrame(cm)

		print("confusion_matrix:\n")
		print(cmtx)
		accuracy = accuracy_score(self.y_test, self.y_pred)
		print("\nAccuracy: %.2f%%" % (accuracy * 100.0))
		#print results to log file
		self.log_file.write("\nD. Model Results\n")
		self.log_file.write("\n\ta. Classification Report\n")
		cr = classification_report(self.y_test,self.y_pred)
		cr1=str(cr)
		for line in cr1.split("\n"):
			self.log_file.write("\n\t\t"+str(line))
		self.log_file.write("\n\tb. Confusion Matrix\n")
		cmtx1=cmtx.to_string()
		for line in cmtx1.split("\n"):
			self.log_file.write("\n\t\t"+str(line))
		self.log_file.write("\n\n\tc. Model accuracy: %.2f%%" % (accuracy * 100.0))



	################################################################################################################
	#plot graph for number of neighbors and accuracy in order to choose enough neighbors for good accuracy 
	################################################################################################################

	def check_best_n (self):
		#try running from k=1 through 100 and record testing accuracy
		k_range= range(1,101)
		scores={}
		scores_list=[]
		for k in k_range:
			knn=KNeighborsClassifier(n_neighbors=k)
			knn.fit(self.X_train,self.y_train)
			y_pred=knn.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))

		plt.plot(k_range,scores_list)
		plt.xlabel('Value of K for KNN')
		plt.ylabel('Testing Accuracy')
		plt.show() 
		