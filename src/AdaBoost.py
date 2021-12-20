import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import os

'''
An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset 
and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly
 classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

The parameters to changed in order to achieve good performance were:
	•	base_estimator: 
		It is a weak learner used to train the model. 
		It uses DecisionTreeClassifier as default weak learner for training purpose. You can also specify different machine learning algorithms.
	•	n_estimators: 
		Number of weak learners to train iteratively. 
		In case of perfect fit, the learning procedure is stopped early.  
	•	learning_rate: 
		Learning rate shrinks the contribution of each classifier by learning_rate. 
		There is a trade-off between learning_rate and n_estimators.
'''

#argument that we can change in order to increase model performance-base_estimator,n_estimators ,learning_rate, algorithm
class AdaBoost:

	def __init__(self, log_file,base_estimator,n_estimators ,learning_rate, algorithm,dataset,test_size):
		self.log_file=log_file
		self.base_estimator=base_estimator
		self.n_estimators  = n_estimators 
		self.learning_rate = learning_rate
		self.algorithm = algorithm
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
		self.model=AdaBoostClassifier(base_estimator=self.base_estimator,n_estimators =self.n_estimators , learning_rate=self.learning_rate, algorithm=self.algorithm)
		self.model.fit(self.X_train, self.y_train)
		#print(X_train.columns)

	############################
	#Predictions
	############################

	def predict(self):
		self.y_pred=self.model.predict(self.X_test)

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
		output_filename="results/Predicted_Results_AdaBoost_for_heart_disease_prediction.csv"
		if os.path.exists(output_filename):
			os.remove(output_filename)
		new_dataset.to_csv(output_filename, index=False)
		#calculating accuracy for the file
		accuracy = accuracy_score(csv_file_y, predicted_y)
		print("Accuracy for  dataset: %.2f%%" % (accuracy * 100.0))




	####################################################################################################################
	#Evaluating the Algorithm (calculating classification_report,confusion_matrix,model Accuracy,)
	####################################################################################################################
	def evaluate(self,label1,label2):
		#writing to screen
		print("\nclassification_report:\n")
		print(classification_report(self.y_test,self.y_pred))  
		#cmtx = pd.DataFrame(confusion_matrix(self.y_test,self.y_pred, labels=[label1, label2]), index=["true:"+label1, "true:"+label2], columns=["pred:"+label1, "pred:"+label2])
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


	############################
	#	features_importance
	############################
	def features_importance(self):
		feature_importances = pd.DataFrame(self.model.feature_importances_,index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False) 
		print ("\n"+str(feature_importances)+'\n')
		#print results to log file
		self.log_file.write("\n\nE. Features Importance\n")
		fm=feature_importances.to_string()
		for line in fm.split("\n"):
			self.log_file.write("\n\t"+str(line))


	################################################################################################
	#plot graph for number of trees and accuracy in order to choose enough trees for good accuracy 
	################################################################################################
	def check_best_n_estimators (self):
		#try running from k=1 through 850 with jump of 50 and record testing accuracy
		k_range= range(50,1001,100)
		scores={}
		scores_list=[]
		for k in k_range:
			model=self.model=AdaBoostClassifier(n_estimators =k , learning_rate=self.learning_rate, algorithm=self.algorithm)
			model.fit(self.X_train,self.y_train)
			y_pred=model.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			print("n_estimators is "+ str(k) +", accuracy ,"+str(metrics.accuracy_score(self.y_test,y_pred))+"\n")
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('Number of trees with learning rate '+str(self.learning_rate))
		plt.ylabel('Testing Accuracy')
		plt.show() 

	################################################################################################
	#plot graph for learning rate and accuracy in order to choose best learning rate
	################################################################################################

	def check_best_learning_rate (self):
		#try running from k=0.1 through 1 and record testing accuracy
		k_range= np.arange(0.1,1.1,0.1)
		scores={}
		scores_list=[]
		for k in k_range:
			model=self.model=AdaBoostClassifier(n_estimators =self.n_estimators , learning_rate=k, algorithm=self.algorithm)
			model.fit(self.X_train,self.y_train)
			y_pred=model.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			print("learning_rate is "+ str(k) +", accuracy ,"+str(metrics.accuracy_score(self.y_test,y_pred))+"\n")
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('learning rate with number of trees of '+str(self.n_estimators))
		plt.ylabel('Testing Accuracy')
		plt.show() 

		