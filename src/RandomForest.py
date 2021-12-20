import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
import os

'''
Random forest is a classifier that evolves from decision trees. It consists of many decision trees. To classify a new instance, each decision tree provides a classification for input data; random forest collects the classifications and chooses the most voted prediction as the result. The input of each tree is sampled data from the original dataset. In addition, a subset of features is randomly selected from the optional features to grow the tree at each node. Each tree is grown without pruning. Essentially, random forest enables a large number of weak or weakly-correlated classifiers to form a strong classifier.
I used Random forest from scikit-learn and modified it per my requirement.
The features evaluation is built in function.
The parameters to changed in order to achieve good performance were:
•	n_estimators- The number of trees in the forest.
•	max_features- The number of features to consider when looking for the best split.
•	max_depth- The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
•	min_samples_split- The minimum number of samples required to split an internal node
'''
#argument that we can change in order to increase model performance- n_estimators,max_depth, max_features,min_samples_split

class RandomForest:
	#argument that we want to change- n_estimators,max_depth, max_features,min_samples_split,
	def __init__(self, log_file,n_estimators,max_depth, max_features,min_samples_split,dataset,test_size):
		self.log_file=log_file
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.max_features = max_features
		self.min_samples_split=min_samples_split
		self.dataset=dataset
		self.test_size=test_size


	####################################################################################
	#Data Preprocessing-reading files ,dividing the dataset into attributes and labels
	####################################################################################
	def preProcessing (self):
		X = self.dataset.drop('HeartDiseaseorAttack', axis=1)  
		y = self.dataset['HeartDiseaseorAttack']	
		#split
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size =self.test_size)

	############################
	#  Training the Algorithm
	############################

	def train(self):
		#Creating the model
		self.model = RandomForestClassifier(n_estimators=self.n_estimators , max_depth=self.max_depth,max_features=self.max_features,min_samples_split=self.min_samples_split)
		#Fit Model on training data
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
	def predict_to_file(self,csv_file,algorithm):
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
		if (algorithm=="rf"):
			output_filename="results/Predicted_Results_Random_Forest_for_heart_disease_prediction.csv"
		elif (algorithm=="bagging"):
			output_filename="results/Predicted_Results_Bagging_for_heart_disease_prediction.csv"
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
	#features_importance
	############################
	def features_importance(self):
		feature_importances = pd.DataFrame(self.model.feature_importances_,index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False) 
		print ("\n"+str(feature_importances)+'\n')
		#print results to log file
		self.log_file.write("\n\nE. Features Importance\n")
		fm=feature_importances.to_string()
		for line in fm.split("\n"):
			self.log_file.write("\n\t"+str(line))

	#############################################################################################################
	#plot graph for number of trees and accuracy in order to choose enough trees for good accuracy 
	#############################################################################################################

	def check_best_n_estimators (self):
		#try running from k=1 through 1001 and record testing accuracy
		k_range= range(1,1501,100)
		scores={}
		scores_list=[]
		for k in k_range:
			rf= RandomForestClassifier(n_estimators=k, max_depth=self.max_depth,max_features=self.max_features,min_samples_split=self.min_samples_split)
			rf.fit(self.X_train,self.y_train)
			y_pred=rf.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			print("n estimators are"+ str(k)+",accuracy ,"+str(metrics.accuracy_score(self.y_test,y_pred))+"\n")
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('Number of trees')
		plt.ylabel('Testing Accuracy')
		plt.show() 

	#############################################################################################################
	#plot graph for max_features and accuracy in order to choose max_features for good accuracy 
	#############################################################################################################

	def check_best_max_features (self):
		#try running from k=1 through 17 and record testing accuracy
		k_range= range(1,17)
		scores={}
		scores_list=[]
		for k in k_range:
			rf= RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,max_features=k,min_samples_split=self.min_samples_split)
			rf.fit(self.X_train,self.y_train)
			y_pred=rf.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			print("max_features is "+ str(k)+",accuracy ,"+str(metrics.accuracy_score(self.y_test,y_pred))+"\n")
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('Number of features')
		plt.ylabel('Testing Accuracy')
		plt.show() 

