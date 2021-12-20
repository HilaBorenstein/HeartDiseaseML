import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
from sklearn import metrics
import os
import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO 
from IPython.display import Image
from sklearn.tree import export_graphviz

'''
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences.
I used Decision Tree from scikit-learn and modified it per my requirement.
The features evaluation is built in function.
The parameters to changed in order to achieve good performance were:
	•	Criterion- The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
	•	max_depth- The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
	•	min_samples_split- The minimum number of samples required to split an internal node

'''

#argument that we can change in order to increase model performance- criterion,max_depth,min_samples_split

class DecisionTree:

	def __init__(self, log_file,criterion,max_depth,min_samples_split,dataset,test_size):
		self.log_file=log_file
		self.criterion = criterion
		self.max_depth = max_depth
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
		self.model=tree.DecisionTreeClassifier(criterion =self.criterion, min_samples_split=self.min_samples_split)
		#Fit Model on training data
		self.model.fit(self.X_train, self.y_train)

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
		output_filename="results/Predicted_Results_Decision_Tree_for_heart_disease_prediction.csv"
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


	def plot_tree(self,label1,label2):
		dot_data = StringIO()
		tree.export_graphviz(self.model, out_file=dot_data,  class_names=[label1,label2], # the target names.
		 feature_names=self.X_train.columns,#self.dataset.drop('HeartDiseaseorAttack', axis=1)  , # the feature names.
		 filled=True, # Whether to fill in the boxes with colours.
		 rounded=True, # Whether to round the corners of the boxes.
		 special_characters=True)
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
		Image(graph.create_png())
		graph.write_png(r"results\DecisionTree_"+label1+"VS"+label2+".png")

	############################################################################################################
	#plot graph for maximum depth and accuracy in order to choose enough maximum depth for good accuracy 
	############################################################################################################

	def check_best_max_depth (self):
		#split
		#try running from k=1 through 25 and record testing accuracy
		k_range= range(1,26)
		scores={}
		scores_list=[]
		for k in k_range:
			clf=tree.DecisionTreeClassifier(criterion =self.criterion,max_depth=k, min_samples_split=self.min_samples_split)
			clf.fit(self.X_train,self.y_train)
			y_pred=clf.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('Value of max depth for Decision tree')
		plt.ylabel('Testing Accuracy')
		plt.show() 
