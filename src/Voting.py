import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import os


'''
Voting is combining the predictions from multiple machine learning algorithms. 
Voting classifier is a wrapper for set of different ones that are trained and 
valuated in parallel in order to exploit the different peculiarities of each algorithm
'''

class Voting:

	def __init__(self,log_file,clf1,clf2,clf3  ,dataset,test_size):
		self.log_file=log_file
		self.clf1 = clf1
		self.clf2 = clf2
		self.clf3 = clf3
		self.dataset=dataset
		self.test_size=test_size


	####################################################################################
	#Data Preprocessing-reading files ,dividing the dataset into attributes and labels
	####################################################################################
	def preProcessing (self):
		self.X = self.dataset.drop('HeartDiseaseorAttack', axis=1)  
		self.y = self.dataset['HeartDiseaseorAttack']
		#split
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size =self.test_size)

	#######################################################################################################
	#  Training the Algorithm- all the parameters already set per my investigation for best parameters
	#######################################################################################################

	def train(self):

		#trianing first clasiffier
		if (self.clf1=="KNN"):
			self.model1=KNeighborsClassifier(n_neighbors=15,weights="distance")
		elif (self.clf1=="SVM"):
			self.model1=self.svclassifier = SVC(max_iter=5000,kernel="rbf",gamma='auto') 
		elif(self.clf1=="DT"):
			self.model1=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
		elif (self.clf1=="RF"):
			self.model1 = RandomForestClassifier(n_estimators=800, max_depth=8,max_features=17,min_samples_split=2)
		elif (self.clf1=="Bagging"):
			self.model1 = RandomForestClassifier(n_estimators=800, max_depth=None,max_features=None,min_samples_split=2)
		elif (self.clf1=="AdaBoost"):
			base_estimator=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
			self.model1=AdaBoostClassifier(base_estimator=base_estimator,n_estimators =250 , learning_rate=0.3)
		elif (self.clf1=="GBoost"):
			self.model1=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,min_samples_split=2,max_depth=7, max_features =None )
		elif (self.clf1=="XGBoost"):
			self.model1=XGBClassifier( learning_rate=0.1, n_estimators=100,max_depth=7,max_features =None )
		else:
			print("\nERROR! Wrong Model Name! ")
			sys.exit(0)
		#trianing second clasiffier
		if (self.clf2=="KNN"):
			self.model2=KNeighborsClassifier(n_neighbors=15,weights="distance")
		elif (self.clf2=="SVM"):
			self.model2=self.svclassifier = SVC(max_iter=5000,kernel="rbf",gamma='auto') 
		elif(self.clf2=="DT"):
			self.model2=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
		elif (self.clf2=="RF"):
			self.model2 = RandomForestClassifier(n_estimators=800, max_depth=8,max_features=17,min_samples_split=2)
		elif (self.clf2=="Bagging"):
			self.model2 = RandomForestClassifier(n_estimators=800, max_depth=None,max_features=None,min_samples_split=2)
		elif (self.clf2=="AdaBoost"):
			base_estimator=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
			self.model2=AdaBoostClassifier(base_estimator=base_estimator,n_estimators =250 , learning_rate=0.3)
		elif (self.clf2=="GBoost"):
			self.model2=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,min_samples_split=2,max_depth=7, max_features =None )
		elif (self.clf2=="XGBoost"):
			self.model2=XGBClassifier( learning_rate=0.1, n_estimators=100,max_depth=7,max_features =None )
		else:
			print("\nERROR! Wrong Model Name! ")
			sys.exit(0)
		#trianing third clasiffier
		if (self.clf3=="KNN"):
			self.model3=KNeighborsClassifier(n_neighbors=15,weights="distance")
		elif (self.clf3=="SVM"):
			self.model3=self.svclassifier = SVC(max_iter=5000,kernel="rbf",gamma='auto') 
		elif(self.clf3=="DT"):
			self.model3=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
		elif (self.clf3=="RF"):
			self.model3 = RandomForestClassifier(n_estimators=800, max_depth=8,max_features=17,min_samples_split=2)
		elif (self.clf3=="Bagging"):
			self.model3 = RandomForestClassifier(n_estimators=800, max_depth=None,max_features=None,min_samples_split=2)
		elif (self.clf3=="AdaBoost"):
			base_estimator=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
			self.model3=AdaBoostClassifier(base_estimator=base_estimator,n_estimators =250 , learning_rate=0.3)
		elif (self.clf3=="GBoost"):
			self.model3=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,min_samples_split=2,max_depth=7, max_features =None )
		elif (self.clf3=="XGBoost"):
			self.model3=XGBClassifier( learning_rate=0.1, n_estimators=100,max_depth=7,max_features =None )
		else:
			print("\nERROR! Wrong Model Name! ")
			sys.exit(0)



		self.model1.fit(self.X_train, self.y_train)
		self.model2.fit(self.X_train, self.y_train)
		self.model3.fit(self.X_train, self.y_train)
		#make ensemble classifcation
		self.ensemble_model=VotingClassifier(estimators=[(self.clf1, self.model1), (self.clf2, self.model2), (self.clf3, self.model3)], voting='hard')

		#Fit Model on training data
		self.ensemble_model.fit(self.X_train, self.y_train)
		'''
		for clf, label in zip([self.model1, self.model2, self.model3, self.ensemble_model], [self.clf1, self.clf2 ,self.clf3, 'Ensemble_Model']):
			scores = cross_val_score(clf, self.X, self.y, cv=5, scoring='accuracy')
			print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
		'''


	############################
	#		Predictions
	############################

	def predict(self):
		self.y_pred=self.ensemble_model.predict(self.X_test)

		self.y_pred1=self.model1.predict(self.X_test)
		self.y_pred2=self.model2.predict(self.X_test)
		self.y_pred3=self.model3.predict(self.X_test)
		#print(self.model.score(self.X_test, self.y_test))


	def features_importance(self):
		a=1
		#feature_importances = pd.DataFrame(self.model.feature_importances_,index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False) 
		#print (feature_importances)

	######################################################################################################
	#	create csv file with the original observations file the real class and the predicted class
	######################################################################################################
	def predict_to_file(self,csv_file):
		#new_dataset= pd.read_csv(csv_file)
		new_dataset=csv_file
		csv_file_X = new_dataset.drop('HeartDiseaseorAttack', axis=1)  
		csv_file_y = new_dataset['HeartDiseaseorAttack']
		predicted_y=self.ensemble_model.predict(csv_file_X)
		#creating the new column
		new_dataset['Predicted_Class'] = predicted_y
		#creating the csv file with the new column
		if not os.path.exists("results"):
			os.mkdir("results")
		output_filename="results/Predicted_Results_Voting_for_heart_disease_prediction.csv"
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
		accuracy = accuracy_score(self.y_test, self.y_pred)
		accuracy1 = accuracy_score(self.y_test, self.y_pred1)
		accuracy2 = accuracy_score(self.y_test, self.y_pred2)
		accuracy3 = accuracy_score(self.y_test, self.y_pred3)
		#writing to screen
		print("\nclassification_report:\n")
		print(classification_report(self.y_test,self.y_pred))  
		print("confusion_matrix:\n")
		#cmtx = pd.DataFrame(confusion_matrix(self.y_test,self.y_pred, labels=[label1, label2]), index=["true:"+label1, "true:"+label2], columns=["pred:"+label1, "pred:"+label2])
		cm=confusion_matrix(self.y_test,self.y_pred)
		cmtx = pd.DataFrame(cm)
		print(cmtx)
		print("\nTotal Accuracy: %.2f%%" % (accuracy * 100.0))
		#for first clasiffier
		print("Model: "+str(self.clf1)+", Accuracy: %.2f%%" % (accuracy1 * 100.0))
		#for second clasiffier
		print("Model: "+str(self.clf2)+", Accuracy: %.2f%%" % (accuracy2 * 100.0))
		#for third clasiffier
		print("Model: "+str(self.clf3)+", Accuracy: %.2f%%" % (accuracy3 * 100.0))
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
		self.log_file.write("\n\n\tc. Model accuracy:\n" )
		self.log_file.write("\n\t\tTotal Accuracy: %.2f%%" % (accuracy * 100.0))
		#for first clasiffier
		self.log_file.write("\n\t\tModel: "+str(self.clf1)+", Accuracy: %.2f%%" % (accuracy1 * 100.0))
		#for second clasiffier
		self.log_file.write("\n\t\tModel: "+str(self.clf2)+", Accuracy: %.2f%%" % (accuracy2 * 100.0))
		#for third clasiffier
		self.log_file.write("\n\t\tModel: "+str(self.clf3)+", Accuracy: %.2f%%" % (accuracy3 * 100.0))
