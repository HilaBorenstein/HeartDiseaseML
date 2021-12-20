import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  ,accuracy_score
from sklearn import metrics
import os



'''
A Support Vector Machine (SVM) is a classifier formally defined by a separating hyperplane. 
In other words, given labeled training data (supervised learning), 
the algorithm outputs an optimal hyperplane which categorizes new examples. 
In two-dimensional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.
'''
#argument that we can change in order to increase model performance- kernel, max iteration

class SVM:

	def __init__(self,log_file, kernel, maxiter,dataset,test_size):
		self.log_file=log_file
		self.kernel = kernel
		self.maxiter = maxiter
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
		if self.kernel=='rbf':
			self.model = SVC(max_iter=self.maxiter,kernel=self.kernel,gamma='auto') 
		else:
			#poly kerel
			self.model = SVC(max_iter=self.maxiter,kernel=self.kernel,degree=3,gamma='auto') 
		self.model.fit(self.X_train, self.y_train)


	############################
	#Predictions
	############################

	def predict(self):
		self.y_pred = self.model.predict(self.X_test)

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
		new_dataset['HeartDiseaseorAttack'] = predicted_y
		#creating the csv file with the new column
		if not os.path.exists("results"):
			os.mkdir("results")
		output_filename="results/Predicted_Results_SVM_for_heart_disease_prediction.csv"
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



	################################################################################################################
	#plot graph for number of neighbors and accuracy in order to choose enough neighbors for good accuracy 
	################################################################################################################

	def check_best_iter (self):
		#try running from k=1000 through 10000 and record testing accuracy
		k_range= range(1000,10001,100)
		scores={}
		scores_list=[]
		for k in k_range:
			#print("K is"+str(k)+";kernel is "+str(self.kernel))
			if self.kernel=='rbf':
				svm = SVC(max_iter=k,kernel=self.kernel,gamma='auto') 
			else:
			#poly kerel
				svm = SVC(max_iter=k,kernel=self.kernel,degree=3,gamma='auto')
			#svm=SVC(max_iter=self.maxiter,kernel=self.kernel,gamma='auto') 
			svm.fit(self.X_train,self.y_train)
			y_pred=svm.predict(self.X_test)
			scores[k]=metrics.accuracy_score(self.y_test,y_pred)
			scores_list.append(metrics.accuracy_score(self.y_test,y_pred))
		plt.plot(k_range,scores_list)
		plt.xlabel('Number of iteration')
		plt.ylabel('Testing Accuracy')
		plt.show() 
