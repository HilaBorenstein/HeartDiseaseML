import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
#from sklearn.metrics import classification_report, confusion_matrix  
import time,sys,os,argparse
from FeatureSelection import *
from KNeighbors import K_NearestNeighbors
from SVM import SVM
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from AdaBoost import AdaBoost
from GBoost import GBoost
from XGBoost import XGBoost
from Voting import Voting
from sklearn import tree
import warnings 




warnings.filterwarnings('ignore')


##################################################################################
#					feature selection
##################################################################################

def feature_selection (dataset,log_file):
	labels= dataset['HeartDiseaseorAttack']
	#print(dataset.head())
	fs= FeatureSelector(log_file,dataset,labels)
	#~~~~~~~~missing values~~~~~~~~~~~~~~
	missing_threshold=0.6
	fs.identify_missing(missing_threshold)
	missing = fs.ops['missing']
	print(missing)
	#fs.plot_missing()
	#~~~~~~~find single unique value~~~~~
	fs.identify_single_unique()
	single_unique = fs.ops['single_unique']
	print(single_unique)
	#fs.plot_unique()
	#~~find Collinear (highly correlated) Features~~
	correlation_threshold=0.975
	fs.identify_collinear(correlation_threshold)
	correlated_features = fs.ops['collinear']
	print(correlated_features)
	#fs.plot_collinear()
	#fs.plot_collinear(plot_all=True)
	
	#~~~Find Zero Importance Features:~~~~~~~~~~~~~
	fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
	zero_importance_features = fs.ops['zero_importance']
	print(zero_importance_features)
	one_hot_features = fs.one_hot_features
	base_features = fs.base_features
	#print('There are %d original features' % len(base_features))
	#print('There are %d one-hot features' % len(one_hot_features))
	#~~~~~~~~~~~~~~~ low Importance Features~~~~~~~
	#fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
	fs.feature_importances.head(10)
	#Low Importance Features
	fs.identify_low_importance(cumulative_importance = 0.99)
	low_importance_features = fs.ops['low_importance']
	print(low_importance_features)
	#'''
	#~~~~~~~~~~~~~~~ Removing Features~~~~~~~~~~~~~
	'''
	all_to_remove = fs.check_removal()
	print(all_to_remove)
	'''
	#train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)
	train_removed_all = fs.remove(methods = ['missing','single_unique','collinear'], keep_one_hot=False)
	#options['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
	print('Original Number of Features', dataset.shape[1])
	print('Final Number of Features: ', train_removed_all.shape[1])
	#print(train_removed_all.head())
	#fs.identify_all(selection_params = {'missing_threshold': 0.5, 'correlation_threshold': 0.7, 'task': 'regression', 'eval_metric': 'l2', 'cumulative_importance': 0.9})
	return train_removed_all

##################################################################################
#								K_NearestNeighbors
##################################################################################
def KNN_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#setting the best values i found in order to acheive the best accuracy
	n_neighbors = 15
	weights="distance"
	#weights="uniform"
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: K Nearest Neighbors\nNumber of neighbors: "+str(n_neighbors) +", weights: "+ weights +"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:K Nearest Neighbors\n\tNumber of neighbors: "+str(n_neighbors) +"\n\tWeights: "+ str(weights)+"\n")
	myKNN = K_NearestNeighbors(log_file,n_neighbors,weights ,dataset,test_size)
	myKNN.preProcessing()
	#myKNN.check_best_n()
	myKNN.train()
	myKNN.predict()
	myKNN.predict_to_file(dataset)
	myKNN.evaluate(label1,label2)


##################################################################################
#							svm evaluate
##################################################################################

def SVM_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	kernel="rbf"
	#kernel="poly"
	maxiter=50000
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: Support Vectors Machine\nKernel: "+str(kernel) +", Maximum iterations: "+ str(maxiter) +"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Support Vectors Machine\n\tKernel: "+str(kernel) +"\n\tMaximum iterations: "+ str(maxiter)+"\n")
	mySVM = SVM(log_file,kernel,maxiter,dataset,test_size)
	mySVM.preProcessing()
	#mySVM.check_best_iter()
	mySVM.train()
	mySVM.predict()
	mySVM.predict_to_file(dataset)
	mySVM.evaluate(label1,label2)


##################################################################################
#									Decision Tree
##################################################################################
def DT_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#setting the best values i found in order to acheive the best accuracy
	criterion = "entropy"#"entropy"#"gini"#"entropy"
	max_depth=8
	min_samples_split=2
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: Decision Tree\ncriterion: "+str(criterion) +", Maximum depth: "+ str(max_depth) +", Minimum samples for splitting node: "+str(min_samples_split)+"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Decision Tree\n\tCriterion: "+str(criterion) +"\n\tMinimum samples for splitting node: "+ str(min_samples_split)+"\n")
	#running the model
	myDT =DecisionTree(log_file,criterion,max_depth,min_samples_split,dataset,test_size)
	myDT.preProcessing()
	#myDT.check_best_max_depth()
	
	myDT.train()
	myDT.predict()
	myDT.predict_to_file(dataset)
	myDT.evaluate(label1,label2)
	myDT.features_importance()
	'''
	if (dataset_size<300000):
		myDT.plot_tree(label1,label2)
	'''
	

##################################################################################
#							Random Forest evaluate
##################################################################################
def RF_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	algorithm="rf" # this is string to distinguish between bagging and randeom forest
	#setting the best values i found in order to acheive the best accuracy
	max_features=17
	n_estimators = 800
	max_depth=8
	min_samples_split=2
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: Random Forest\nNumber of trees: "+str(n_estimators)+", Maximum depth for each tree: "+ str(max_depth) +", Minimum samples for splitting node: "+str(min_samples_split)+", Maximum features used for each tree: "+str(max_features)+"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Random Forest\n\tNumber of trees: "+str(n_estimators) +"\n\tMinimum samples for splitting node: "+ str(min_samples_split)+"\n\tMaximum features used for each tree: "+ str(max_features)+"\n")
	#running the model
	myRandForest = RandomForest(log_file,n_estimators,max_depth,max_features,min_samples_split,dataset,test_size)
	myRandForest.preProcessing()
	#myRandForest.check_best_n_estimators()
	#myRandForest.check_best_max_features()
	myRandForest.train()
	myRandForest.predict()
	myRandForest.predict_to_file(dataset,algorithm)
	myRandForest.evaluate(label1,label2)
	myRandForest.features_importance()
	

##################################################################################
#							Bagging evaluate
##################################################################################
def Bagging_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	algorithm="bagging" # this is string to distinguish between bagging and randeom forest
	#Special case of Random Forest, all features are considered for splitting a node.
	n_estimators =800
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: Bagging\nNumber of trees: "+str(n_estimators)+" (all Features are used to develop each tree)\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Bagging\n\tNumber of trees: "+str(n_estimators)+"\n")
	#running the model
	Bagging = RandomForest(log_file,n_estimators,None,None,2,dataset,test_size)
	Bagging.preProcessing()
	#Bagging.check_best_n_estimators()
	Bagging.train()
	Bagging.predict()
	Bagging.predict_to_file(dataset,algorithm)
	Bagging.evaluate(label1,label2)
	Bagging.features_importance()


##################################################################################
#							AdaBoost_Model
##################################################################################
def AdaBoost_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#setting the best values i found in order to acheive the best accuracy
	base_estimator=tree.DecisionTreeClassifier(criterion ="entropy",  min_samples_split=2)
	learning_rate=0.3
	n_estimators=250
	#printing model information to screen
	algorithm   ="SAMME" # {‘SAMME’, ‘SAMME.R’}
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size) )
	print("Model Name: Adaboost\nNumber of trees: "+str(n_estimators)+", learning rate: "+ str(learning_rate)+"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:AdaBoost\n\tNumber of trees: "+str(n_estimators) +"\n\tlearning rate: "+ str(learning_rate))
	log_file.write("\n\tMinimum samples for splitting node: 2\n")
	#running the model
	myAdaBoost = AdaBoost(log_file,base_estimator,n_estimators ,learning_rate,algorithm  ,dataset,test_size)
	myAdaBoost.preProcessing()
	#the following 2 darked lines is in order to find best parameteres for the models
	#myAdaBoost.check_best_n_estimators()
	#myAdaBoost.check_best_learning_rate()
	myAdaBoost.train()
	myAdaBoost.predict()
	myAdaBoost.predict_to_file(dataset)
	myAdaBoost.evaluate(label1,label2)
	myAdaBoost.features_importance()

##################################################################################
#									GBoost_Model
##################################################################################
def GBoost_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#setting the best values i found in order to acheive the best accuracy
	learning_rate=0.1
	n_estimators=100
	min_samples_split=2
	max_depth = 7
	max_features  =None
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size))
	print("Model Name:Gradient Boosting\nNumber of boosting stages: "+str(n_estimators) +", learning rate: "+ str(learning_rate) +", Minimum samples for splitting node: "+str(min_samples_split))
	print("Maximum depth: "+str(max_depth)+", Maximum features to use every node: "+str(max_features)+"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Gradient Boosting\n\tNumber of boosting stages: "+str(n_estimators) +"\n\tlearning rate: "+ str(learning_rate))
	log_file.write("\n\tMinimum samples for splitting node: "+str(min_samples_split)+"\n\tMaximum depth: "+ str(max_depth)+"\n")
	#running the model
	myGBoost = GBoost(log_file,learning_rate,n_estimators,min_samples_split,max_depth,max_features  ,dataset,test_size)
	myGBoost.preProcessing()
	#the following 2 darked lines is in order to find best parameteres for the models
	#myGBoost.check_best_n_estimators()
	#myGBoost.check_best_learning_rate()
	myGBoost.train()
	myGBoost.predict()
	myGBoost.predict_to_file(dataset)
	myGBoost.evaluate(label1,label2)
	myGBoost.features_importance()
	
##################################################################################
#									XGBoost_Model
##################################################################################

def XGBoost_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#setting the best values i found in order to acheive the best accuracy

	learning_rate=0.1
	n_estimators=100
	max_depth = 7
	max_features  =None

	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size))
	print("Model Name:Extreme Gradient Boosting\nNumber of boosting stages: "+str(n_estimators) +", learning rate: "+ str(learning_rate) )
	print("Maximum depth: "+str(max_depth)+", Maximum features to use every node: "+str(max_features)+"\n 		***")
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Extreme Gradient Boosting\n\tNumber of boosting stages: "+str(n_estimators) +"\n\tlearning rate: "+ str(learning_rate))
	log_file.write("\n\tMaximum depth: "+str(max_depth)+"\n")
	#running the model
	myXGBoost = XGBoost(log_file,learning_rate,n_estimators,max_depth,max_features  ,dataset,test_size)
	myXGBoost.preProcessing()
	#the following 2 darked lines is in order to find best parameteres for the models
	#myXGBoost.check_best_n_estimators()
	#myXGBoost.check_best_learning_rate()
	myXGBoost.train()
	myXGBoost.predict()
	myXGBoost.predict_to_file(dataset)
	myXGBoost.evaluate(label1,label2)
	myXGBoost.features_importance()

	
##################################################################################
#								Voting_Model
##################################################################################
def Voting_Model(label1,label2,log_file,dataset,test_size,dataset_size):
	#first="KNN"
	#first="AdaBoost"
	first="RF"
	#second="DT"
	#second="RF"
	#second="SVM"
	#second="GBoost"
	second="AdaBoost"
	third="XGBoost"
	#printing model information to screen
	print("		***\nModel Information:\nAmount of observations: "+str(dataset_size)+", Test size: "+str(test_size))
	print("Model Name:Voting \nModels Names: "+str(first) +", " +str(second)+", "+str(third))
	#printing model information to log file
	log_file.write("\nC. Model Information\n")
	log_file.write("\n\tModel Name:Voting\n\tModels Names: "+ str(first) +", "+ str(second)+", "+str(third)+"\n")
	myclf=Voting(log_file,first,second,third ,dataset,test_size)
	myclf.preProcessing()
	myclf.train()
	myclf.predict()
	myclf.predict_to_file(dataset)
	myclf.evaluate(label1,label2)

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
###################################################		Main      ############################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


def main(args):

	##################################################################################
	#								setting arguments
	##################################################################################

	start_time = time.time()

	#receiving the arguments and checking arguments correctness

	Model= args.Model
	if Model not in (["KNN","SVM","DT","RF","Bagging","AdaBoost","GBoost","XGBoost","Voting"]):
	 	print("\nERROR! Wrong Model Name!! Please type one of the following: KNN, SVM, DT, RF, Bagging, AdaBoost, GBoost, XGBoost, Voting")
	 	sys.exit(0)
	

	test_size=args.Test_size
	if (test_size<0.1 or test_size>0.3):
	 	print("\nERROR! Wrong test size!! Please choose number between 0.1 to 0.3")
	 	sys.exit(0)


	#this code is for myself if i want to change the arguments
	'''
		
	dataset_size=10000


	#Model="KNN"
	#Model="SVM"
	Model="DT"
	#Model="RF"
	#Model="Bagging"
	#Model="AdaBoost"
	#Model="GBoost"
	#Model="XGBoost"
	#Model="Voting"
	'''

	#test_size=0.2
	#Model="Voting"
	

	##################################################################################
	#					dataset selection and sampling
	##################################################################################
 
	
	label1="HeartDisease"
	label2="NoHeartDisease"
	#label1="1"
	#label2="0"
	dataset_file=pd.read_csv(r"heart_disease_dataset\heart_disease_health_indicators.csv")

	#minutes= float((time.time() - start_time)/60)
	#print("current time ---" +str(minutes)+" minutes ---" )
	
	headers = [*pd.read_csv(r"heart_disease_dataset\heart_disease_health_indicators.csv", nrows=1)]
	dataset = pd.read_csv(r"heart_disease_dataset\heart_disease_health_indicators.csv", usecols=[c for c in headers if c != 'error'])


	dataset_size = len(dataset_file)
	print("dataset_size  ---" +str(dataset_size)+"  ---" )
	##################################################################################
	#					Creating log file for results
	##################################################################################
	#creating the log file of results.
	if not os.path.exists("results"):
		os.mkdir("results")
	log_file_name="results/log_"+str(Model)+"_for_heart_disease_prediction.txt"
	if os.path.exists(log_file_name):
		os.remove(log_file_name)
	log_file=open(log_file_name,"a")
	log_file.write("\nA. General Information\n\n")
	log_file.write("	Amount of observations: "+str(dataset_size)+"\n\tTest size: "+str(test_size)+"\n")

	##################################################################################
	#					feature selection-preprocessing
	##################################################################################
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("~~~~~~~~~~~~Preprocessing- feature selection:~~~~~~~~~~~~~~~~~~~~~")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	train_removed_all=feature_selection (dataset,log_file)
	##################################################################################
	#								activating model 
	##################################################################################
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("~~~~~~~~~~~~Working on model "+Model+ "~~~~~~~~~~~~~~~~~~")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

	if (Model=="KNN"):
		KNN_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="SVM"):
		SVM_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="DT"):
		DT_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="RF"):
		RF_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="Bagging"):
		Bagging_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="AdaBoost"):
		AdaBoost_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="GBoost"):
		GBoost_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="XGBoost"):
		XGBoost_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	elif (Model=="Voting"):
		Voting_Model(label1,label2,log_file,train_removed_all,test_size,dataset_size)
	else:
		print("\nERROR! Wrong Model Name! ")
		sys.exit(0)

	##################################################################################
	#					Time Performance
	##################################################################################
	minutes= float((time.time() - start_time)/60)

	print("Total time ---" +str(minutes)+" minutes ---" )





if __name__ == "__main__":
	"""
	Our deafult vars for now....
	"""
	parser = argparse.ArgumentParser(description='Choose model \n(optional: test size)\n')
	parser.add_argument("--Model", type=str, default="XGBoost",
						help='The model you want to run\n(options:KNN, SVM, DT, RF, Bagging, AdaBoost, GBoost, XGBoost, Voting)\n')
	parser.add_argument("--Test_size", type=float, default= 0.2,
						help='Number of observations to use for testing the model\n(default: 0.2)\n')
	args = parser.parse_args()

	main(args)



