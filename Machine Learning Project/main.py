import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset=pd.read_csv('diabetes.csv')
diabetes_dataset.head()                                 #getting the information about first 5 rows
diabetes_dataset.shape                                #getting the shape (number of rows and colums) from the dataset
diabetes_dataset.describe()                             #getting all the statistical informations like mean,count, Standard deviation etc. of each columns
diabetes_dataset['Outcome'].value_counts()              #getting the total number of diabetic pateints and non diabetic patients
# 0-------> Non-Diabetic
#1--------> Diabetic
diabetes_dataset.groupby('Outcome').mean()             #getting the mean values for each field after grouping ny Outcomes of level 0 (Non-Diabetic) or Level 1(Diabetic)


#Separating the outcomes and the Datas
X=diabetes_dataset.drop(columns='Outcome',axis=1)       # We use axis=1 for dropping a column and we use axis= 0 for dropping a row
Y=diabetes_dataset['Outcome']                           #Storing the values for Outcome in the variable Y


#Standardizing the data
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
X=standardized_data
Y=diabetes_dataset['Outcome']


#Splitting the data into training data and testing data
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)    #test_size represents the % of data for testing, stratify is important so that datas from both the levels are sent for training

#Training the model
classifier=svm.SVC(kernel='linear')

#Training the support vector machine classifier
classifier.fit(X_train,Y_train)

#Evaluation of model- Checking how many times the model is predict correctly
#Accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of the training data: ", training_data_accuracy)

#Accuracy score on testing data
X_test_prediction=classifier.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of the testing data: ", testing_data_accuracy)

#Making a Predictive System
input_data=(6,92,92,0,0,19.9,0.188,28)

#changing the input data from a tuple to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#Reshape the array as we are predicting for only 1 instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#Standardized the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

#Making prediction
prediction= classifier.predict(std_data)
print(prediction)                               #The prediction is a list and not a normal variable

if(prediction[0]==0):
    print("The person is not Diabetic. \n The level is:0")
elif(prediction[0]==1):
    print("The person is Diabetic. \n The level is:1")
