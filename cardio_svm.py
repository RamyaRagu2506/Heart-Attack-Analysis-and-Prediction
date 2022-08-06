"""
1. Import all the necessary libraries
2. Import the data from the csv file.
3. If you want, try to play with the data
4. segregate the Features and labels respectively. 
5. Instantiate the model given 
6. split the data into training and test set
7. Fit the model with the Feature set and label set (from the data u had split munadi)
8. Predict the test set
9. build the confusion matrix
10. Show the classification report (You can check on google, or ping me, naan solren enadhu. ) - Instead of precision, recall and accuracy

Note: Use proper variables and do it slowly, no hurries, do one algorithm, eat, relax, watch tv or play something u wish, again do it! 
    Make sure u also remove - Age, Sex, Exng, oldpeak, caa, thall from the dataset and run the algorithm again (U will have two models per algorithm)
    Do as many as u can ma. :)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

#read the data 
heart = pd.read_csv('heart.csv')
print(heart.shape)

#spilt it into features and label 
features = heart.drop('output', axis=1)
label =  heart['output']

#model in use 
svc = SVC()

##Split the data into training and testing dataset with Shuffling
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, shuffle=True)

#fit into model
model = svc.fit(X_train, y_train)

#Predict using the test set
y_pred = svc.predict(X_test)

#confusion matrix
conf = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
cm_display.plot()
plt.show()

#confusion matrix metrics
print(classification_report(y_test, y_pred))