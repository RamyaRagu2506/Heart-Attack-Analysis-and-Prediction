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

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

#Read data from file
heart = pd.read_csv('heart.csv')
print(heart.shape)

#Split data into features and labels
features = heart.drop('output', axis=1)
label =  heart['output']

#model using knn
knn = KNeighborsClassifier()

#Split the data into training and testing dataset with Shuffling
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, shuffle=True)

#fit into the model 
model = knn.fit(X_train, y_train)

#predict using the test set 
y_pred =knn.predict(X_test)

#confusion matrix
conf = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
cm_display.plot()
plt.show()

#confusion matrix metrics
print(classification_report(y_test, y_pred))
