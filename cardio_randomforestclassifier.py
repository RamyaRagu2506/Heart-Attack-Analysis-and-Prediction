import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Read data from file
heart = pd.read_csv('heart.csv')
print(heart.shape)

#Split data into features and labels
features = heart.drop('output', axis=1)
label =  heart['output']

#test the features and labels
#print(features.columns, label.head(5))

#Split the data into training and testing dataset with Shuffling
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, shuffle=True)

#Model the data using rfc
rfc = RandomForestClassifier()
model = rfc.fit(X_train, y_train)

#predict using test data
y_pred = rfc.predict(X_test)
score = rfc.score(y_test, y_pred)

#confusion matrix
conf = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
cm_display.plot()
plt.show()

recall = recall_score(y_test, y_pred)
prec =  precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(recall, prec, acc)

print(f" Will Srinivas have a heart attack? : {rfc.predict([[25,1,0,90,150,0,0, 160, 0,1.5,1, 0, 0]])}")