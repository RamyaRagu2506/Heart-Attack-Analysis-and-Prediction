from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

#read the data file 
heart_data = pd.read_csv('heart.csv')

last_ten = heart_data.tail(10)
#print(last_ten)


#check Null or NA in dataset
#print(heart_data.isnull().sum())

#print(heart_data['thalachh'].mean())

#heart_data['thalachh'].fillna(heart_data['thalachh'].mean(), inplace = True)

#Analyse every feature
# - Count the feature's values
#normalized_chol = heart_data.chol.value_counts(normalize = True)
#normalized_chol.plot.barh()
#plt.show()
#age = heart_data.age.value_counts()
#age.plot.pie()
#plt.show()

#print(heart_data.head(10))

#bivariate analysis
#sns.swarmplot(x='sex', y='thalachh', data=heart_data)
#plt.show()

sns.heatmap(heart_data.corr())
plt.show()