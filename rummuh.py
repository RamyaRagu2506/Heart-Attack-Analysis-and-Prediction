import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#read the data file 
heart_data= pd.read_csv('heart.csv')

#in this study we get to know how cholestrol in comparison to type of heart diseases 
#extract cholestrol data 
heart_chol_data = heart_data.chol
print(type(heart_chol_data))
print(heart_chol_data)
#plot chol data 
index_chol = range(0,len(heart_chol_data))
#plt.scatter(index_chol, heart_chol_data)
#plt.ylabel("chol level")
#plt.show()

#divide them into categories 
chol_between_0_200 = heart_chol_data[np.logical_and(heart_chol_data >0,heart_chol_data<200)]
#print(chol_between_0_200)

chol_between_200_300 = heart_chol_data[np.logical_and(heart_chol_data > 200, heart_chol_data < 300)]
#print(len(chol_between_200_300))

chol_above_300 = heart_chol_data[heart_chol_data > 300]
#print(np.mean(chol_above_300 ))

#Relate Chest pain type and cholestrol 
chest_pain_type = heart_data[np.logical_and(heart_data.chol > 200, heart_data.chol < 300)]
chest_pain_type = chest_pain_type.cp
#print(chest_pain_type.head())
compare_chol_between_200_300_chest_pain_type = pd.concat([chest_pain_type, chol_between_200_300], axis=1).reset_index()
new_compare_data = compare_chol_between_200_300_chest_pain_type.drop(['index'],axis=1)
#print(len(new_compare_data))
#print(new_compare_data.head())

#grouping similar chol levels 
group_cp_type = new_compare_data.groupby('cp').count()
print(group_cp_type)
print(group_cp_type.index)

#plot the correlation
plt.bar(group_cp_type.index, group_cp_type.chol )
plt.ylabel("Chol levels")
plt.xlabel("Chest pain type")
plt.show()




