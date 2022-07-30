#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#read the data file
heart_data = pd.read_csv('heart.csv')

#displaying sample data
#print(heart_data.head())

#extract age column
heart_age_data = heart_data.age
print(type(heart_age_data))

#age based statistics
mean_age = np.mean(heart_age_data)
print(mean_age)

#plot age data
#index_age = range(0,len(heart_age_data))
#plt.scatter(index_age, heart_age_data)
#plt.ylabel("Age of the patients")
#plt.show()

#divide the age groups
age_under_forty = heart_age_data[heart_age_data < 40]
print(np.mean(age_under_forty))

age_between_fortysixty = heart_age_data[np.logical_and(heart_age_data > 40, heart_age_data < 60)]
print(np.median(age_between_fortysixty))
print(len(age_between_fortysixty))

age_above_sixty = heart_age_data[heart_age_data > 60]
print(np.mean(age_above_sixty))

#Relate Chest pain type and age group
chest_pain_type = heart_data[np.logical_and(heart_data.age > 40, heart_data.age < 60)]
chest_pain_type = chest_pain_type.cp
#print(chest_pain_type.head())
compare_age_fortysixty_chest_pain_type = pd.concat([chest_pain_type, age_between_fortysixty], axis=1).reset_index()
print(compare_age_fortysixty_chest_pain_type.head())
new_compare_data = compare_age_fortysixty_chest_pain_type.drop(['index'],axis=1)
print(len(new_compare_data))

#grouping similar age
group_cp_type = new_compare_data.groupby('cp').count()
print(group_cp_type.columns, new_compare_data.columns)

#plot the correlation
plt.bar(new_compare_data.cp.unique(), group_cp_type.age )

#Change the age to count
plt.ylabel("Count of age group")
plt.xlabel("Chest pain type")
plt.show()