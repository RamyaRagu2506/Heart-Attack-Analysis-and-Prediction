#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#read all data
heart_data= pd.read_csv('heart.csv')
#read restecg data 
heart_restecg=heart_data.restecg
print(heart_restecg)
#restecg stat
print(np.mean(heart_restecg))
#find null vallues 
heart_restecg.isnull().sum()