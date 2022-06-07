import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


df = pd.read_csv("diabetes.csv") 
df.head()
df.describe()
df.hist()
# plt.show() # Histograms provide interesting insights into the data.
# We are able to detect outliers , null values and the distribution of the data of  each column.

# create a subplot of 3 x 3
plt.subplots(3,3,figsize=(15,15))
# Plot a density plot for each variable
for i, col in enumerate(df.columns):
 ax = plt.subplot(3,3,i+1)
 ax.yaxis.set_ticklabels([])
 sns.distplot(df.loc[df.Outcome == 0][col], axlabel= False, hist=False)
 kde_kws={'linestyle':'-',
 'color':'black', 'label':"No Diabetes"}
 sns.distplot(df.loc[df.Outcome == 1][col], axlabel= False, hist=False)
 kde_kws={'linestyle':'--',
 'color':'black', 'label':"Diabetes"}
 ax.set_title(col)
# Hide the 9th subplot (bottom right) since there are only 8 plots
plt.subplot(3,3,9).set_visible(False)
plt.show()
# By using the density plots we get a rough idea about the poor predictors and good predictors of diabetes

# Data Preprocessing

(df.isnull())
(df.duplicated())
(df.isnull().sum().sort_values(ascending=False))

# There are no null and duplicate values in the data , but in the histograms we did observe values of BMI,Insulin,SkinThickness and BP are 0 which is IRL not possible.

# Handling 0 values
# for i in df.columns:
#     missing_rows=df.loc[df[i]==0].shape[0]
#     print(f"{i} : {str(missing_rows)}")
# There are many rows with missing values so we cannot discard those as we would observe significant drop in performance of the model

# There are several techniques to handle these missing values 
# We will replace the missing values with the mean/median/mode of the non-missing values

df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

for col in df.columns:
    missing_rows=df.loc[df[col]==0].shape[0]
    #print(f"{col} : {(missing_rows)}")
# Data Standardization
df_scaled = preprocessing.scale(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['Outcome'] = df['Outcome']
df = df_scaled

print(df.describe().loc[['mean', 'std','max'],].round(2).abs())

# Spilting data for training, validation and testing

X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)



