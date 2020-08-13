# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn.exceptions
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
import sklearn.exceptions

df = pd.read_csv('C:\python Projects/zoo.csv') #../input/zoo.csv
df.info()
print ('End')
df.head()


# %%
df02 = pd.read_csv('C:\python Projects/class.csv') #../input/class.csv
df02.info()
print ('End')
df02.head()

df03 = df.merge(df02,how='left',left_on='class_type',right_on='Class_Number')
df03.head()



# %%
#Plotting class types and the number of them.
plt.figure(figsize = (10,8))
sns.countplot(df03['Class_Type'],label="Count")
plt.show()


# %%
#Merging DataFrames


# %%
#Creating a heatmap for the percentage of features on class types.
f_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic']

df03['clt'] = 1

br = df03.groupby(by='Class_Type').mean()
columns = ['class_type','Class_Number','Number_Of_Animal_Species_In_Class','clt','legs']
br.drop(columns, inplace=True, axis=1)
plt.subplots(figsize=(10,4))
sns.heatmap(br, annot=True, cmap="YlGnBu")


# %%
#Creating a decision tree model with 20% training of the data. 
#Model 1
X = df[f_names]
y = df['class_type'] 

#Spliting the dataframe into train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

#Specifing the model for trainning 
classifier = DecisionTreeClassifier().fit(X_train, y_train)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) #Ignore warning for dividing by zero equals zero

#Results
pred1 = classifier.predict(X_test)
print('Accuracy of classifier: {:.2f}' #Accuracy of the classifier on the test set.
     .format(classifier.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred1))
print()
print(classification_report(y_test, pred1))

df03[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type') #The order of the labels in the confusion matrix


# %%
#Displaying the most important features of the model 1
importance = pd.DataFrame(classifier.feature_importances_)
feature = pd.DataFrame(f_names)
feature_importance = pd.concat([feature,importance],axis=1, )
feature_importance.columns = ['Feature', 'Importance']
feature_importance.sort_values(by='Importance',ascending=False)


# %%
#Creating a decision tree model with 10% training of the data. 
#Model 2
X = df[f_names]
y = df['class_type'] #there are multiple classes in this column

#spliting the dataframe into train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1, test_size=.9)

#specifing the model for trainning
classifier2 = DecisionTreeClassifier().fit(X_train, y_train)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)  #ignore warning for dividing by zero equals zero

#Results
pred = classifier2.predict(X_test)
print('Accuracy of classifier: {:.2f}' #Accuracy of the classifier on the test set.
     .format(classifier2.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))
df03[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type') #The order of the labels in the confusion matrix


# %%
#Displaying the most important features of the model 2
importance2 = pd.DataFrame(classifier2.feature_importances_)
feature = pd.DataFrame(f_names)
feature_importance2 = pd.concat([feature,importance2],axis=1, )
feature_importance2.columns = ['Feature', 'Importance']
feature_importance2.sort_values(by='Importance',ascending=False)


# %%
#Reducing the depth size of the tree.
classifier3= DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

#Results
pred = classifier3.predict(X_test)
print('Accuracy of classifier: {:.2f}'
     .format(classifier3.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# %%
#Displaying the most important features of the model 3
importance3 = pd.DataFrame(classifier2.feature_importances_)
feature = pd.DataFrame(f_names)
feature_importance3 = pd.concat([feature,importance3],axis=1, )
feature_importance3.columns = ['Feature', 'Importance']
feature_importance3.sort_values(by='Importance',ascending=False)


# %%
#Visualising the comparison of the models for Accuracy, Precision, Recall and F1
columns = ['Model','Test %', 'Accuracy','Precision','Recall','F1',]
df_final = pd.DataFrame(columns=columns)

df_final.loc[len(df_final)] = ["Model 1",20,.78,.80,.78,.77,] #The metrics where written manually after running the models.
df_final.loc[len(df_final)] = ["Model 2",10,.68,.62,.68,.64,] 
df_final.loc[len(df_final)] = ["Model 3",20,.91,.93,.91,.91,]
box=df_final[['Accuracy','Precision','Recall','F1']].plot(kind='bar', figsize=(10,6))
box.set_xticklabels(df_final.Model)


# %%


