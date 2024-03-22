import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('Loan_Train.csv')
df.drop('Loan_ID', axis=1, inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
categorical_column=['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Property_Area','Loan_Status']
for x in categorical_column:
    print(df[x].unique())
#import seaborn as sns
lab=LabelEncoder()
label_encoders = {}
for column in categorical_column:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])
x=df.iloc[:,df.columns!='Loan_Status']
#print(x)
y=df.iloc[:,df.columns=='Loan_Status']
#print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
y_pred = tree_clf.predict(X_train)
#print(y_train)
print("Training Data Set Accuracy: ", accuracy_score(y_train,y_pred))
print("Training Data F1 Score ", f1_score(y_train,y_pred))
print("Validation Mean F1 Score: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='accuracy').mean())

training_accuracy = []
val_accuracy = []
training_f1 = []
val_f1 = []
tree_depths = []

for depth in range(1,20):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(X_train,y_train)
    y_training_pred = tree_clf.predict(X_train)

    training_acc = accuracy_score(y_train,y_training_pred)
    train_f1 = f1_score(y_train,y_training_pred)
    val_mean_f1 = cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='accuracy').mean()

    training_accuracy.append(training_acc)
    val_accuracy.append(val_mean_accuracy)
    training_f1.append(train_f1)
    val_f1.append(val_mean_f1)
    tree_depths.append(depth)


Tuning_Max_depth = {"Training Accuracy": training_accuracy, "Validation Accuracy": val_accuracy, "Training F1": training_f1, "Validation F1":val_f1, "Max_Depth": tree_depths }
Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)

plot_df = Tuning_Max_depth_df.melt('Max_Depth',var_name='Metrics',value_name="Values")
#fig,ax = plt.subplots(figsize=(15,5))
# sns.pointplot(x="Max_Depth", y="Values",hue="Metrics", data=plot_df,ax=ax)

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train,y_train)
model=tree_clf
y_pred = tree_clf.predict(X_train)
print("Training Data Set Accuracy: ", accuracy_score(y_train,y_pred))
print("Training Data F1 Score ", f1_score(y_train,y_pred))
print("Validation Mean F1 Score: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='accuracy').mean())

import pickle

with open('LoanFinal.pickle','wb') as f:
    pickle.dump(model,f)
    f.close()

model = pickle.load(open('LoanFinal.pickle', 'rb'))


# answer=np.array((1.668e+03, 3.890e+03, 2.010e+02, 3.600e+02, 0.000e+00, 1.000e+00,)).reshape(1,-1)
#answer=[[1,1,2,0,0,8333,3167.0,165.0,7,1.0,0]] #1
answer=[[1,1,0,1,0,7660,0.0,104.0,7,0.0,2]] #0

print("Predicts: " + str(model.predict(answer)))
