# import all the necessary python packages -----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import LabelEncoder


# import datasets -------------------------------
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv")

# # for Kaggle
# train = pd.read_csv("/kaggle/input/titanic/train.csv") 
# test = pd.read_csv("/kaggle/input/titanic/test.csv")

# check details of training dataset -------------
train.head()
train.info()
train.describe()
train.isna().sum()

# 2 rows of missing embarked. Replacing the NAs by "S", which is the most commun value
train.Embarked.value_counts()
train = train.fillna(value = {"Embarked" : "S"})

plt.hist(train["Age"])
plt.show()

train["SibSp"].value_counts() # keep them as numeric
train["Parch"].value_counts() # keep them as numeric

plt.hist(train["Fare"])
plt.show() # there may be some outliers!

sns.histplot(x = train["Sex"],
             hue = train["Survived"])
plt.show() # better survival for women

train['LastName'], train['FirstName'] = train['Name'].str.split(',', n = 1).str
train['Title'], train['FirstNames'] = train['FirstName'].str.split('.', n = 1).str
train['Title'] = train['Title'].str.replace(" ","")
sns.histplot(x = train["Title"],
             hue = train["Survived"])
plt.show() # Title will be useless: we know already that the ladies will survive more!

train = train.drop(["Ticket","Name", "LastName","Title","FirstNames","FirstName"], axis = 1)


# preprocessing part 1: say if they had a cabin or not
train["Cabin_NA"] = train.Cabin.isna().astype(int)
train = train.drop(["Cabin"], axis = 1)

# preprocessing part 2: remove outliers
train = train[train["Fare"] < train["Fare"].quantile(0.99)]

# preprocessing part 3: remove NAs
le=LabelEncoder()
train["Embarked"] = le.fit_transform(train["Embarked"])

avg_age_train = train['Age'].median()
train['Age'] = train['Age'].fillna(avg_age_train).round(decimals = 0)
train["Fare"] = train['Fare'].fillna(train['Fare'].median()).round(decimals = 0)


# preprocessing part 4: create dummy categories
feature_to_categories = ["Pclass", "Sex", "Embarked",  "Cabin_NA"] #"Title"
train[feature_to_categories] = train[feature_to_categories].apply(lambda x: x.astype("category"))
train = pd.get_dummies(train, columns = feature_to_categories , dtype = np.int64)



# preprocessing part 5: separate X and Y, train and test dataset for model
nottaken = ["PassengerId","Survived"]
X = train.drop(nottaken, axis = 1)
y = train["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21,shuffle=True)


# Model: XGBRegressor with GridSearch
rfc = XGBRegressor()
params={"eta":[0.05,0.1,0.2,0.3],
        "gamma":[0,0.01,0.1,0.15,0.2],
        "max_depth":[3,4,5,6,7] }
mdl1 = GridSearchCV(rfc,params,cv=10,n_jobs=-1,verbose=1)
mdl1.fit(X_train, y_train)

ypred=mdl1.predict(X_train)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
tpred=mdl1.predict(X_test)
tpred_bin = np.where(tpred >= 0.5, 1, 0)
print(accuracy_score(ypred_bin,y_train))
print(accuracy_score(tpred_bin,y_test))
mdl1.best_params_




# predict on the test dataset
Val = test.drop(["Ticket","Name"],axis = 1)
Val["Cabin_NA"] = Val.Cabin.isna().astype(int)
Val = Val.drop(["Cabin"], axis = 1)
Val["Embarked"]=le.fit_transform(Val["Embarked"])

Val['Age'] = Val['Age'].fillna(Val['Age'].median()).round(decimals = 0)
Val["Fare"] = Val['Fare'].fillna(Val['Fare'].median()).round(decimals = 0)

Val[feature_to_categories] = Val[feature_to_categories].apply(lambda x: x.astype("category"))
Val = pd.get_dummies(Val, columns = feature_to_categories , dtype = np.int64)

pred1 = mdl1.predict(Val.drop(["PassengerId"], axis = 1))
pred1_bin = np.where(pred1 >= 0.5, 1, 0)

output = pd.DataFrame({'PassengerId':test["PassengerId"], 
                        'Survived': pred1_bin})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")