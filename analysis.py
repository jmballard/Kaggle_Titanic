# import all the necessary python packages -----
import numpy as np
import pandas as pd


# import datasets -------------------------------
train = pd.read_csv("/kaggle/input/titanic/train.csv") 
test = pd.read_csv("/kaggle/input/titanic/test.csv")

df = train

# check object columns
df.Name.value_counts() # names with Surname, Title Firstname -> title may be useful!
df.Ticket.value_counts() # number of the Ticket -> not useful for now
df.Cabin.value_counts() # number of the cabin, format LETTER + int. May be interesting to separate them
df.Embarked.value_counts() # Where they embarked. May be useful. 

# remove the Ticket & PassengerId columns
df = df.drop(["Ticket"], axis = 1)


# preprocessing part 1: separate the last name, title and first names from the names. Keep only title
df['LastName'], df['FirstName'] = df['Name'].str.split(',', n = 1).str
df['Title'], df['FirstNames'] = df['FirstName'].str.split('.', n = 1).str
df['Title'] = df['Title'].str.replace(" ","")

df = df.drop(["LastName"], axis = 1)
df = df.drop(["FirstNames"], axis = 1)
df = df.drop(["FirstName"], axis = 1)
df = df.drop(["Name"], axis = 1)


# preprocessing part 2: separate letter from numbers in the Cabin column, fill the NA with X
df["Cabin_letter"] = df.Cabin.str.slice(start = 0, stop = 1)
df = df.drop(["Cabin"], axis = 1)

# preprocessing part 3: remove NAs
df = df.fillna(value = {"Cabin_letter" : "X",
                              "Embarked" : "X"})
avg_age_df = df['Age'].mean()
df['Age'] = df['Age'].fillna(avg_age_df).round(decimals = 0)
df["Fare"] = df['Fare'].fillna(df['Fare'].mean()).round(decimals = 0)


# preprocessing part 4: create dummy categories
feature_to_categories = ["Pclass", "Sex", "SibSp", "Parch","Embarked",  "Cabin_letter"] #"Title"
df[feature_to_categories] = df[feature_to_categories].apply(lambda x: x.astype("category"))
df = pd.get_dummies(df, columns = feature_to_categories , dtype = np.int64)


# preprocessing part 5: separate X and Y
nottaken = ["PassengerId","Survived","Title"]
X_train = df.drop(nottaken, axis = 1)
y_train = df["Survived"]


# Model 1: RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
mdl1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
mdl1.fit(X_train, y_train)

# predict
X_test = test.drop(["Ticket","Name"],axis = 1)
X_test["Cabin_letter"] = X_test.Cabin.str.slice(start = 0, stop = 1)
X_test = X_test.drop(["Cabin"], axis = 1)

X_test = X_test.fillna(value = {"Cabin_letter" : "X",
                              "Embarked" : "X"})
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()).round(decimals = 0)
X_test["Fare"] = X_test['Fare'].fillna(X_test['Fare'].mean()).round(decimals = 0)

X_test[feature_to_categories] = X_test[feature_to_categories].apply(lambda x: x.astype("category"))
X_test = pd.get_dummies(X_test, columns = feature_to_categories , dtype = np.int64)

pred1 = mdl1.predict(X_test)

output = pd.DataFrame({'PassengerId':test["PassengerId"], 
                        'Survived': pred1})
output.to_csv('submission2.csv', index=False)

print("Your submission was successfully saved!")