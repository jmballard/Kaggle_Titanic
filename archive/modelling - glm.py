#%% import all the necessary python packages -----
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score



output_preprocessing = os.path.join(
    "outputs",
    "preprocessing")

output_submissions = os.path.join(
    "outputs",
    "submissions")


output_models = os.path.join(
    "outputs",
    "models")

if not os.path.exists(output_submissions):
    os.makedirs(output_submissions)

if not os.path.exists(output_models):
    os.makedirs(output_models)

random_seed = 21

#%% 1 Import pre processed files

train = pd.read_pickle(os.path.join(output_preprocessing,
                                    "preprocessed_train.csv"))

#%% 2 Separating training data into train/test

X = train.drop(['PassengerId','sample','Survived'], axis = 1, inplace = False)
y = train["Survived"]

features = X.columns

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = random_seed,
    shuffle = True)


#%% 3 Modelling

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model: XGBRegressor with GridSearch
std_slc = StandardScaler()
logistic_reg = linear_model.LogisticRegression()

pipe = Pipeline(steps = [
    ('std_slc' , std_slc),
    ('logistic_reg', logistic_reg)
    ])

params={"logistic_reg__C": np.logspace(-4, 4, 50),
        #"logistic_reg__penalty" : ['l1','l2'],
        }
my_grid = GridSearchCV(pipe,
                    params,
                    cv = 10,
                    n_jobs = -1,
                    verbose = 1)
my_grid.fit(X_train, y_train)
#my_grid.best_params_

# Save
pickle.dump(my_grid.best_estimator_,
            open(os.path.join(output_models,
                              "glm.pickle"),
                 "wb"))



#%% 4 Load model


my_model = pickle.load(open(os.path.join(output_models,
                                        "glm.pickle"), "rb+"))


#%% 5 Scoring on training set

# predictions and accuracy
ypred = my_model.predict(X_train)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy train: {round(accuracy_score(ypred_bin,y_train) * 100,4)}%")
# Accuracy train: 80.7092%

ypred = my_model.predict(X_test)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy test: {round(accuracy_score(ypred_bin,y_test) * 100,4)}%")
# Accuracy test: 87.5706%


ypred = my_model.predict(X)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy all: {round(accuracy_score(ypred_bin,y) * 100,4)}%")
# Accuracy all: 82.0862%


#%% 6 Scoring on validation set

validation = pd.read_pickle(os.path.join(output_preprocessing,
                                    "preprocessed_test.csv"))

pred_proba= my_model.predict(validation[features])
pred_bin = np.where(pred_proba >= 0.5, 1, 0)

output = pd.DataFrame({'PassengerId': validation["PassengerId"], 
                        'Survived': pred_bin})
output.to_csv(os.path.join(output_submissions,
                           'submission_GLM.csv'),
              index=False)
