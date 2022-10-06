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

output_assessment = os.path.join(
    "outputs",
    "assessment")

output_assessment_ebm = os.path.join(
    output_assessment,
    "ebm")

if not os.path.exists(output_submissions):
    os.makedirs(output_submissions)

if not os.path.exists(output_models):
    os.makedirs(output_models)

if not os.path.exists(output_assessment):
    os.makedirs(output_assessment)

if not os.path.exists(output_assessment_ebm):
    os.makedirs(output_assessment_ebm)

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

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import set_visualize_provider, preserve
from interpret.provider import InlineProvider

set_visualize_provider(InlineProvider())

# Model: XGBRegressor with GridSearch
ebm = ExplainableBoostingRegressor(random_state = random_seed)
params={#'objective' : ['binary:logistic'],
        "learning_rate":[0, 0.005, 0.01, 0.015, 0.02],
        "max_leaves":[3,4,5,6,7],
       # "min_child_weight": [1,2,3,4,5,6,7]
       }
my_grid = GridSearchCV(ebm,
                    params,
                    cv = 5,
                    n_jobs = -1,
                    verbose = 1)
my_grid.fit(X_train, y_train)
tuned_params = my_grid.best_params_

ebm = ExplainableBoostingRegressor(random_state = random_seed,
                                   learning_rate = tuned_params['learning_rate'],
                                   max_leaves = tuned_params['max_leaves'])
ebm.fit(X_train, y_train)

# Save
pickle.dump(ebm,
            open(os.path.join(output_models,
                              "ebm.pickle"),
                 "wb"))



#%% 4 Load model


my_model = pickle.load(open(os.path.join(output_models,
                                        "ebm.pickle"), "rb+"))


#%% 5 Scoring on training set

# predictions and accuracy
ypred = my_model.predict(X_train)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy train: {round(accuracy_score(ypred_bin,y_train) * 100,4)}%")


ypred = my_model.predict(X_test)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy test: {round(accuracy_score(ypred_bin,y_test) * 100,4)}%")


ypred = my_model.predict(X)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy all: {round(accuracy_score(ypred_bin,y) * 100,4)}%")
# Accuracy train: 84.3972%
# Accuracy test: 85.3107%
# Accuracy all: 84.5805%


#%% 6 Scoring on validation set

validation = pd.read_pickle(os.path.join(output_preprocessing,
                                    "preprocessed_test.csv"))

pred_proba= my_model.predict(validation[features])
pred_bin = np.where(pred_proba >= 0.5, 1, 0)

output = pd.DataFrame({'PassengerId': validation["PassengerId"], 
                        'Survived': pred_bin})
output.to_csv(os.path.join(output_submissions,
                           'submission_EBM.csv'),
              index=False)


#%% 7 Feature assessment


ebm_global = ebm.explain_global()
for col in ['Age', 'SibSp', 'Parch', 'Fare',
            'Pclass_2', 'Pclass_3', 'Sex_male',
            'Embarked_1', 'Embarked_2',
            'Cabin_NA_1', 'Pclass_3 x Sex_male',
            'Age x Sex_male',
            'Pclass_2 x Sex_male',
            'Sex_male x Embarked_1',
            'Fare x Cabin_NA_1', 'SibSp x Fare',
            'Fare x Sex_male', 'Age x Parch',
            'Fare x Pclass_3', 'Age x Fare']:
    preserve(ebm_global,
             col,
             file_name = os.path.join(output_assessment_ebm,
                                      f"global_{col}_graph.html"))

# ebm_local = ebm.explain_local(X_train, y_train)
# ebm_local.visualize(5).write_html(os.path.join(output_assessment_ebm,
#                          f"local_graph_5.html"))