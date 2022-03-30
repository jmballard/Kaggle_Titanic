#%% import all the necessary python packages -----
import os
import numpy as np
import pandas as pd
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

output_lgb = os.path.join(
    output_models,
    "lightbm_cv")

if not os.path.exists(output_submissions):
    os.makedirs(output_submissions)

if not os.path.exists(output_models):
    os.makedirs(output_models)

if not os.path.exists(output_lgb):
    os.makedirs(output_lgb)

random_seed = 21

#%% 1 Import pre processed files

train = pd.read_pickle(os.path.join(output_preprocessing,
                                    "preprocessed_train.csv"))

#%% 2 Separating training data into train/test

X = train.drop(['PassengerId','sample','Survived'], axis = 1, inplace = False)
y = train["Survived"]

features = X.columns.to_list()


#%% 3 Modelling

import lightgbm as lgb
from bayes_opt import BayesianOptimization

dtrain = lgb.Dataset(
    data = X,
    label = y,
    feature_name = features)

np.random.seed(random_seed)
cv = np.random.randint(0,5, X.shape[0])

pbounds = {
    'lambda_l1' : (0,0.1),
    'max_depth' : (3,8),
    'feature_fraction' : (0.5, 1),
    }

params = {
    'boosting_type' : 'gbdt',
    'objective' : 'poisson',
    'metric' : 'poisson',
    'num_leaves' : 32,
    'max_depth' : 8,
    'min_data_in_leaf' : 10
    }

def lgb_cv_error(**kwargs):
    for key, value in kwargs.items():
        if key in ['min_data_in_leaf', 'max_depth']:
            value = int(np.floor(value))
        params[key] = value


    lgb_cv_model = lgb.cv(
        params = params,
        train_set = dtrain,
        num_boost_round = 10000,
        folds = (([idx for idx in range(len(cv)) if cv[idx] != i], 
                  [idx for idx in range(len(cv)) if cv[idx] == i]) for i in range(int(cv.max())+1) ),
        shuffle = False, # we have created the folds already
        verbose_eval = True,
        early_stopping_rounds = 10,
        return_cvbooster = True
        )
    
    return -np.min(lgb_cv_model[params['metric'] + '-mean'])

optimiser = BayesianOptimization(
    f = lgb_cv_error,
    pbounds = pbounds,
    verbose = 100,
    random_state = random_seed)

optimiser.maximize(
    init_points = 5,
    n_iter = 15,
    kappa = 0.5)

for key, value in optimiser.max['params'].items():
    if key in ['min_data_in_leaf', 'max_depth']:
        value = int(np.floor(value))
    params[key] = value

lgb_cv_model = lgb.cv(
    params = params,
    train_set = dtrain,
    num_boost_round = 10000,
    folds = (([idx for idx in range(len(cv)) if cv[idx] != i], 
              [idx for idx in range(len(cv)) if cv[idx] == i]) for i in range(int(cv.max())+1) ),
    shuffle = False, # we have created the folds already
    verbose_eval = True,
    early_stopping_rounds = 10,
    return_cvbooster = True
    )

for i, booster in enumerate(lgb_cv_model['cvbooster'].boosters):
    booster.save_model(os.path.join(output_lgb,
                                    f'mod {i}.txt'),
                       importance_type = 'gain')





#%% 4 Load model

my_model = []

for i in range(5):
    with open(os.path.join(output_lgb,
                           f'mod {i}.txt')) as f:
        my_model.append(lgb.Booster(model_str = f.read() ))

def predict_out_of_fold(model_boosters,
                        x,
                        cvfold):
    index = np.array(range(len(cvfold)))
    
    preds = []
    for i in range(int(cvfold.max()) + 1):
        print(f'Predicting fold {i}')
        preds.append(pd.Series(model_boosters[i].predict(x[cvfold == i]),
                               index = index[cvfold == i]))
        
    pred = pd.concat(preds).sort_index()
    
    return pred.values

#%% 5 Scoring on training set

# predictions and accuracy
ypred = predict_out_of_fold(model_boosters = my_model,
                        x = X[features],
                        cvfold = cv)
ypred_bin = np.where(ypred >= 0.5, 1, 0)
print(f"Accuracy all: {round(accuracy_score(ypred_bin,y) * 100,4)}%")
# Accuracy all: 82.2671%


#%% 6 Scoring on validation set

validation = pd.read_pickle(os.path.join(output_preprocessing,
                                    "preprocessed_test.csv"))

np.random.seed(random_seed)
cv = np.random.randint(0,5, validation.shape[0])

pred_proba= predict_out_of_fold(model_boosters = my_model,
                        x = validation[features],
                        cvfold = cv)
pred_bin = np.where(pred_proba >= 0.5, 1, 0)

output = pd.DataFrame({'PassengerId': validation["PassengerId"], 
                        'Survived': pred_bin})
output.to_csv(os.path.join(output_submissions,
                           'submission_lightgbm_cv.csv'),
              index=False)

