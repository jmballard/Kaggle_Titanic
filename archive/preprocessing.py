#%% 0 import all the necessary python packages --------------------------------
import os
# import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# homemade functions
from utils.plot import plot_hist

# create folder structure for outputs
output_preprocessing = os.path.join(
    "outputs",
    "preprocessing")
output_proc_plots = os.path.join(
    output_preprocessing,
    "plots"
)

if not os.path.exists(output_preprocessing):
    os.makedirs(output_preprocessing)

if not os.path.exists(output_proc_plots):
    os.makedirs(output_proc_plots)


#%% 1 import datasets --------------------------------------------------------
train = pd.read_csv("data/train.csv") 
train['sample'] = 'train'
test = pd.read_csv("data/test.csv")
test['sample'] = 'test'

modelling_data = pd.concat([train,test], axis = 0)

assert train.shape[0] + test.shape[0] == modelling_data.shape[0], "We have a problem of merging"

# check details of training dataset 
modelling_data.head()
modelling_data.info()
modelling_data.describe()
modelling_data.dtypes
modelling_data.isna().sum()

#%% 2 Pre processing ---------------------------------------------------------


# rows of missing embarked. Replacing the NAs by "S", which is the most commun value
modelling_data.Embarked.value_counts()
modelling_data.Embarked.isna().value_counts()
modelling_data = modelling_data.fillna(value = {"Embarked" : "S"})

modelling_data["SibSp"].value_counts() # keep them as numeric
modelling_data["Parch"].value_counts() # keep them as numeric

modelling_data[['LastName','FirstName']] = modelling_data['Name'].str.split(',', n = 1,expand = True)
modelling_data[['Title','FirstNames']] = modelling_data['FirstName'].str.split('.', n = 1,expand = True)
modelling_data['Title'] = modelling_data['Title'].str.replace(" ","")

for col in ['Age','Fare','Sex','Title']:
    plot_hist(
        df = modelling_data,
        var_to_plot = col,
        var_hue = "Survived",
        save_fig = True,
        save_location = output_proc_plots
    )



# preprocessing part 1: say if they had a cabin or not
modelling_data["Cabin_NA"] = modelling_data.Cabin.isna().astype(int)
modelling_data = modelling_data.drop(["Cabin"], axis = 1)

# preprocessing part 2: remove outliers from training data only
training_fare_threshold = modelling_data[modelling_data['sample'] == 'train']["Fare"].quantile(0.99)

modelling_data = modelling_data[ (
    (modelling_data["Fare"] < training_fare_threshold) | 
    (modelling_data['sample'] == 'test'))]

# preprocessing part 3: remove NAs
le=LabelEncoder()
modelling_data["Embarked"] = le.fit_transform(modelling_data["Embarked"])

avg_age_train = modelling_data['Age'].median()
modelling_data['Age'] = modelling_data['Age'].\
    fillna(avg_age_train).\
        round(decimals = 0)
modelling_data["Fare"] = modelling_data['Fare'].\
    fillna(modelling_data['Fare'].median()).\
        round(decimals = 0)


# preprocessing part 4: one hot encoding
# features to encode as categorical
features_to_ohe = ["Pclass", "Sex", "Embarked",  "Cabin_NA"]
modelling_data[features_to_ohe] = modelling_data[features_to_ohe].apply(lambda x: x.astype("category"))

# set up the encoder
ohe = OneHotEncoder(sparse = False,
                    handle_unknown = 'error',
                    drop = 'first')
ohe = ohe.fit(modelling_data[features_to_ohe])

# output of encoder
index_name = str(modelling_data.index.name)
transformed = pd.DataFrame(ohe.transform(modelling_data[features_to_ohe]))
transformed.columns = ohe.get_feature_names_out(features_to_ohe)

# put things back together
modelling_data = pd.concat([modelling_data.reset_index(drop = True),
                            transformed.reset_index(drop = True)],
                            axis = 1)
# delete input columns
modelling_data.drop(features_to_ohe,
                    axis = 1,
                    inplace = True)

# preprocessing part 5: separate X and Y, train and test dataset for model
nottaken = ["Ticket",
            "Name",
            "LastName",
            "FirstNames",
            "FirstName",
            "Title"]
modelling_data = modelling_data.drop(nottaken, axis = 1)


#%% 3 Write outputs ----------------------------------------------------------

for sample in ['train','test']:
    
    print(f"Write sample {sample}")
    
    df = modelling_data[modelling_data['sample'] == sample].copy()
    
    print(f"\tNumber of rows: {df.shape[0]}")
    
    df.drop('sample',
            axis = 1)

    if sample == 'train':
        df.drop('PassengerId',
                axis = 1)

    df.to_pickle(
        os.path.join(output_preprocessing,f"preprocessed_{sample}.csv"))