
# Kaggle competition on Titanic dataset

## Motivations and goals of this project

The goal of this project is to predict the survivality of people on the Titanic, using different models.

We selected the following 4 questions:

Question 1: Did the "Women and children first" have a real impact on the probability of survival of someone?

Question 2: Was there really a link between which ticket class you took and your survival?

Question 3: Which model has the highest score in Kaggle?

Question 4: What are the most important features of an EBM?

### Library used
For this project, we will only use Python as language. The packages/modules used for this project are:

- os
- numpy
- pandas
- matplotlib
- sklearn
- interpret
- xgboost

### File where to find the analysis

All the analysis - previously done in different files, has been summarised and grouped in the file "blogpost_notebook.ipynb".

## Data Understanding

### Data Dictionary

From Kaggle, here is the data dictionary.

|Variable	|Definition	         |Key                       |
|-----------|--------------------|--------------------------|
|survival	|Survival	         |0 = No, 1 = Yes           |
|pclass	    |Ticket class	     |1 = 1st, 2 = 2nd, 3 = 3rd |
|sex	    | Sex	                                        |
|Age	    |  Age in years	                                |
|sibsp	    | # of siblings / spouses aboard the Titanic	|
|parch	    |  # of parents / children aboard the Titanic	|
|ticket	    | Ticket number	                                |
|fare	    |Passenger fare	                                |
|cabin	    |Cabin number	                                |
|embarked	|Port of Embarkation    C = Cherbourg,     Q = Queenstown,   S = Southampton         |


### Variable Notes

#### pclass
This is a  proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

#### age
Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

#### sibsp
The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

#### parch
The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## Data Preparation

### Import and quick data check
We first import the datasets and look at them with some plots and statistics. We remove Name and Ticket.

### Preprocessing
The preprocessing contains 5 steps:

- We filled the missing value in the feature Embarked by "S", as it is the most common one.
- We look into more details at the Name column. We try to separate the title and see the impact. We decided not to use them in the end.
- We create a column to say if they had a cabin number or not.
- Remove the missing value, with either a label encoder or the median.
- Create categories from the Class, Sex, Embarked and Cabin_NA, then dummies columns
- Separate features and label from train dataset, with only the columns we want to use for the modelling

In the end, separate the train dataset into training and validation subdatasets.


## Modelling

We tried 3 different types of models:

- A GLM model (LogisticRegression) inside a Pipeline containing a Standard Scaler
- A XGBoost with parameters tuned with GridSearchCV
- A EBM model with parameters tuned with GridSearchCV


## Answer questions

### Q1 - Women and Children first

From the histograms and EBM plots, we do see that women and/or young children had a higher probability to survive than the others.

### Q2 - Survival in third class

From the histograms and EBM plots, we do see that passengers with a third class ticket had a lower probability to survive than the others.


### Q3 - best score in Kaggle

The scores in Kaggle for the models are:

- GLM: 0.76315
- EBM: 0.77272
- XGB: 0.78468

This time, the XGB is the best performing model, followed by EBM then GLM last.

### Q4 - Feature assessment with EBM

We use the "Explain_global" function from the interpret package to give us the most important features

It seems that the most important feature is the Sex. 

Then, very closely, follow the Class (especially the 3rd), Fare, Age and the fact to have a Cabin number written or not.

## Links

Kaggle Competition: https://www.kaggle.com/c/titanic 

Medium article : https://medium.com/@bronnimannj/best-model-to-predict-titanic-survival-2b77fc938543
