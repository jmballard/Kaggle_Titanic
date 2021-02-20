
# Kaggle competition on Titanic dataset

### Data Dictionary
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
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## Analysis

### Import and quick data check
We first import the datasets and look at them with some plots and statistics. We remove name and Ticket.
### Preprocessing
The preprocessing contains 5 steps:

- We create a column to say if they had a cabin number or not.
- Remove the outlier of the Fare column
- Remove the NAS, with either a label encoder or the median.
- Create categories from the Class, Sex, Siblings, Embarked and Cabin_NA, then dummies columns
- Separate features and label from train dataset, with only the columns we want to use for the modelling

In the end, separate the train dataset into training and testing subdatasets. The "test" file will be used as validation.
### Models chosen
The first model chosen is a XGBRegressor with GridSearch. 
We tried the hyper parameters:

- "eta":[0.05,0.1,0.2,0.3]
- "gamma":[0,0.01,0.1,0.15,0.2]
- "max_depth":[3,4,5,6,7]

The best hyper parameters are: 'eta': 0.1, 'gamma': 0, 'max_depth': 3.
### Score in Kaggle

The current Score in Kaggle for the current model is 0.77272.

