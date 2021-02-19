
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
We first import the datasets and look at them with some plots and statistics.
### Preprocessing
The preprocessing contains 5 steps:

- Separate last name, first name and titles (some titles may have an impact on their survival)
- From the cabin number, separate the alphabetical component and the numerical component
- Remove the NAS
- Create categories from the Class, Sex, Siblings, Embarked and Cabin letter, then dummies columns
- Separate features and label from train dataset, with only the columns we want to use for the modelling
### Models chosen
The first model chosen is a Random Forest Classifier on the whole train dataset.
### Score in Kaggle

The current Score in Kaggle for the current model is 0.34449.

