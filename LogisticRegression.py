import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation as validate

titanic = pandas.read_csv("train.csv");
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"]=='male',"Sex"] = 0
titanic.loc[titanic["Sex"]=='female',"Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"]=='S',"Embarked"] = 0
titanic.loc[titanic["Embarked"]=='C',"Embarked"] = 1
titanic.loc[titanic["Embarked"]=='Q',"Embarked"] = 2
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
logit = LogisticRegression(random_state=1)
logit.fit(titanic[predictors],titanic["Survived"])

titanic_test = pandas.read_csv("test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male","Sex"] = 0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna(titanic_test["Embarked"].value_counts().idxmax())
titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"] = 2
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

predictions = logit.predict(titanic_test[predictors])

submission = pandas.DataFrame({
    "PassengerId" : titanic_test["PassengerId"],
    "Survived" : predictions
    })

submission.to_csv("logit_result.csv",index=False)

scores=validate.cross_val_score(logit,titanic[predictors],titanic["Survived"], cv=3)
score = scores.mean()

print("Score : %f" % score)
