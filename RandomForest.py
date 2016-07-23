import pandas
import math
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier as Rforest
from sklearn import cross_validation as validate
from sklearn.feature_selection import SelectKBest, f_classif
import operator

def _readData(filepath):
    data = pandas.read_csv(filepath)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data.loc[data["Sex"] == 'male', "Sex"] = 0
    data.loc[data["Sex"] == 'female', "Sex"] = 1
    data["Embarked"] = data["Embarked"].fillna('S')
    data.loc[data["Embarked"] == 'S', "Embarked"] = 0
    data.loc[data["Embarked"] == 'C', "Embarked"] = 1
    data.loc[data["Embarked"] == 'Q', "Embarked"] = 2
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # new features
    data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["NameLength"] = data["Name"].apply(lambda x: len(x))

    # _family id
    # people with same (last name + family size) = family member
    family_id_mapping = {}

    def _getFamilyId(row):
        lastName = row["Name"].split(",")[0]
        family_id = lastName + str(row["FamilySize"])
        if family_id not in family_id_mapping:
            if len(family_id_mapping) == 0:
                current_id = 1;
            else:
                current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
            family_id_mapping[family_id] = current_id
        return family_id_mapping[family_id]

    family_ids = data.apply(_getFamilyId, axis=1)
    family_ids[data["FamilySize"] < 3] = -1
    data["FamilyId"] = family_ids
    # family id_

    #_get family title
    def _getTitle(name):
        title = re.search(" ([A-Za-z]+)\.",name)
        if title:
            return title.group(1)
        return ""
    titles = data["Name"].apply(_getTitle)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9,
     "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    '''
    j=0
    for i in titles:
        title_mapping[i]=j
        j=j+1
    '''

    for k,v in title_mapping.items():
        titles[titles == k] = v
    data["Title"] = titles
    data["Title"]=data["Title"].apply(lambda x: pandas.to_numeric(x,errors='coerce'))
    data.loc[data["Title"].apply(math.isnan),"Title"] = -1
    # get family title_

    data = data.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
    return data

def _checkColumnsAndConvertToNumeric(data,columns):
    for i in data.columns:
        if data[i].dtypes not in [np.int64, np.float64] and i in columns:
            data[i] = data[i].apply(lambda x: pandas.to_numeric(x, errors='coerce'))
            data.loc[data[i].apply(math.isnan), "Title"] = -1
    return data

# select best features
def _selectBestPredictors(data, initialPredictors, featureThreshold=10, showScore=True):
    selector = SelectKBest(f_classif, k=5)
    selector.fit(data[initialPredictors], data["Survived"])
    scores = -np.log10(selector.pvalues_)
    score_mapping = {}
    k = 0
    for i in scores:
        score_mapping[initialPredictors[k]] = i
        k = k + 1
    # check below dictionary to select threshold value
    if showScore:
        print(score_mapping)
    predictors = list(dict((k, v) for k, v in score_mapping.items() if v >= featureThreshold).keys())
    return predictors

def trainAndTest():
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "FamilyId", "Title"]

    #train
    titanic = _readData("train.csv")
    forest = Rforest(n_estimators=150, min_samples_split=4, min_samples_leaf=2, random_state=1)
    predictors = _selectBestPredictors(titanic,predictors)
    #predictors selected
    print(predictors)
    forest.fit(titanic[predictors], titanic["Survived"])

    #test
    titanic_test = _readData("test.csv")
    titanic_test=_checkColumnsAndConvertToNumeric(titanic_test,predictors)
    predictions = forest.predict(titanic_test[predictors])

    submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("forest_result.csv",index=False)

    scores = validate.cross_val_score(forest, titanic[predictors], titanic["Survived"], cv=3)
    score = scores.mean()
    print("Score : %f" % score)

if __name__ == '__main__':
    trainAndTest()