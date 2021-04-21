from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
import sklearn.feature_selection as feature_selection
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import decomposition
from matplotlib import pyplot as plt
from sklearn import neighbors


class DCV():
    outer_repeats = 5
    inner_repeats = 10
    num_features = 150
    cv_outer = StratifiedShuffleSplit(n_splits=outer_repeats, test_size=1/outer_repeats, random_state=1)  # check if we have to shuffle
    cv_inner = StratifiedShuffleSplit(n_splits=inner_repeats, test_size=1/inner_repeats, random_state=1)
    space = dict()

    def __init__(self, Model):
        self.Model = Model
        print(self.Model)
        self._best_score = 0
        self._best_model = None
        self._best_features = None
        self._best_params = None
        self.models = []
        self.features = []
        self.all_accuracies = {}


    def train_fit(self, data, classes, loop=100):
        self.all_accuracies = {x:[] for x in self.space}
        print(self.all_accuracies)
        data = self.__data_cleaner__(data)
        classes = self.__class_cleaner__(classes)

        for i in range(loop):
            for outer_train_index, outer_test_index in self.cv_outer.split(data, classes):
                outer_train_x = data.iloc[outer_train_index]
                outer_train_y = classes.iloc[outer_train_index]
                outer_test_x = data.iloc[outer_test_index]
                outer_test_y = classes.iloc[outer_test_index]
                best_features = SelectKBest(feature_selection.chi2, k=self.num_features).fit(outer_train_x+1, outer_train_y)
                outer_train_x = outer_train_x.transpose().loc[best_features.get_support()].transpose()
                outer_test_x = outer_test_x.transpose().loc[best_features.get_support()].transpose()
                search = GridSearchCV(self.Model, self.space,scoring='accuracy', cv=self.cv_inner, n_jobs=-1)
                search.fit(outer_train_x, outer_train_y.to_numpy().ravel())
                accuracy = accuracy_score(outer_test_y, search.predict(outer_test_x))
                if accuracy > self._best_score:
                    self._best_score = accuracy
                    self._best_model = search.best_estimator_
                    self._best_features = None
                    self._best_params = search.best_params_
                    self._best_features = best_features.get_support()
                print(accuracy, search.best_params_, search.best_estimator_)
        self.Model.set_params(**self._best_params)
        data = data.transpose().loc[self._best_features].transpose()
        self.Model.fit(data, classes.to_numpy().ravel())

    def predict(self, data):
        print(f"model with parmas: {self._best_params}")
        self.Model.set_params(**self._best_params)
        data = self.__data_cleaner__(data)
        data = data.transpose().loc[self._best_features].transpose()
        return self.Model.predict(data)

    def __data_cleaner__(self, data):
        print(type(data))
        if type(data) == pd.DataFrame:
            print(data.shape)
            return data
        elif type(data) == np.ndarray:
            return pd.DataFrame(data)
        raise TypeError("TypeError: data is not correct")

    def __class_cleaner__(self, classes):
        if type(classes) == pd.core.indexes.base.Index:
            return classes.to_frame()
        elif len(classes.flatten()) > len(classes):
            print(classes.flatten())
            raise AttributeError
        elif type(classes) == pd.DataFrame:
            return data
        elif type(classes) == np.ndarray:
            return pd.DataFrame(classes)
        elif type(classes) == pd.Series:
            return classes.to_frame("classes")

        raise TypeError("TypeError: data is not correct")

    def get_best_model(self):
        self.Model = self.Model.set_params(self._best_params)
        return self.Model

    def get_best_params(self):
        return self._best_params

    def get_best_features(self):
        return self._best_features





if __name__ == '__main__':

    # imports the data
    data = pd.read_csv("../raw_data/train_call.txt", index_col=0, delimiter="\t").transpose()
    topInfo = data.iloc[:3]
    data = data.iloc[4:].dropna(axis=1)
    classes = pd.read_csv("../raw_data/train_clinical.txt", delimiter="\t", index_col=0)
    data = classes.join(data).set_index("Subgroup").dropna()
    # Add 1 to data because of chi-square feature selection


    """
    
    
    """
    """
    lr_model = linear_model.LogisticRegression(penalty="l2", max_iter=10000, multi_class='ovr', solver='liblinear')
    modeller = DCV(lr_model)
    modeller.space['C'] = [100, 10, 1.0, 0.1, 0.01]
    modeller.train_fit(data, data.index, loop=10)
    """

    model = neighbors.KNeighborsClassifier(n_jobs=-1)
    modeller = DCV(model)
    # prints all the params you can change :)
    print(model.get_params().keys())
    # select which params you want to test with the inner loop
    # space is the paramater space
    modeller.space['n_neighbors'] = list(range(3,51,2))
    # this does everything for you :)
    # first is the data used, seconds comes the classifications and than with loop the amount of loops you want to do
    modeller.train_fit(data, data.index, loop=10)
    # some test code
    print(accuracy_score(DCV.__class_cleaner__(None, data.index),modeller.predict(data)))
    print(topInfo.transpose().iloc[:-1].loc[modeller.get_best_features()])