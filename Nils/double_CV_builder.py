from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
import sklearn.feature_selection as feature_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import decomposition
from sklearn import ensemble
from matplotlib import pyplot as plt
from sklearn import neighbors


class DCV():
    outer_repeats = 5
    inner_repeats = 10
    num_features = 150
    cv_outer = StratifiedShuffleSplit(n_splits=outer_repeats, test_size=1/outer_repeats, random_state=1)  # check if we have to shuffle
    cv_inner = StratifiedShuffleSplit(n_splits=inner_repeats, test_size=1/inner_repeats, random_state=1)


    def __init__(self, Model):
        self.Model = Model
        self._best_score = 0
        self._best_model = None
        self._best_features = None
        self._best_params = None
        self._best_score = 0
        self.models = []
        self.features = []
        self.indexes = None
        self.all_accuracies = None
        self.all_precision = None
        self.all_recall = None
        self.hyperParams = dict()


    def train_fit(self, data, classes, loop=1):
        self.indexes = pd.MultiIndex.from_product(self.hyperParams.values(), names=self.hyperParams.keys())
        self.all_recall = pd.DataFrame()
        self.all_precision = pd.DataFrame()
        self.all_accuracies = pd.DataFrame()
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
                search = GridSearchCV(self.Model, self.hyperParams,scoring=self.__scoring__, cv=self.cv_inner, n_jobs=-1, refit="accuracy")
                #print(search)
                search.fit(outer_train_x, outer_train_y.to_numpy().ravel())
                #print(search.cv_results_)
                #print(pd.DataFrame(search.cv_results_["mean_test_score"],index=self.indexes))
                self.all_accuracies = self.all_precision.append(pd.DataFrame(search.cv_results_["mean_test_accuracy"],index=self.indexes))
                self.all_precision = self.all_precision.append(pd.DataFrame(search.cv_results_["mean_test_precision"],index=self.indexes))
                self.all_recall = self.all_recall.append(pd.DataFrame(search.cv_results_["mean_test_recall"],index=self.indexes))
                #print(self.all_accuracies)
                accuracy = accuracy_score(outer_test_y, search.predict(outer_test_x))
                precision = precision_score(outer_test_y, search.predict(outer_test_x), average="micro")
                if accuracy > self._best_score:
                    self._best_model = search.best_estimator_
                    self._best_params = [search.best_params_]
                    self._best_score = accuracy
                    self._best_features = [best_features.get_support()]
                elif accuracy == self._best_score:
                    self._best_params += [search.best_params_]
                    self._best_features += [best_features.get_support()]
                print(f"accuracy: {accuracy}; precision:{precision}; model:{search.best_estimator_}")
        self.Model.set_params(**self._best_params[0])
        data = data.transpose().loc[self._best_features[0]].transpose()
        self.Model.fit(data, classes.to_numpy().ravel())

    def fit(self, data, classes):
        if self._best_params == None:
            raise RuntimeError("RuntimeError: you did not train yet. please use the Train_fit function.")
        data = self.__data_cleaner__(data)
        data = data.transpose().loc[self._best_features].transpose()
        classes = self.__class_cleaner__(classes)
        self.Model.set_params(**self._best_params)
        self.Model.fit(data,classes)

    def predict(self, data):
        data = self.__data_cleaner__(data)
        data = data.transpose().loc[self._best_features[0]].transpose()
        return self.Model.predict(data)

    def __data_cleaner__(self, data):
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

        raise TypeError("TypeError: class data is not correct")

    def get_best_model(self):
        self.Model = self.Model.set_params(**self._best_params[0])
        return self.Model

    def get_best_params(self):
        return self._best_params

    def get_best_features(self):
        return self._best_features

    def get_best_score(self):
        return self._best_score

    def save_all(self):
        pass

    def __scoring__(self, model, x,y):
        y_pred = model.predict(x)
        acc = accuracy_score(y_true=y,y_pred=y_pred)
        pres = precision_score(y_true=y, y_pred=y_pred, average="micro")
        recall = recall_score(y_true=y,y_pred=y_pred, average="micro")
        return {"accuracy":acc, "precision":pres,"recall":recall}

    # TODO:
    # recall, precision, negative prediction info




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
    modeller.hyperParams['C'] = [100, 10, 1.0, 0.1, 0.01]
    modeller.train_fit(data, data.index, loop=10)
    """

    model = linear_model.LogisticRegression(penalty="elasticnet", max_iter=10000, multi_class='ovr', solver='saga')
    modeller = DCV(model)
    print()
    # prints all the params you can change :)
    print(model.get_params().keys())
    # select which params you want to test with the inner loop
    # space is the paramater space
    modeller.hyperParams['C'] = [100,10]
    modeller.hyperParams['l1_ratio'] = [0.1,0.2]
    # this does everything for you :)
    # first is the data used, seconds comes the classifications and than with loop the amount of loops you want to do
    modeller.train_fit(data, data.index, loop=1)
    # some test code
    print(accuracy_score(DCV.__class_cleaner__(None, data.index),modeller.predict(data)))
    print(topInfo.transpose().iloc[:-1].loc[modeller.get_best_features()[0]])
    print(modeller.get_best_score())
    print(modeller.get_best_features())
    print(modeller.get_best_features())
    print(modeller.all_accuracies)
    print(modeller.all_precision)
    print(modeller.all_recall)