from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFECV, RFE, chi2
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import sklearn.feature_selection as feature_selection
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn


class DCV():
    """
    Do Cross validation

    """
    outer_repeats = 5
    inner_repeats = 10
    num_features = 150
    cv_outer = StratifiedShuffleSplit(n_splits=outer_repeats, test_size=1 / outer_repeats)
    cv_inner = StratifiedShuffleSplit(n_splits=inner_repeats, test_size=1 / inner_repeats)

    def __init__(self, Model: sklearn.base.BaseEstimator):
        self.Model = Model
        self.hyperParams = dict()
        self.selected_features = []
        self.reset_model_results()

    def reset_model_results(self):
        self._best_score = 0
        self._best_model = None
        self._best_features = None
        self._best_params = None
        self._best_score = 0
        self.models = []
        self.features = []
        self.indexes = None
        self.train_accuracies = []
        self.all_features = []
        self.train_precision = []
        self.train_recall = []
        self.test_accuracy = []
        self.test_precision = []
        self.test_recall = []
        self.all_params = []

    def train_fit(self, data, classes, loop=1, verbose=False):
        """

        :param data:
        :param classes:
        :param loop:
        :param verbose:
        :return:
        """
        data = self.__data_cleaner__(data)
        classes = self.__class_cleaner__(classes)
        self.reset_model_results()

        self.hyperParams_est = {"classification__{}".format(x): self.hyperParams[x] for x in self.hyperParams.keys()}
        self.feature_names = list(data)
        self.indexes = pd.MultiIndex.from_product(self.hyperParams_est.values(), names=self.hyperParams_est.keys())

        self.update_progress(0)
        for i in range(loop):
            for j, (outer_train_index, outer_test_index) in enumerate(self.cv_outer.split(data, classes)):
                outer_train_x, outer_test_x = data.iloc[outer_train_index], data.iloc[outer_test_index]
                outer_train_y, outer_test_y = classes.iloc[outer_train_index], classes.iloc[outer_test_index]
                #

                pipeline = Pipeline([("feature_selection", SelectKBest(chi2, k=self.num_features)),
                                     ("classification", self.Model)])

                search = GridSearchCV(pipeline, self.hyperParams_est, scoring=self.__scoring__, cv=self.cv_inner,
                                      n_jobs=4, refit="accuracy")

                search.fit(outer_train_x + 1, outer_train_y.to_numpy().ravel())
                feature_selection = search.best_estimator_.named_steps['feature_selection']

                self.train_accuracies += [search.cv_results_["mean_test_accuracy"]]
                self.train_precision += [search.cv_results_["mean_test_precision"]]
                self.train_recall += [search.cv_results_["mean_test_recall"]]

                # print(self.all_accuracies)
                self.__test__(search, outer_test_x, outer_test_y)
                if self.test_accuracy[-1][0] > self._best_score:
                    self._best_model = search.best_estimator_
                    self._best_params = [search.best_params_]
                    self._best_score = self.test_accuracy[-1][0]
                    self._best_features = feature_selection.transform(np.arange(len(data.columns)).reshape(1, -1))
                elif self.test_accuracy[-1][0] == self._best_score:
                    self._best_params += [search.best_params_]
                    self._best_features += feature_selection.transform(np.arange(len(data.columns)).reshape(1, -1))
                self.all_params += [search.best_params_]
                self.all_features += [feature_selection.transform(np.arange(len(data.columns)).reshape(1, -1))]
                # print(self._best_features)
                if verbose:
                    print(
                        f"accuracy: {self.test_accuracy[-1][0]}; "
                        f"precision:{self.test_precision[-1][0]};"
                        f" model:{search.best_estimator_}")
                self.update_progress(progress=(i * self.outer_repeats + j + 1) / (self.outer_repeats * loop))
        self.selected_features = self._best_features[0]
        self.train_accuracies = pd.DataFrame(self.train_accuracies, columns=self.indexes)
        self.train_precision = pd.DataFrame(self.train_precision, columns=self.indexes)
        self.train_recall = pd.DataFrame(self.train_recall, columns=self.indexes)
        self.test_accuracy = pd.DataFrame(self.test_accuracy, columns=["values"] + list(self.hyperParams.keys()))
        self.test_precision = pd.DataFrame(self.test_precision, columns=["values"] + list(self.hyperParams.keys()))
        self.test_recall = pd.DataFrame(self.test_recall, columns=["values"] + list(self.hyperParams.keys()))
        self.fit(data, classes)

    def __test__(self, model, data, classes):
        """
        calculates all the accuracy, precision, and recall of a given model with data and classes
        :param model: an Sklearn model that is already fitted.
        :param data:
        :param classes:
        :return:
        """
        self.test_accuracy += [
            [accuracy_score(classes, model.predict(data))] + list(model.best_params_.values())]
        self.test_precision += [[precision_score(classes, model.predict(data), average="macro", zero_division=0)] +
                                list(model.best_params_.values())]
        self.test_recall += [[recall_score(classes, model.predict(data), average="macro", zero_division=0)] +
                             list(model.best_params_.values())]

    def fit(self, data, classes, params: list = None, features: dict = None):
        """
        Fit

        :param data:
        :param classes:
        :param params:
        :param features:
        :return:
        """
        if self._best_params == None and params == None:
            raise RuntimeError("RuntimeError: you did not train yet. please use the Train_fit function.")
        elif params != None:
            self.Model.set_params(**params)
        else:
            print(self._best_params[0].values())
            params = {x: list(self._best_params[0].values())[i] for i, x in enumerate(self.hyperParams)}
            self.Model.set_params(**params)
        data = self.__data_cleaner__(data)
        ##################
        if features == "all":
            pass
        elif len(self.selected_features) == 0 and features:
            raise RuntimeError("RuntimeError: you did not train yet. please use the Train_fit function. features")
        elif features != None:
            data = data.iloc[:, features]
        else:
            print(self.selected_features)
            data = data.iloc[:, self.selected_features]
        ##################
        classes = self.__class_cleaner__(classes)
        self.Model.fit(data, classes.to_numpy().ravel())

    def set_features(self, features: list):
        """

        :param features:
        :return:
        """
        self.selected_features = features

    def predict(self, data: pd.DataFrame) -> list:
        """

        :param data:
        :return:
        """
        data = self.__data_cleaner__(data)
        if self._best_features.any():
            data = data.iloc[:, self._best_features[0]]
        return self.Model.predict(data)

    def __data_cleaner__(self, data: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        """
        cleans the data so that it is always a pandas dataframe
        :param data: the data in an numpy array of pandas dataframe
        :return:
        returns a pandas dataframe
        """
        if type(data) == pd.DataFrame:
            # print(data.shape)
            return data
        elif type(data) == np.ndarray:
            return pd.DataFrame(data)
        raise TypeError("TypeError: data is not correct")

    def __class_cleaner__(self, classes: pd.DataFrame or pd.Series or np.ndarray or pd.Index) -> pd.DataFrame:
        """
        cleanes the data that consist of the classes so that the script can use the data for most applications
        :param classes: an array (pandas, numpy or pandas series) of all the classes.
        :return:
        returns the classes in a pandas array.
        """
        if type(classes) == pd.core.indexes.base.Index:
            return classes.to_frame()
        elif type(classes) == pd.DataFrame:
            return classes
        elif type(classes) == np.ndarray:
            return pd.DataFrame(classes)
        elif type(classes) == pd.Series:
            return classes.to_frame("classes")
        raise TypeError("TypeError: class data is not correct")

    def get_best_model(self) -> sklearn.base.BaseEstimator:
        self.Model = self.Model.set_params(**self._best_params[0])
        return self.Model

    def get_best_params(self) -> list:
        return self._best_params

    def get_best_features(self) -> list:
        return self._best_features

    def get_best_score(self) -> float:
        return self._best_score

    def save_all_data(self, modelname: str = "model"):
        """

        :param modelName:
        :return:
        """
        self.test_accuracy.to_csv("{}_test_accuracy.csv".format(modelname))
        self.test_precision.to_csv("{}_test_precision.csv".format(modelname))
        self.test_recall.to_csv("{}_test_recall.csv".format(modelname))
        self.train_accuracies.to_csv("{}_train_accuracy.csv".format(modelname))
        self.train_precision.to_csv("{}_train_precision.csv".format(modelname))
        self.train_recall.to_csv("{}_train_recall.csv".format(modelname))
        pd.DataFrame(self._best_features).to_csv("{}_features.csv")
        pd.DataFrame(self._best_params).to_csv("{}_features.csv")

    def __scoring__(self, model: sklearn.base.BaseEstimator, x: pd.DataFrame, y: np.array) -> dict:
        """
        scoring for the gridsearch. this function is used during the innerloop to create all the data.
        :param model: the model
        :param x: the data.
        :param y: the classes
        :return:
        :returns the accuracy, precision, and recall for the gridsearch in a dict format
        """
        y_pred = model.predict(x)
        acc = accuracy_score(y_true=y, y_pred=y_pred)
        pres = precision_score(y_true=y, y_pred=y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true=y, y_pred=y_pred, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": pres, "recall": recall}

    def update_progress(self, progress: int or float):
        """
        update_progress() : Displays or updates a console progress bar
        Accepts a float between 0 and 1. Any int will be converted to a float.
        A value under 0 represents a 'halt'.
        A value at 1 or bigger represents 100%
        :param progress:
        :return:
        """
        barLength = 50  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rTraining progress: [{0}] {1:.1f}% {2}".format("#" * block + "-" * (barLength - block), progress * 100,
                                                                status)
        print(text)


if __name__ == '__main__':
    # imports the data
    data = pd.read_csv("raw_data/train_call.txt", index_col=0, delimiter="\t").transpose()
    topInfo = data.iloc[:3]
    data = data.iloc[4:].dropna(axis=1)
    classes = pd.read_csv("raw_data/train_clinical.txt", delimiter="\t", index_col=0)
    data = classes.join(data).set_index("Subgroup").dropna()
    # Add 1 to data because of chi-square feature selection

    """
    
    
    """

    lr_model = linear_model.LogisticRegression(penalty="l2", max_iter=10000, multi_class='ovr', solver='liblinear')
    modeller = DCV(lr_model)
    modeller.hyperParams['C'] = [1.0, 0.1]
    modeller.train_fit(data, data.index, loop=1)

    # model = linear_model.LogisticRegression(penalty="elasticnet", max_iter=10000, multi_class='ovr', solver='saga')
    #
    # modeller = DCV(model)
    # modeller.fit(data, data.index, {"C": 10, "l1_ratio": 0.9}, "all")
    # # prints all the params you can change :)
    # print(model.get_params().keys())
    # # select which params you want to test with the inner loop
    # # space is the paramater space
    # modeller.hyperParams['C'] = [1]
    # modeller.hyperParams['l1_ratio'] = [0.9]
    # # this does everything for you :)
    """
    model = RandomForestClassifier(random_state=123)
    modeller = DCV(model)
    # prints all the params you can change :)
    print(model.get_params().keys())
    # select which params you want to test with the inner loop
    # space is the paramater space
    modeller.hyperParams['n_estimators'] = [int(x) for x in np.linspace(start=50, stop=500, num=1)]
    modeller.hyperParams['max_features'] = ["sqrt", ]
    # modeller.hyperParams['max_depth'] = [int(x) for x in np.linspace(10, 110, num = 11)]
    """
    # first is the data used, seconds comes the classifications and than with loop the amount of loops you want to do
    # modeller.train_fit(data, data.index, loop=1)
    # some test code
    print("accuracy:")
    print(accuracy_score(modeller.__class_cleaner__(data.index).to_numpy().ravel(), modeller.predict(data)))
    print()
    print(topInfo.transpose().iloc[:-1].iloc[modeller.get_best_features()[0]])
    print(modeller.get_best_score())
    print(modeller.get_best_features())
    print(pd.DataFrame(modeller.get_best_features()).replace(True, 1))
    print(modeller.train_accuracies)
    print(modeller.train_precision)
    print(modeller.train_recall)
    print(modeller.test_accuracy)
    print(modeller.test_precision)
    print(modeller.test_recall)
    # print(modeller.save_all())
    # modeller.fit(data,data.index,{"C":1,"l1_ratio":0.9})
