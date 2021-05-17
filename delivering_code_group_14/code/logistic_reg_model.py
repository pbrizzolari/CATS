#%%

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import decomposition
from matplotlib import pyplot as plt
from sklearn import neighbors

from repeated_CV_builder import DCV

#%%

# imports the data
data = pd.read_csv("raw_data/train_call.txt", index_col=0, delimiter="\t").transpose()
topInfo = data.iloc[:3]
data = data.iloc[4:].dropna(axis=1)
classes = pd.read_csv("raw_data/train_clinical.txt", delimiter="\t", index_col=0)
data = classes.join(data).set_index("Subgroup").dropna()
# Add 1 to data because of chi-square feature selection


DCV.outer_repeats = 5
DCV.inner_repeats = 10
DCV.num_features = 150

model = linear_model.LogisticRegression(penalty="elasticnet", max_iter=10000, multi_class='ovr', solver='saga')
modeller = DCV(model)
# prints all the params you can change :)
print(model.get_params().keys())
# select which params you want to test with the inner loop
# space is the paramater space
modeller.hyperParams['C'] = [100, 10, 1.0, 0.1]
modeller.hyperParams['l1_ratio'] = [0.3, 0.5, 0.7, 0.9]

#%%

# this does everything for you :)
# first is the data used, seconds comes the classifications and than with loop the amount of loops you want to do

modeller.train_fit(data=data, classes=data.index, loop=100)
# some test code

print(modeller.test_accuracy.groupby(["C", "l1_ratio"])["values"].describe())


modelName = "LR"
modeller.save_all_data(modelName)
