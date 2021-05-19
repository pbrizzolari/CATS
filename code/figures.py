# %%

import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%

RF_accuracies = pd.read_csv("RF_test_accuracy.csv", index_col=0)
LR_accuracies = pd.read_csv("LR_test_accuracy.csv", index_col=0)
knn_accuracies = pd.read_csv("Knn_test_accuracy.csv", index_col=0)
SVM_accuracies = pd.read_csv("SVM_test_accuracy.csv", index_col=0)

RF_precision = pd.read_csv("RF_test_precision.csv", index_col=0)
LR_precision = pd.read_csv("LR_test_precision.csv", index_col=0)
knn_precision = pd.read_csv("Knn_test_precision.csv", index_col=0)
SVM_precision = pd.read_csv("SVM_test_precision.csv", index_col=0)

RF_recall = pd.read_csv("RF_test_recall.csv", index_col=0)
LR_recall = pd.read_csv("LR_test_recall.csv", index_col=0)
knn_recall = pd.read_csv("Knn_test_recall.csv", index_col=0)
SVM_recall = pd.read_csv("SVM_test_recall.csv", index_col=0)

# %%

def formLatex(acc, prec, recall, index=None, header=[], add=False):
    if index == None:
        header_text = "\\hline " + " & ".join(["count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"
        table = []
    elif len(index) == 1:
        table = [str(x) for x in index] + ["{}"]
        index = tuple(index)[0]
        header_text = "\\hline " + " & ".join(
            header + ["{}", "count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"
    else:
        table = [str(x) for x in index]
        index = tuple(index)
        header_text = "\\hline " + " & ".join(header + ["count", "accuracy", "precision", "recall"]) + "\\\\ \\hline"

    for nr, i in enumerate([acc, prec, recall]):
        if index == None:
            stats = i["values"].agg(['mean', 'count', 'std'])
        else:
            stats = i.groupby(header).get_group(index)["values"].agg(['mean', 'count', 'std'])
        m, c, s = stats
        ci95_hi = m + 1.95 * s / math.sqrt(c)
        ci95_lo = m - 1.95 * s / math.sqrt(c)
        if nr == 0:
            table += [str(c)]
        table += ["{:.2f} ({:.2f}-{:.2f})".format(m, ci95_lo, ci95_hi)]
    text = ""
    # print(header_text)
    # print(table)
    table_text = " & ".join(table) + "\\\\"
    # print(table_text)
    if add:
        return table_text
    else:
        text += header_text.replace("_", " ") + "\n"
        text += table_text

    return text


print(formLatex(LR_accuracies, LR_precision, LR_recall, index=None))
print(formLatex(RF_accuracies, RF_precision, RF_recall, index=None, add=True))
print(formLatex(knn_accuracies, knn_precision, knn_recall, index=None, add=True))
print(formLatex(SVM_accuracies, SVM_precision, SVM_recall, index=None, add=True))

# %%

RF_acc_data = RF_accuracies["values"]
LR_acc_data = LR_accuracies["values"]
KNN_acc_data = knn_accuracies["values"]
SVM_acc_data = SVM_accuracies["values"]

RF_pre_data = RF_precision
LR_pre_data = LR_precision.groupby(["C", "l1_ratio"]).get_group((1.0, 0.9))["values"]
KNN_pre_data = knn_precision.groupby(["n_neighbors"]).get_group((7))["values"]
SVM_pre_data = SVM_precision.groupby(["C", 'kernel']).get_group((10, "rbf"))["values"]

RF_rec_data = RF_recall
LR_rec_data = LR_recall.groupby(["C", "l1_ratio"]).get_group((1.0, 0.9))["values"]
KNN_rec_data = knn_recall.groupby(["n_neighbors"]).get_group((7))["values"]
SVM_rec_data = SVM_recall.groupby(["C", 'kernel']).get_group((10, "rbf"))["values"]

# %%


df = pd.concat([RF_acc_data, LR_acc_data, KNN_acc_data, SVM_acc_data], axis=1)

df.columns = ["Random Forest", "Logistic\nRegression", "K-Nearest\nNeighbour", "SVM"]
df.describe()
ax = df.boxplot()
title_boxplot = 'Accuracy on repeated cross validation test data'
plt.title(title_boxplot)
plt.suptitle('')
ax.set_ylabel("Accuracy")  # that's what you're after
plt.show()

# %%



print(formLatex(LR_accuracies, LR_precision, LR_recall, index=[0.1, 0.0], header=["C", "l1_ratio"]))
for i in sorted(set(tuple(i) for i in LR_accuracies[["C", "l1_ratio"]].to_numpy().tolist())):
    print(formLatex(LR_accuracies, LR_precision, LR_recall, index=i, header=["C", "l1_ratio"], add=True))

# print(formLatex(knn_accuracies,knn_precision,knn_recall, index=[3],header=["n_neighbors"]))
for i in sorted(set(tuple(i) for i in knn_accuracies[["n_neighbors"]].to_numpy().tolist())):
    print(formLatex(knn_accuracies, knn_precision, knn_recall, index=i, header=["n_neighbors"], add=True))

# print(formLatex(SVM_accuracies,SVM_precision,SVM_recall, index=[10, "rbf"],header=["C",'kernel']))
for i in sorted(set(tuple(i) for i in SVM_accuracies[["C", 'kernel']].to_numpy().tolist())):
    print(formLatex(SVM_accuracies, SVM_precision, SVM_recall, index=i, header=["C", 'kernel'], add=True))

# print(formLatex(RF_accuracies,RF_precision,RF_recall, index=["sqrt",250],header=["max_features","n_estimators"]))
for i in sorted(set(tuple(i) for i in RF_accuracies[["max_features", "n_estimators"]].to_numpy().tolist())):
    print(formLatex(RF_accuracies, RF_precision, RF_recall, index=i, header=["max_features", "n_estimators"], add=True))

# %%

pd.DataFrame([RF_acc_data.describe(), LR_acc_data.describe(), KNN_acc_data.describe(), SVM_acc_data.describe()],
             index=["RF", "LR", "KNN", "SVM"])
