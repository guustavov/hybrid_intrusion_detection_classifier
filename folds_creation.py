import sys, os
import pandas as pd
import numpy as np
from dataSet import DataSet
from sklearn.model_selection import StratifiedKFold

dts = DataSet()
dts.setFilePath("../cicids2017/10-folds/")
dts.setFileName("../cicids2017/total_selectedFeatures.csv")
dts.loadData()

directory = os.path.dirname(dts.file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

dataset = dts.dataframe_data_set

classFeatureName = dataset.columns[len(dataset.columns) - 1]

#removing all instances that have no class value
dataset = dts.dataframe_data_set.dropna(subset=[classFeatureName])
names = dataset.columns

x = dataset.drop([classFeatureName], axis = 1)
y = getattr(dataset, classFeatureName).values

splitter = StratifiedKFold(n_splits = 10)

folds = []
i = 0
for indexes in splitter.split(x, y):
    fold = pd.DataFrame(dataset.values[indexes[1],], columns = names)

    #define filter to transform non "benign" events to "malign"
    notBenignFilter = fold[classFeatureName] != "BENIGN"
    fold.loc[notBenignFilter, classFeatureName] = "MALIGN"

    folds.append(fold)

    fold.to_csv(dts.file_path + "fold_" + str(i+1) + ".csv", index = False)
    i = i + 1
