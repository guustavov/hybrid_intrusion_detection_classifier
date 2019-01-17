import sys, os
import pandas as pd
import numpy as np
from dataSet import DataSet
from sklearn.model_selection import StratifiedKFold

dts = DataSet()
dts.setFilePath("../bases/CICIDS2017/10-folds/")
dts.setFileName("../bases/CICIDS2017/total_selectedFeatures.csv")
dts.loadData()

directory = os.path.dirname(dts.file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

#removing all instances that have 'label' value missing
dataset = dts.dataframe_data_set.dropna(subset=['label'])
names = dataset.columns

x = dataset.drop(['label'], axis = 1)
y = dataset.label

splitter = StratifiedKFold(n_splits = 10)

folds = []
i = 0
for indexes in splitter.split(x, y):
    fold = pd.DataFrame(dataset.values[indexes[1],], columns = names)
    folds.append(fold)

    fold.to_csv(dts.file_path + "fold_" + str(i+1) + ".csv", index = False)
    i = i + 1
