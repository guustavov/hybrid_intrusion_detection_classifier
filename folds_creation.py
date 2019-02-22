import sys, os
import pandas as pd
import numpy as np
from dataSet import DataSet
from sklearn.model_selection import StratifiedKFold

def binarizeDataset(dataset, classFeatureName):
    benignFilter = dataset[classFeatureName] == "BENIGN"
    notBenignFilter = dataset[classFeatureName] != "BENIGN"

    dataset.loc[benignFilter, classFeatureName] = 0
    dataset.loc[notBenignFilter, classFeatureName] = 1
    
    return dataset

def splitDataset(dataset, classFeatureName, numberOfSplits = 10):
    names = dataset.columns

    x = dataset.drop([classFeatureName], axis = 1) #all instances with no class feature
    y = getattr(dataset, classFeatureName).values #class feature of all instances

    splitter = StratifiedKFold(n_splits = 10)

    folds = []
    for indexes in splitter.split(x, y):
        folds.append(pd.DataFrame(dataset.values[indexes[1],], columns = names))

    return folds

def writeFoldToCsv(fold, foldIndex, destinationPath):
    fold.to_csv(destinationPath + "fold_" + str(foldIndex) + ".csv", index = False)


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

dataset = binarizeDataset(dataset, classFeatureName)

folds = splitDataset(dataset, classFeatureName, 10)

#using only 10% of the original dataset
folds = splitDataset(folds[0], classFeatureName, 10)

for index, fold in enumerate(folds):
    writeFoldToCsv(fold, index, dts.file_path)
