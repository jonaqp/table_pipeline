from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

TYPE_DATASET = Union[pd.DataFrame, np.ndarray]


class CrossValidator:
    def __init__(self, model, cv: BaseCrossValidator):
        self.oof = None
        self.predictions = None
        self.cv = cv
        self.model = model

    def run(self, train: TYPE_DATASET, test: TYPE_DATASET,
            target: TYPE_DATASET):
        self.oof = np.zeros(len(train))
        self.predictions = np.zeros(len(test))

        for fold_idx, (trn_idx, val_idx) in enumerate(
            self.cv.split(train, target)):
            print('Fold {}/{}'.format(fold_idx + 1, self.cv.n_splits))
            train_x, train_y = train.iloc[trn_idx], target.iloc[trn_idx]
            valid_x, valid_y = train.iloc[val_idx], target.iloc[val_idx]
