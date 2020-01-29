from typing import Optional, List
from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool, CatBoostClassifier
from lightgbm import LGBMModel, LGBMClassifier
from sklearn.model_selection import BaseCrossValidator

TYPE_DATASET = Union[pd.DataFrame, np.ndarray]


class Trainer:

    def __init__(self, model: Union[LGBMModel, CatBoost]):
        self.model = model
        self.is_catboost = isinstance(model, CatBoost)
        self.is_classifier = None

    def train(self, train: TYPE_DATASET, target: TYPE_DATASET,
              cat_features: Optional[List[str]] = None):
        if self.is_catboost:
            trn_data = Pool(train, label=target, cat_features=cat_features)
            self.model.fit(X=trn_data)
        else:
            self.model.fit(X=train, y=target)

    def _get_best_iteration(self):
        if self.is_catboost:
            return self.model.get_best_iteration()
        else:
            return self.model.best_iteration_

    def predict(self, test: TYPE_DATASET):
        if self.is_catboost:
            return self.model.predict(test)
        else:
            return self.model.predict(test,
                                      num_iteration=self._get_best_iteration())


class CrossValidator:
    def __init__(self, model, cv: BaseCrossValidator):
        self.oof = None
        self.predictions = None
        self.cv = cv
        self.model = model

    def run(self, train: TYPE_DATASET, test: TYPE_DATASET,
            target: TYPE_DATASET, verbose: bool = True):
        self.oof = np.zeros(len(train))
        self.predictions = np.zeros(len(test))

        for idx, (trn_idx, val_idx) in enumerate(self.cv.split(train, target)):
            if verbose:
                print('Fold: {}/{}'.format(idx + 1, self.cv.n_splits))
                print('Length train: {} / valid: {}'.format(len(trn_idx),
                                                            len(val_idx)))
            train_x, train_y = train.iloc[trn_idx], target.iloc[trn_idx]
            valid_x, valid_y = train.iloc[val_idx], target.iloc[val_idx]
