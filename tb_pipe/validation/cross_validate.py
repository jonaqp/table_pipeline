from copy import copy
from typing import Optional, List, Union, Callable, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from category_encoders.utils import convert_input, convert_input_vector
from lightgbm import LGBMModel

from ..typing_template import TYPE_DATASET, TYPE_CV


class Trainer:

    def __init__(self, model: Union[LGBMModel, CatBoost]):
        self.model = model
        self.is_catboost = isinstance(model, CatBoost)
        self.is_classifier = getattr(model, "_estimator_type") == "classifier"

    def train(self,
              train_x: TYPE_DATASET, train_y: TYPE_DATASET,
              valid_x: Optional[TYPE_DATASET] = None,
              valid_y: Optional[TYPE_DATASET] = None,
              fit_params: Optional[Union[Dict[str, Any], Callable]] = None,
              cat_features: Optional[List[str]] = None):
        if fit_params is None:
            _fit_params = {}
        else:
            _fit_params = copy(fit_params)

        is_valid_training = valid_x is not None and valid_y is not None
        if self.is_catboost:
            trn_data = Pool(train_x, label=train_y, cat_features=cat_features)
            if is_valid_training:
                val_data = Pool(valid_x, label=valid_y,
                                cat_features=cat_features)
                self.__set_fit_params(_fit_params, val_data)
            self.model.fit(X=trn_data, **_fit_params)
        else:
            if is_valid_training:
                self.__set_fit_params(_fit_params, [(valid_x, valid_y)])
            self.model.fit(X=train_x, y=train_y, **_fit_params)

    @staticmethod
    def __set_fit_params(fit_params, valid_data):
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = valid_data
        if "early_stopping_rounds" not in fit_params:
            fit_params["early_stopping_rounds"] = 100
        return fit_params

    def _get_best_iteration(self):
        if self.is_catboost:
            return self.model.get_best_iteration()
        else:
            return self.model.best_iteration_

    def get_feature_importance(self):
        if self.is_catboost:
            return self.model.get_feature_importance()
        else:
            # LGBMClassifier, LGBMRegressor
            return self.model.feature_importances_

    def predict(self, test: TYPE_DATASET):
        if self.is_catboost:
            return self.model.predict(test)
        else:
            best_iter = self._get_best_iteration()
            return self.model.predict(test, num_iteration=best_iter)

    def predict_proba(self, test: TYPE_DATASET):
        if self.is_classifier:
            if self.is_catboost:
                return self.model.predict_proba(test)
            else:
                best_iter = self._get_best_iteration()
                return self.model.predict_proba(test, num_iteration=best_iter)
        else:
            error_msg = f"{type(self.model).__name__}" \
                        f" is not supported predict_proba method."
            raise NotImplementedError(error_msg)


class CrossValidator:
    def __init__(self, trainer: Trainer, cv: TYPE_CV,
                 scoring: Optional[Callable] = None):
        self.oof = None
        self.predictions = None
        self.cv = cv
        self.trainer = trainer
        self.scoring = scoring

    def run(self, train: TYPE_DATASET, test: TYPE_DATASET,
            target: TYPE_DATASET,
            groups: Optional[pd.Series] = None,
            verbose: bool = True):
        train = convert_input(train)
        target = convert_input_vector(target, train.index)

        if test is not None:
            test = convert_input(test)
            self.predictions = np.zeros(len(test))
        self.oof = np.zeros(len(train))

        scores = []
        for idx, (trn_idx, val_idx) in enumerate(
            self.cv.split(train, target, groups)):
            if verbose:
                print('Fold: {}/{}'.format(idx + 1, self.cv.n_splits))
                print('Length train: {} / valid: {}'.format(len(trn_idx),
                                                            len(val_idx)))
            train_x, train_y = train.iloc[trn_idx], target.iloc[trn_idx]
            valid_x, valid_y = train.iloc[val_idx], target.iloc[val_idx]

            self.trainer.train(train_x, train_y)

            self.oof[val_idx] = self.trainer.predict(valid_x)
            if test is not None:
                self.predictions += self.trainer.predict(test)
            if self.scoring is not None:
                score = self.scoring(valid_y, self.oof[val_idx])
                scores.append(score)
