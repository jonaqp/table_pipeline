from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from ..typing_template import TYPE_DATASET, TYPE_CV


def run(estimator: BaseEstimator,
        x: TYPE_DATASET, y: Optional[TYPE_DATASET] = None,
        scoring=None,
        cv: TYPE_CV = None,
        groups: Optional[pd.Series] = None):
    pass
