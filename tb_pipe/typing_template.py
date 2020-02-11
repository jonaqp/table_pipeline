from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

TYPE_DATASET = Union[pd.DataFrame, np.ndarray]
TYPE_CV = Optional[Union[int, Iterable, BaseCrossValidator]]
