import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from tb_pipe.experiment.experiment import run
from tb_pipe.validation.cross_validate import Trainer


def get_classification_models():
    models = [LGBMClassifier(),
              CatBoostClassifier(verbose=0, save_snapshot=False)]
    for _model in models:
        yield _model


def get_regression_models():
    models = [LGBMRegressor(),
              CatBoostRegressor(verbose=0, save_snapshot=False)]
    for _model in models:
        yield _model


@pytest.mark.parametrize("model", list(get_classification_models()))
def test_classification_run(model):
    from sklearn.datasets import load_wine
    x, y = load_wine(True)

    trainer = Trainer(model)
    n_splits = 2
    kf = KFold(n_splits)
    tr_x, val_x, tr_y, val_y = train_test_split(x, y)
    oof, predictions, feature_imp = run(trainer, tr_x, val_x, tr_y,
                                        scoring=accuracy_score, cv=kf)
    assert len(oof) == len(tr_x)
    assert len(predictions) == len(val_y)


@pytest.mark.parametrize("model", list(get_regression_models()))
def test_regression_run(model):
    from sklearn.datasets import load_boston
    x, y = load_boston(True)

    trainer = Trainer(model)
    n_splits = 2
    kf = KFold(n_splits)
    tr_x, val_x, tr_y, val_y = train_test_split(x, y)
    oof, predictions, feature_imp = run(trainer, tr_x, val_x, tr_y,
                                        scoring=mean_squared_error, cv=kf)
    assert len(oof) == len(tr_x)
    assert len(predictions) == len(val_y)
