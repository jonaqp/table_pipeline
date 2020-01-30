import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score

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
def test_trainer_predict(model):
    from sklearn.datasets import load_wine
    x, y = load_wine(True)

    trainer = Trainer(model)
    trainer.train(train=x, target=y)
    predicts = trainer.predict(x)
    score = accuracy_score(y, predicts)
    assert score >= 0.9


@pytest.mark.parametrize("model", list(get_classification_models()))
def test_trainer_predict_proba(model):
    import numpy as np
    from sklearn.datasets import load_wine
    x, y = load_wine(True)
    trainer = Trainer(model)
    trainer.train(train=x, target=y)
    predicts = trainer.predict_proba(x)
    predicts = np.argmax(predicts, axis=1)
    score = accuracy_score(y, predicts)
    assert score >= 0.9


@pytest.mark.parametrize("model", list(get_regression_models()))
def test_trainer_predict_proba_with_error(model):
    from sklearn.datasets import load_boston

    x, y = load_boston(True)
    trainer = Trainer(model)
    trainer.train(x, y)
    with pytest.raises(NotImplementedError):
        trainer.predict_proba(y)
