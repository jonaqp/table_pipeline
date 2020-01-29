from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from tb_pipe.validation.cross_validate import Trainer


def test_trainer_classification():
    from sklearn.datasets import load_wine
    x, y = load_wine(True)

    trainer = Trainer(LGBMClassifier())
    trainer.train(train=x, target=y)
    predicts = trainer.predict(x)
    score = accuracy_score(y, predicts)
    assert score >= 0.9

    trainer = Trainer(CatBoostClassifier())
    trainer.train(train=x, target=y)
    predicts = trainer.predict(x)
    score = accuracy_score(y, predicts)
    assert score >= 0.9
