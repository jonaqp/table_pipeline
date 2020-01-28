import os

# https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/utils.py#L17
_is_kaggle: bool = 'KAGGLE_WORKING_DIR' in os.environ


def is_kaggle():
    return _is_kaggle
