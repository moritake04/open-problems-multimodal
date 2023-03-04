import os

import joblib
from sklearn.linear_model import Ridge


class RidgeRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.ridge = Ridge(**self.cfg["ridge_params"], copy_X=False)

    def train(self):
        self.ridge.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.ridge,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.ridge.predict(test_X)
        return preds


class RidgeRegressorInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.ridge = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.ridge.predict(test_X)
        return preds
