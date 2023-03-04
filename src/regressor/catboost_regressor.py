import os

import catboost
import joblib


class CatBoostRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.catb = catboost.CatBoostRegressor(**self.cfg["xgb_params"])

    def train(self):
        self.catb.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.catb,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.catb.predict(test_X)
        return preds


class CatBoostRegressorInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.catb = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.catb.predict(test_X)
        return preds
