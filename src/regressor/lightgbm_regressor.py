import os

import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor


class LightGBMRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.lgbm = MultiOutputRegressor(lgb.LGBMRegressor(**self.cfg["lgbm_params"]))

    def train(self):
        self.lgbm.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.lgbm,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.lgbm.predict(test_X)
        return preds


class LightGBMRegressorInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.lgbm = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.lgbm.predict(test_X)
        return preds
