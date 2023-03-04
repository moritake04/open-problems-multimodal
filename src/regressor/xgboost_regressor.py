import os

import joblib
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


class XGBoostRegressor:
    def __init__(
        self,
        cfg,
        train_X,
        train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        self.xgbr = MultiOutputRegressor(xgb.XGBRegressor(**self.cfg["xgb_params"]))

    def train(self):
        self.xgbr.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.xgbr,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.xgbr.predict(test_X)
        return preds


class XGBoostRegressorInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.xgbr = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.xgbr.predict(test_X)
        return preds
