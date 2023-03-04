import os

import joblib
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge


class KernelRidgeRBFRegressor:
    def __init__(
        self, cfg, train_X, train_y,
    ):
        self.train_X = train_X
        self.train_y = train_y
        self.cfg = cfg
        kernel = RBF(length_scale=10)
        self.kernelridge = KernelRidge(kernel=kernel, **self.cfg["ridge_params"])

    def train(self):
        self.kernelridge.fit(self.train_X, self.train_y)
        if self.cfg["model_save"]:
            os.makedirs(f"../weights/{self.cfg['general']['save_name']}", exist_ok=True)
            joblib.dump(
                self.ridge,
                f"../weights/{self.cfg['general']['save_name']}/fold{self.cfg['fold_n']}.ckpt",
                compress=3,
            )

    def predict(self, test_X):
        preds = self.kernelridge.predict(test_X)
        return preds


class KernelRidgeRBFRegressorInference:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg
        self.kernelridge = joblib.load(weight_path)

    def predict(self, test_X):
        preds = self.kernelridge.predict(test_X)
        return preds
