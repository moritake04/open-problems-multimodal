import argparse
import gc

import numpy as np
import pandas as pd
import scipy
import sklearn.metrics as metrics
import torch
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from regressor import (
    CatBoostRegressorInference,
    LightGBMRegressorInference,
    MLPLargeRegressorInference,
    MLPRegressorInference,
    OneDCNNRegressorInference,
    RidgeRegressorInference,
    XGBoostRegressorInference,
)


def correlation_score(y_true, y_pred):
    if type(y_true) == pd.DataFrame:
        y_true = y_true.values
    if type(y_pred) == pd.DataFrame:
        y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument(
        "-s", "--save_sub", action="store_true", help="to save or not to save"
    )
    args = parser.parse_args()
    return args


def inference(cfg, input_size, output_size, test_X, valid_X=None):
    if cfg["model"] == "mlp":
        model = MLPRegressorInference(
            cfg,
            input_size,
            output_size,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )
    elif cfg["model"] == "largemlp":
        model = MLPLargeRegressorInference(
            cfg,
            input_size,
            output_size,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )
    elif cfg["model"] == "cnn":
        model = OneDCNNRegressorInference(
            cfg,
            input_size,
            output_size,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )
    elif cfg["model"] == "ridge":
        if cfg["task"] == "multi":
            test_preds = 0
            valid_preds = 0
            for n in range(5):
                model = RidgeRegressorInference(
                    cfg,
                    f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_{n}.ckpt",
                )
                test_preds += model.predict(test_X)
                if valid_X is not None:
                    valid_preds += model.predict(valid_X)
            if valid_X is None:
                return test_preds
            else:
                return test_preds, valid_preds
        else:
            model = RidgeRegressorInference(
                cfg,
                f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
            )
    elif cfg["model"] == "lgbm":
        model = LightGBMRegressorInference(
            cfg,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )

    elif cfg["model"] == "xgboost":
        model = XGBoostRegressorInference(
            cfg,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )

    elif cfg["model"] == "catboost":
        model = CatBoostRegressorInference(
            cfg,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )

    test_preds = model.predict(test_X)

    if valid_X is None:
        return test_preds
    else:
        valid_preds = model.predict(valid_X)
        return test_preds, valid_preds


def hold_out(cfg, valid_X, valid_y, test_X):
    print(f"[hold out] start")
    seed_everything(cfg["general"]["seed"], workers=True)

    # inference
    test_preds, valid_preds = inference(
        cfg, valid_X.shape[1], valid_y.shape[1], test_X, valid_X
    )

    mse = metrics.mean_squared_error(valid_y.todense(), valid_preds)
    corr = correlation_score(valid_y.todense(), valid_preds)
    print(f"[hold out] finished, mse:{mse}, corrscore:{corr}")

    torch.cuda.empty_cache()

    return valid_preds, test_preds, mse, corr


def one_fold(fold_n, skf, cfg, train_X, train_y, test_X):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X[valid_indices],
        train_y[valid_indices],
    )

    # inference
    test_preds, valid_preds = inference(
        cfg, valid_X_cv.shape[1], valid_y_cv.shape[1], test_X, valid_X_cv
    )

    mse = metrics.mean_squared_error(valid_y_cv.todense(), valid_preds)
    corr = correlation_score(valid_y_cv.todense(), valid_preds)
    print(f"[fold_{fold_n}] finished, mse:{mse}, corrscore:{corr}")

    torch.cuda.empty_cache()

    return valid_preds, test_preds, mse, corr


def one_fold_gkf(fold_n, gkf, groups, cfg, train_X, train_y, test_X):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(gkf.split(train_X, train_y, groups=groups))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X[valid_indices],
        train_y[valid_indices],
    )

    # inference
    test_preds, valid_preds = inference(
        cfg, valid_X_cv.shape[1], valid_y_cv.shape[1], test_X, valid_X_cv
    )

    mse = metrics.mean_squared_error(valid_y_cv.todense(), valid_preds)
    corr = correlation_score(valid_y_cv.todense(), valid_preds)
    print(f"[fold_{fold_n}] finished, mse:{mse}, corrscore:{corr}")

    torch.cuda.empty_cache()

    return valid_preds, test_preds, mse, corr


def all_train(cfg, train_X, train_y, test_X):
    print("[all_train] start")

    seed_everything(cfg["general"]["seed"], workers=True)

    # inference
    test_preds = inference(cfg, train_X.shape[1], train_y.shape[1], test_X)

    torch.cuda.empty_cache()

    return test_preds


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {[args.fold]}")
    print(f"pred_save: {args.save_sub}")

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read h5
    svd_dim = cfg["svd_dim"]
    if cfg["task"] == "cite":
        if cfg["data_type"] == "svd":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
        elif cfg["data_type"] == "svd_important_ambrosm":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}_important_ambrosm.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}_important_ambrosm.npz"
            )
        elif cfg["data_type"] == "svd_important_laurent":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}_important_laurent.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}_important_laurent.npz"
            )
        elif cfg["data_type"] == "svd_important_ambrosm_and_laurent":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}_important_ambrosm_and_laurent.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}_important_ambrosm_and_laurent.npz"
            )
        elif cfg["data_type"] == "raw":
            # raw data
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            raw_train_X_values = scipy.sparse.load_npz(
                "../data/input/compressed/train_cite_inputs_values.sparse.npz"
            )
            raw_test_X_values = scipy.sparse.load_npz(
                "../data/input/compressed/test_cite_inputs_values.sparse.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, raw_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, raw_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_10":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_cite_abg_abs_cor_10_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_cite_abg_abs_cor_10_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_50":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_cite_abg_abs_cor_50_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_cite_abg_abs_cor_50_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_100":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_cite_abg_abs_cor_100_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_cite_abg_abs_cor_100_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_500":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_cite_abg_abs_cor_500_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_cite_abg_abs_cor_500_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "umap50":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            umap = pd.read_csv(
                "../data/input/map/citeseq_UMAPfromPCA500_n_components50_n_neighbors5_min_dist0d9_rs42.csv"
            ).drop(["cell_id"], axis=1)
            umap_train_X_values = umap[:70988]
            umap_test_X_values = umap[70988:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, umap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, umap_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "trimap":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}.npz"
            )
            trimap = pd.read_csv(
                "../data/input/map/citeseq_TRIMAPfromPCA500_n_components50.csv"
            ).drop(["cell_id"], axis=1)
            trimap_train_X_values = trimap[:70988]
            trimap_test_X_values = trimap[70988:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, trimap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, trimap_test_X_values], format="csr"
            )
        train_X_idxcol = np.load(
            "../data/input/compressed/train_cite_inputs_idxcol.npz", allow_pickle=True
        )
        train_y_values = scipy.sparse.load_npz(
            "../data/input/compressed/train_cite_targets_values.sparse.npz"
        )
        train_y_idxcol = np.load(
            "../data/input/compressed/train_cite_targets_idxcol.npz", allow_pickle=True
        )
        test_X_idxcol = np.load(
            "../data/input/compressed/test_cite_inputs_idxcol.npz", allow_pickle=True
        )
    else:
        if cfg["data_type"] == "svd":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
        elif cfg["data_type"] == "raw":
            # raw data
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            raw_train_X_values = scipy.sparse.load_npz(
                "../data/input/compressed/train_multi_inputs_values.sparse.npz"
            )
            raw_test_X_values = scipy.sparse.load_npz(
                "../data/input/compressed/test_multi_inputs_values.sparse.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, raw_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, raw_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_10":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_multi_abg_abs_cor_10_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_multi_abg_abs_cor_10_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_50":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_multi_abg_abs_cor_50_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_multi_abg_abs_cor_50_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_100":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_multi_abg_abs_cor_100_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_multi_abg_abs_cor_100_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "corr_500":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            corr_train_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/train_multi_abg_abs_cor_500_inputs.npz"
            )
            corr_test_X_values = scipy.sparse.load_npz(
                f"../data/input/correlation/test_multi_abg_abs_cor_500_inputs.npz"
            )
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, corr_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, corr_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "umap10":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            umap = pd.read_csv(
                "../data/input/map/multiome_UMAPfromPCA500_n_components10_n_neighbors5_min_dist0d9_rs42.csv"
            ).drop(["Unnamed: 0"], axis=1)
            umap_train_X_values = umap[:105942]
            umap_test_X_values = umap[105942:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, umap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, umap_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "umap50":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            umap = pd.read_csv(
                "../data/input/map/multiome_UMAPfromPCA500_n_components50_n_neighbors5_min_dist0d9_rs42.csv"
            ).drop(["Unnamed: 0"], axis=1)
            umap_train_X_values = umap[:105942]
            umap_test_X_values = umap[105942:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, umap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, umap_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "umap100":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            umap = pd.read_csv(
                "../data/input/map/multiome_UMAPfromPCA500_n_components100_n_neighbors5_min_dist0d9_rs42.csv"
            ).drop(["Unnamed: 0"], axis=1)
            umap_train_X_values = umap[:105942]
            umap_test_X_values = umap[105942:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, umap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, umap_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "trimap10":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            trimap = pd.read_csv(
                "../data/input/map/multiome_TRIMAPfromPCA500_n_components10.csv"
            ).drop(["Unnamed: 0"], axis=1)
            trimap_train_X_values = trimap[:105942]
            trimap_test_X_values = trimap[105942:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, trimap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, trimap_test_X_values], format="csr"
            )
        elif cfg["data_type"] == "trimap50":
            svd_train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            svd_test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
            trimap = pd.read_csv(
                "../data/input/map/multiome_TRIMAPfromPCA500_n_components50.csv"
            ).drop(["Unnamed: 0"], axis=1)
            trimap_train_X_values = trimap[:105942]
            trimap_test_X_values = trimap[105942:]
            train_X_values = scipy.sparse.hstack(
                [svd_train_X_values, trimap_train_X_values], format="csr"
            )
            test_X_values = scipy.sparse.hstack(
                [svd_test_X_values, trimap_test_X_values], format="csr"
            )
        train_X_idxcol = np.load(
            "../data/input/compressed/train_multi_inputs_idxcol.npz", allow_pickle=True
        )
        train_y_values = scipy.sparse.load_npz(
            "../data/input/compressed/train_multi_targets_values.sparse.npz"
        )
        train_y_idxcol = np.load(
            "../data/input/compressed/train_multi_targets_idxcol.npz", allow_pickle=True
        )
        test_X_idxcol = np.load(
            "../data/input/compressed/test_multi_inputs_idxcol.npz", allow_pickle=True
        )
    metadata = pd.read_csv("../data/input/metadata.csv", index_col="cell_id")
    metadata["gender"] = 0
    metadata[metadata["donor"] == "13176"]["gender"] = 1
    print("read data")

    # reindex
    train_metadata = metadata.reindex(train_X_idxcol["index"])
    test_metadata = metadata.reindex(test_X_idxcol["index"])

    # inputs 日付ごとにデータを標準化（batch effect対策） -> increased LB
    if cfg["day_std_inputs"]:
        if cfg["task"] == "cite":
            # retrieve each day data
            train_X_values_day2, test_X_values_day2 = (
                train_X_values[train_metadata["day"] == 2],
                test_X_values[test_metadata["day"] == 2],
            )
            train_X_values_day3, test_X_values_day3 = (
                train_X_values[train_metadata["day"] == 3],
                test_X_values[test_metadata["day"] == 3],
            )
            train_X_values_day4, test_X_values_day4 = (
                train_X_values[train_metadata["day"] == 4],
                test_X_values[test_metadata["day"] == 4],
            )
            test_X_values_day7 = test_X_values[test_metadata["day"] == 7]

            # std
            # day2
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day2, test_X_values_day2], format="csr"
                )
            )
            train_X_values_day2, test_X_values_day2 = ss.transform(
                train_X_values_day2
            ), ss.transform(test_X_values_day2)
            # day3
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day3, test_X_values_day3], format="csr"
                )
            )
            train_X_values_day3, test_X_values_day3 = ss.transform(
                train_X_values_day3
            ), ss.transform(test_X_values_day3)
            # day4
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day4, test_X_values_day4], format="csr"
                )
            )
            train_X_values_day4, test_X_values_day4 = ss.transform(
                train_X_values_day4
            ), ss.transform(test_X_values_day4)
            # day7
            ss = StandardScaler(with_mean=False)
            ss.fit(test_X_values_day7)
            test_X_values_day7 = ss.transform(test_X_values_day7)

            # assign
            (
                train_X_values[train_metadata["day"] == 2],
                test_X_values[test_metadata["day"] == 2],
            ) = (train_X_values_day2, test_X_values_day2)
            (
                train_X_values[train_metadata["day"] == 3],
                test_X_values[test_metadata["day"] == 3],
            ) = (train_X_values_day3, test_X_values_day3)
            (
                train_X_values[train_metadata["day"] == 4],
                test_X_values[test_metadata["day"] == 4],
            ) = (train_X_values_day4, test_X_values_day4)
            test_X_values[test_metadata["day"] == 7] = test_X_values_day7
            del train_X_values_day2, train_X_values_day3, train_X_values_day4
            del (
                test_X_values_day2,
                test_X_values_day3,
                test_X_values_day4,
                test_X_values_day7,
            )
            gc.collect()
        else:
            # retrieve each day data
            train_X_values_day2, test_X_values_day2 = (
                train_X_values[train_metadata["day"] == 2],
                test_X_values[test_metadata["day"] == 2],
            )
            train_X_values_day3, test_X_values_day3 = (
                train_X_values[train_metadata["day"] == 3],
                test_X_values[test_metadata["day"] == 3],
            )
            train_X_values_day4 = train_X_values[train_metadata["day"] == 4]
            train_X_values_day7, test_X_values_day7 = (
                train_X_values[train_metadata["day"] == 7],
                test_X_values[test_metadata["day"] == 7],
            )
            test_X_values_day10 = test_X_values[test_metadata["day"] == 10]

            # std
            # day2
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day2, test_X_values_day2], format="csr"
                )
            )
            train_X_values_day2, test_X_values_day2 = ss.transform(
                train_X_values_day2
            ), ss.transform(test_X_values_day2)
            # day3
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day3, test_X_values_day3], format="csr"
                )
            )
            train_X_values_day3, test_X_values_day3 = ss.transform(
                train_X_values_day3
            ), ss.transform(test_X_values_day3)
            # day4
            ss = StandardScaler(with_mean=False)
            ss.fit(train_X_values_day4)
            train_X_values_day4 = ss.transform(train_X_values_day4)
            # day7
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day7, test_X_values_day7], format="csr"
                )
            )
            train_X_values_day7, test_X_values_day7 = ss.transform(
                train_X_values_day7
            ), ss.transform(test_X_values_day7)
            # day10
            ss = StandardScaler(with_mean=False)
            ss.fit(test_X_values_day10)
            test_X_values_day10 = ss.transform(test_X_values_day10)

            # assign
            (
                train_X_values[train_metadata["day"] == 2],
                test_X_values[test_metadata["day"] == 2],
            ) = (train_X_values_day2, test_X_values_day2)
            (
                train_X_values[train_metadata["day"] == 3],
                test_X_values[test_metadata["day"] == 3],
            ) = (train_X_values_day3, test_X_values_day3)
            train_X_values[train_metadata["day"] == 4] = train_X_values_day4
            (
                train_X_values[train_metadata["day"] == 7],
                test_X_values[test_metadata["day"] == 7],
            ) = (train_X_values_day7, test_X_values_day7)
            test_X_values[test_metadata["day"] == 10] = test_X_values_day10
            del (
                train_X_values_day2,
                train_X_values_day3,
                train_X_values_day4,
                train_X_values_day7,
            )
            del (
                test_X_values_day2,
                test_X_values_day3,
                test_X_values_day7,
                test_X_values_day10,
            )
            gc.collect()

    # inputs 日付とdonorごとにデータを標準化（batch effect対策）
    if cfg["day_donor_std_inputs"]:
        if cfg["task"] == "cite":
            # retrieve each day data
            train_dict = {}
            test_dict = {}
            # train
            for day in [2, 3, 4]:
                for donor in [13176, 31800, 32606]:
                    train_dict[f"{day}_{donor}"] = train_X_values[
                        (train_metadata["day"] == day)
                        & (train_metadata["donor"] == donor)
                    ]
            # test
            day = 7
            for donor in [13176, 31800, 32606, 27678]:
                test_dict[f"{day}_{donor}"] = test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ]
            donor = 27678
            for day in [2, 3, 4]:
                test_dict[f"{day}_{donor}"] = test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ]

            # std
            # train
            for day in [2, 3, 4]:
                for donor in [13176, 31800, 32606]:
                    ss = StandardScaler(with_mean=False)
                    train_dict[f"{day}_{donor}"] = ss.fit_transform(
                        train_dict[f"{day}_{donor}"]
                    )
            day = 7
            # test
            for donor in [13176, 31800, 32606, 27678]:
                ss = StandardScaler(with_mean=False)
                test_dict[f"{day}_{donor}"] = ss.fit_transform(
                    test_dict[f"{day}_{donor}"]
                )
            donor = 27678
            for day in [2, 3, 4]:
                ss = StandardScaler(with_mean=False)
                test_dict[f"{day}_{donor}"] = ss.fit_transform(
                    test_dict[f"{day}_{donor}"]
                )

            # assign
            # train
            for day in [2, 3, 4]:
                for donor in [13176, 31800, 32606]:
                    train_X_values[
                        (train_metadata["day"] == day)
                        & (train_metadata["donor"] == donor)
                    ] = train_dict[f"{day}_{donor}"]
            # test
            day = 7
            for donor in [13176, 31800, 32606, 27678]:
                test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ] = test_dict[f"{day}_{donor}"]
            donor = 27678
            for day in [2, 3, 4]:
                test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ] = test_dict[f"{day}_{donor}"]

            del train_dict, test_dict
            gc.collect()
        else:
            # retrieve each day data
            train_dict = {}
            test_dict = {}
            # train
            for day in [2, 3, 4, 7]:
                for donor in [13176, 31800, 32606]:
                    train_dict[f"{day}_{donor}"] = train_X_values[
                        (train_metadata["day"] == day)
                        & (train_metadata["donor"] == donor)
                    ]
            # test
            day = 10
            for donor in [13176, 31800, 32606, 27678]:
                test_dict[f"{day}_{donor}"] = test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ]
            donor = 27678
            for day in [2, 3, 7]:
                test_dict[f"{day}_{donor}"] = test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ]

            # std
            # train
            for day in [2, 3, 4, 7]:
                for donor in [13176, 31800, 32606]:
                    ss = StandardScaler(with_mean=False)
                    train_dict[f"{day}_{donor}"] = ss.fit_transform(
                        train_dict[f"{day}_{donor}"]
                    )
            day = 10
            # test
            for donor in [13176, 31800, 32606, 27678]:
                ss = StandardScaler(with_mean=False)
                test_dict[f"{day}_{donor}"] = ss.fit_transform(
                    test_dict[f"{day}_{donor}"]
                )
            donor = 27678
            for day in [2, 3, 7]:
                ss = StandardScaler(with_mean=False)
                test_dict[f"{day}_{donor}"] = ss.fit_transform(
                    test_dict[f"{day}_{donor}"]
                )

            # assign
            # train
            for day in [2, 3, 4, 7]:
                for donor in [13176, 31800, 32606]:
                    train_X_values[
                        (train_metadata["day"] == day)
                        & (train_metadata["donor"] == donor)
                    ] = train_dict[f"{day}_{donor}"]
            # test
            day = 10
            for donor in [13176, 31800, 32606, 27678]:
                test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ] = test_dict[f"{day}_{donor}"]
            donor = 27678
            for day in [2, 3, 7]:
                test_X_values[
                    (test_metadata["day"] == day) & (test_metadata["donor"] == donor)
                ] = test_dict[f"{day}_{donor}"]

            del train_dict, test_dict
            gc.collect()

    # targets 日付ごとにデータを標準化（ドメインシフト対策？） -> ゴミになる
    if cfg["day_std_targets"]:
        train_y_values = train_y_values.todense()
        if cfg["task"] == "cite":
            # retrieve each day data
            train_y_values_day2 = train_y_values[train_metadata["day"] == 2]
            train_y_values_day3 = train_y_values[train_metadata["day"] == 3]
            train_y_values_day4 = train_y_values[train_metadata["day"] == 4]

            # std
            # day2
            ss = StandardScaler(with_mean=False)
            train_y_values_day2 = ss.fit_transform(train_y_values_day2)
            # day3
            ss = StandardScaler(with_mean=False)
            train_y_values_day3 = ss.fit_transform(train_y_values_day3)
            # day4
            ss = StandardScaler(with_mean=False)
            train_y_values_day4 = ss.fit_transform(train_y_values_day4)

            # assign
            train_y_values[train_metadata["day"] == 2] = train_y_values_day2
            train_y_values[train_metadata["day"] == 3] = train_y_values_day3
            train_y_values[train_metadata["day"] == 4] = train_y_values_day4
            del train_y_values_day2, train_y_values_day3, train_y_values_day4
            gc.collect()
        else:
            # retrieve each day data
            train_y_values_day2 = train_y_values[train_metadata["day"] == 2]
            train_y_values_day3 = train_y_values[train_metadata["day"] == 3]
            train_y_values_day4 = train_y_values[train_metadata["day"] == 4]
            train_y_values_day7 = train_y_values[train_metadata["day"] == 7]

            # std
            # day2
            ss = StandardScaler(with_mean=False)
            train_y_values_day2 = ss.fit_transform(train_y_values_day2)
            # day3
            ss = StandardScaler(with_mean=False)
            train_y_values_day3 = ss.fit_transform(train_y_values_day3)
            # day4
            ss = StandardScaler(with_mean=False)
            train_y_values_day4 = ss.fit_transform(train_y_values_day4)
            # day7
            ss = StandardScaler(with_mean=False)
            train_y_values_day7 = ss.fit_transform(train_y_values_day7)

            # assign
            train_y_values[train_metadata["day"] == 2] = train_y_values_day2
            train_y_values[train_metadata["day"] == 3] = train_y_values_day3
            train_y_values[train_metadata["day"] == 4] = train_y_values_day4
            train_y_values[train_metadata["day"] == 7] = train_y_values_day7
            del (
                train_y_values_day2,
                train_y_values_day3,
                train_y_values_day4,
                train_y_values_day7,
            )
            gc.collect()
        train_y_values = scipy.sparse.csr_matrix(train_y_values)

    # add metadata (cell_type: 7 categories)
    if cfg["use_meta"] == "day_type":
        train_meta = train_metadata.drop(["donor", "technology", "day"], axis=1)
        test_meta = test_metadata.drop(["donor", "technology", "day"], axis=1)
        # one-hot encoding (cell_type)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        train_meta = scipy.sparse.csr_matrix(ohe.fit_transform(train_meta.values))
        test_meta = scipy.sparse.csr_matrix(ohe.transform(test_meta.values))
        # retrieve day
        train_day = train_metadata["day"].values.reshape(-1, 1)
        test_day = test_metadata["day"].values.reshape(-1, 1)
        # 水平結合
        train_X_values = scipy.sparse.hstack(
            [train_X_values, train_day, train_meta], format="csr"
        )
        test_X_values = scipy.sparse.hstack(
            [test_X_values, test_day, test_meta], format="csr"
        )
    elif cfg["use_meta"] == "day_only":
        # retrieve day
        train_day = train_metadata["day"].values.reshape(-1, 1)
        test_day = test_metadata["day"].values.reshape(-1, 1)
        # 水平結合
        train_X_values = scipy.sparse.hstack([train_X_values, train_day], format="csr")
        test_X_values = scipy.sparse.hstack([test_X_values, test_day], format="csr")
    elif cfg["use_meta"] == "type_only":
        train_meta = train_metadata.drop(["donor", "technology", "day"], axis=1)
        test_meta = test_metadata.drop(["donor", "technology", "day"], axis=1)
        # one-hot encoding (cell_type)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        train_meta = scipy.sparse.csr_matrix(ohe.fit_transform(train_meta.values))
        test_meta = scipy.sparse.csr_matrix(ohe.transform(test_meta.values))
        # 水平結合
        train_X_values = scipy.sparse.hstack([train_X_values, train_meta], format="csr")
        test_X_values = scipy.sparse.hstack([test_X_values, test_meta], format="csr")

    # add gender
    if cfg["gender"]:
        train_X_values = scipy.sparse.hstack(
            [train_X_values, train_metadata["gender"].values.reshape(-1, 1)],
            format="csr",
        )
        test_X_values = scipy.sparse.hstack(
            [test_X_values, test_metadata["gender"].values.reshape(-1, 1)],
            format="csr",
        )

    if cfg["general"]["cv"] == "donor":
        # group k-fold cross-validation
        n_splits = len(train_metadata["donor"].unique())
        gkf = GroupKFold(n_splits=n_splits)
        test_pred_list = []
        valid_mse_list = []
        valid_corr_list = []
        if args.fold is not None:
            for_list = [args.fold]
        else:
            for_list = range(n_splits)
        for fold_n in tqdm(for_list):
            cfg["fold_n"] = fold_n
            _, test_preds, mse, corr = one_fold_gkf(
                fold_n,
                gkf,
                train_metadata["donor"],
                cfg,
                train_X_values,
                train_y_values,
                test_X_values,
            )
            test_pred_list.append(test_preds)
            valid_mse_list.append(mse)
            valid_corr_list.append(corr)

        valid_mse_mean = np.mean(valid_mse_list, axis=0)
        valid_corr_mean = np.mean(valid_corr_list, axis=0)
        print(f"cv mean mse:{valid_mse_mean}, corr:{valid_corr_mean}")

        final_test_pred = np.mean(test_pred_list, axis=0)
    elif cfg["general"]["cv"] == "day":
        # day holdout
        cfg["fold_n"] = "day_hold_out"
        split_day = 4 if cfg["task"] == "cite" else 7

        train_X_cv, train_y_cv = (
            train_X_values[train_metadata["day"] != split_day],
            train_y_values[train_metadata["day"] != split_day],
        )
        valid_X_cv, valid_y_cv = (
            train_X_values[train_metadata["day"] == split_day],
            train_y_values[train_metadata["day"] == split_day],
        )

        _, test_preds, mse, corr = hold_out(
            cfg,
            train_X_cv,
            train_y_cv,
            valid_X_cv,
            valid_y_cv,
            test_X_values,
        )
        print(f"cv mean mse:{mse}, corr:{corr}")

        final_test_pred = test_preds
    elif cfg["general"]["cv"] == "normal":
        # k-fold cross-validation
        skf = KFold(n_splits=5, shuffle=True, random_state=cfg["general"]["seed"])
        test_pred_list = []
        valid_mse_list = []
        valid_corr_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            _, test_preds, mse, corr = one_fold(
                fold_n,
                skf,
                cfg,
                train_X_values,
                train_y_values,
                test_X_values,
            )
            test_pred_list.append(test_preds)
            valid_mse_list.append(mse)
            valid_corr_list.append(corr)

        valid_mse_mean = np.mean(valid_mse_list, axis=0)
        valid_corr_mean = np.mean(valid_corr_list, axis=0)
        print(f"cv mean mse:{valid_mse_mean}, corr:{valid_corr_mean}")

        final_test_pred = np.mean(test_pred_list, axis=0)
    else:
        # train all data
        cfg["fold_n"] = "all"
        final_test_pred = all_train(cfg, train_X_values, train_y_values, test_X_values)

    if args.save_sub:
        if cfg["task"] == "cite":
            final_test_pred = final_test_pred.flatten()
            submission = pd.DataFrame(data={"target": final_test_pred})
            submission.index.name = "row_id"
        else:
            eval_ids = pd.read_parquet("../data/input/compressed/evaluation.parquet")
            eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
            eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

            submission = pd.Series(
                name="target",
                index=pd.MultiIndex.from_frame(eval_ids),
                dtype=np.float32,
            )

            y_columns = np.load(
                "../data/input/compressed/train_multi_targets_idxcol.npz",
                allow_pickle=True,
            )["columns"]
            test_index = np.load(
                "../data/input/compressed/test_multi_inputs_idxcol.npz",
                allow_pickle=True,
            )["index"]

            cell_dict = dict((k, v) for v, k in enumerate(test_index))
            assert len(cell_dict) == len(test_index)
            gene_dict = dict((k, v) for v, k in enumerate(y_columns))
            assert len(gene_dict) == len(y_columns)

            eval_ids_cell_num = eval_ids.cell_id.apply(lambda x: cell_dict.get(x, -1))
            eval_ids_gene_num = eval_ids.gene_id.apply(lambda x: gene_dict.get(x, -1))
            valid_multi_rows = (eval_ids_gene_num != -1) & (eval_ids_cell_num != -1)

            submission.iloc[valid_multi_rows] = final_test_pred[
                eval_ids_cell_num[valid_multi_rows].to_numpy(),
                eval_ids_gene_num[valid_multi_rows].to_numpy(),
            ]

            submission.reset_index(drop=True, inplace=True)
            submission.index.name = "row_id"

        submission.to_csv(f"../data/output/{cfg['general']['save_name']}.csv")


if __name__ == "__main__":
    main()
