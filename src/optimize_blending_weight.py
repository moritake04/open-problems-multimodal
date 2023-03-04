import argparse
import gc

import joblib
import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from pytorch_lightning import seed_everything
from scipy.optimize import minimize
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
    parser.add_argument("task", type=str, help="cite or multi")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    # parser.add_argument("-s", "--pred_save", action="store_true", help="to save or not to save pred")
    # parser.add_argument("-sn", "--save_name", type=str, help="save_name for pred")
    args = parser.parse_args()
    return args


def inference(cfg, input_size, output_size, valid_X):
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
            valid_preds = 0
            for n in range(5):
                model = RidgeRegressorInference(
                    cfg,
                    f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_{n}.ckpt",
                )
                valid_preds += model.predict(valid_X)
                return valid_preds
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

    valid_preds = model.predict(valid_X)
    return valid_preds


def one_fold(fold_n, skf, cfg, train_X, train_y):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_y))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X[valid_indices],
        train_y[valid_indices],
    )

    # inference
    valid_preds = inference(cfg, valid_X_cv.shape[1], valid_y_cv.shape[1], valid_X_cv)
    torch.cuda.empty_cache()

    return valid_preds, valid_indices


def one_fold_gkf(fold_n, gkf, groups, cfg, train_X, train_y):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(gkf.split(train_X, train_y, groups=groups))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X[valid_indices],
        train_y[valid_indices],
    )

    # inference
    valid_preds = inference(cfg, valid_X_cv.shape[1], valid_y_cv.shape[1], valid_X_cv)
    torch.cuda.empty_cache()

    return valid_preds, valid_indices


def get_preds(cfg, fold):
    cfg["general"]["fold"] = [fold]

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
        elif cfg["data_type"] == "svd_important_laurent_nodrop":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}_important_laurent_nodrop.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}_important_laurent_nodrop.npz"
            )
        elif cfg["data_type"] == "svd_important_laurent_nodrop_raw_counts":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_cite_inputs_values_{svd_dim}_important_laurent_nodrop_raw.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_cite_inputs_values_{svd_dim}_important_laurent_nodrop_raw.npz"
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
        if cfg["data_type"] == "svd_important_laurent_nodrop_raw_counts":
            train_X_idxcol = np.load(
                "../data/input/compressed/train_cite_inputs_raw_idxcol.npz",
                allow_pickle=True,
            )
            test_X_idxcol = np.load(
                "../data/input/compressed/test_cite_inputs_raw_idxcol.npz",
                allow_pickle=True,
            )
    else:
        if cfg["data_type"] == "svd":
            train_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}.npz"
            )
            test_X_values = scipy.sparse.load_npz(
                f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}.npz"
            )
        # elif cfg["data_type"] == "svd_raw_counts":
        #    train_X_values = scipy.sparse.load_npz(
        #        f"../data/input/svd/svd_train_multi_inputs_values_{svd_dim}_raw.npz"
        #    )
        #    test_X_values = scipy.sparse.load_npz(
        #        f"../data/input/svd/svd_test_multi_inputs_values_{svd_dim}_raw.npz"
        #    )
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
        """
        if cfg["data_type"] == "svd_raw_counts":
            train_X_idxcol = np.load(
                "../data/input/compressed/train_multi_inputs_raw_idxcol.npz",
                allow_pickle=True,
            )
            test_X_idxcol = np.load(
                "../data/input/compressed/test_multi_inputs_raw_idxcol.npz",
                allow_pickle=True,
            )
        """
    metadata = pd.read_csv("../data/input/metadata.csv", index_col="cell_id")
    metadata["gender"] = 0
    metadata[metadata["donor"] == "13176"]["gender"] = 1
    print("read data")

    # reindex metadata
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
        # one-hot encoding (cell_type)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        train_meta = scipy.sparse.csr_matrix(ohe.fit_transform(train_meta.values))
        # retrieve day
        train_day = train_metadata["day"].values.reshape(-1, 1)
        # 水平結合
        train_X_values = scipy.sparse.hstack(
            [train_X_values, train_day, train_meta], format="csr"
        )
    elif cfg["use_meta"] == "day_only":
        # retrieve day
        train_day = train_metadata["day"].values.reshape(-1, 1)
        # 水平結合
        train_X_values = scipy.sparse.hstack([train_X_values, train_day], format="csr")
    elif cfg["use_meta"] == "type_only":
        train_meta = train_metadata.drop(["donor", "technology", "day"], axis=1)
        # one-hot encoding (cell_type)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        train_meta = scipy.sparse.csr_matrix(ohe.fit_transform(train_meta.values))
        # 水平結合
        train_X_values = scipy.sparse.hstack([train_X_values, train_meta], format="csr")

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
        for_list = [fold]
        for fold_n in tqdm(for_list):
            cfg["fold_n"] = fold_n
            valid_preds, valid_indices = one_fold_gkf(
                fold_n,
                gkf,
                train_metadata["donor"],
                cfg,
                train_X_values,
                train_y_values,
            )
    elif cfg["general"]["cv"] == "normal":
        # k-fold cross-validation
        skf = KFold(n_splits=5, shuffle=True, random_state=cfg["general"]["seed"])
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            valid_preds, valid_indices = one_fold(
                fold_n,
                skf,
                cfg,
                train_X_values,
                train_y_values,
            )

    return valid_preds, valid_indices


def main():
    args = get_args()
    if args.fold is not None:
        fold_list = [args.fold]
    else:
        fold_list = range(3)  # 3 donor k-fold
    print(f"fold: {fold_list}")

    cfg_list = []
    if args.task == "cite":
        # cfg_list.append("../configs/cite/mlp_cite_e50_dayss/mlp_cite_e50_dayss")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss/mlp_cite_1dcnn_dayss")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss_corr_100/mlp_cite_1dcnn_dayss_corr_100")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss_corr_500/mlp_cite_1dcnn_dayss_corr_500")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss_trimap/mlp_cite_1dcnn_dayss_trimap")
        # 没 cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss_umap/mlp_cite_1dcnn_dayss_umap")
        # cfg_list.append("../configs/cite/ridge_cite_dayss/ridge_cite_dayss")
        # cfg_list.append("../configs/cite/mlp_cite_largemlp_dayss/mlp_cite_largemlp_dayss")
        # cfg_list.append("../configs/cite/mlp_cite_largemlp_dayss_mse/mlp_cite_largemlp_dayss_mse")
        # cfg_list.append("../configs/cite/mlp_cite_largemlp_raw/mlp_cite_largemlp_raw")
        # cfg_list.append("../configs/cite/lightgbm_cite_dayss/lightgbm_cite_dayss")
        # cfg_list.append("../configs/cite/lightgbm_cite_corr_100_dayss/lightgbm_cite_corr_100_dayss")
        # 没 cfg_list.append("../configs/cite/lightgbm_cite_ambrosm_dayss/lightgbm_cite_ambrosm_dayss")
        # 没 cfg_list.append("../configs/cite/ridge_cite_corr_100_dayss/ridge_cite_corr_100_dayss")
        # cfg_list.append("../configs/cite/mlp_cite_largemlp_dayss_svdnodrop/mlp_cite_largemlp_dayss_svdnodrop")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss_svdnodrop/mlp_cite_1dcnn_dayss_svdnodrop")
        # 没 cfg_list.append("../configs/cite/mlp_cite_largemlp_dayss_mse_svdnodrop/mlp_cite_largemlp_dayss_mse_svdnodrop")
        # cfg_list.append("../configs/cite/mlp_cite_largemlp_dayss_dropout/mlp_cite_largemlp_dayss_dropout")
        # cfg_list.append("../configs/cite/mlp_cite_1dcnn_raw_dropout/mlp_cite_1dcnn_raw_dropout")

        use_base_pred = True
        base_pred = joblib.load(
            "../data/output/pred/cite/valid/_1030_nosvddrop1dcnn_dropout_xgboost_catboost_dropoutraw_1115.txt"
        )
        cfg_list.append(
            "../configs/cite/mlp_cite_largemlp_raw_dropout/mlp_cite_largemlp_raw_dropout"
        )

        seed_averaging_num_list = [5]
    else:
        # cfg_list.append("../configs/multi/mlp_multi_e50_dayss_umap10/mlp_multi_e50_dayss_umap10")
        # cfg_list.append("../configs/multi/mlp_multi_1dcnn_dayss_umap10/mlp_multi_1dcnn_dayss_umap10")
        # cfg_list.append("../configs/multi/mlp_multi_1dcnn_dayss_trimap50/mlp_multi_1dcnn_dayss_trimap50")
        # cfg_list.append("../configs/multi/mlp_multi_1dcnn_dayss_corr_50/mlp_multi_1dcnn_dayss_corr_50")
        # cfg_list.append("../configs/multi/mlp_multi_largemlp_dayss/mlp_multi_largemlp_dayss")
        # cfg_list.append("../configs/multi/mlp_multi_largemlp_dayss/mlp_multi_1dcnn_raw")
        # cfg_list.append("../configs/multi/mlp_multi_1dcnn_dayss_umap10_dropout/mlp_multi_1dcnn_dayss_umap10_dropout")
        # cfg_list.append("../configs/multi/ridge_multi_dayss/ridge_multi_dayss")

        use_base_pred = True
        base_pred = joblib.load(
            "../data/output/pred/multi/valid/all_mlp_cnn_trimap50_corr10_dropout.txt"
        )
        cfg_list.append(
            "../configs/multi/ridge_multi_corr_10_dayss/ridge_multi_corr_10_dayss"
        )

        seed_averaging_num_list = [0]

    # if args.pred_save:
    #    pred_list_cv = [0] * len(fold_list)

    weights_list = []
    for j, fold_n in enumerate(fold_list):
        pred_list = []

        # 保存した予測値を使用するとき
        if use_base_pred:
            pred_list.append(base_pred[j])

        for i, c in enumerate(cfg_list):
            if seed_averaging_num_list[i] == 0:
                with open(c + ".yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                valid_preds, valid_indices = get_preds(cfg, fold_n)
                pred_list.append(valid_preds)
                del valid_preds
                gc.collect()
            else:
                pred_list_for_averaging = []
                with open(c + ".yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                valid_preds, valid_indices = get_preds(cfg, fold_n)
                pred_list_for_averaging.append(valid_preds)
                del valid_preds
                gc.collect()
                for i in range(seed_averaging_num_list[i] - 1):
                    with open(c + f"_seed{i}.yaml", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    valid_preds, valid_indices = get_preds(cfg, fold_n)
                    pred_list_for_averaging.append(valid_preds)
                    del valid_preds
                    gc.collect()
                pred_list.append(np.mean(pred_list_for_averaging, axis=0))
                del pred_list_for_averaging
                gc.collect()

        valid_y = scipy.sparse.load_npz(
            f"../data/input/compressed/train_{args.task}_targets_values.sparse.npz"
        )[valid_indices]

        corr = correlation_score(valid_y.todense(), np.mean(pred_list, axis=0))
        print(f"averaged_corr:{corr}")
        for i, p in enumerate(pred_list):
            corr = correlation_score(valid_y.todense(), p)
            print(f"corr_[{i}]:{corr}")

        def f(x):
            pred = 0
            for i, p in enumerate(pred_list):
                if i < len(x):
                    pred += p * x[i]
                else:
                    pred += p * (1 - sum(x))
            score = correlation_score(valid_y.todense(), pred) * -1
            return score

        init_state = [round(1 / len(pred_list), 3) for _ in range(len(pred_list) - 1)]
        result = minimize(f, init_state, method="Nelder-Mead")
        print(f"optimized_corr:{-result['fun']}")

        weights = [0] * len(pred_list)
        for i in range(len(pred_list) - 1):
            weights[i] = result["x"][i]
        weights[len(pred_list) - 1] = 1 - sum(result["x"])
        weights_list.append(weights)
        print(f"weights:{weights}")

        # if args.pred_save:
        #    pred_list_cv[j] = pred_list

    avg_weights = np.mean(weights_list, axis=0)
    print(f"averaged_weights:{avg_weights}")

    """
    if args.pred_save:
        if use_base_pred:
            cfg_list_n_plus = 1
        else:
            cfg_list_n_plus = 0
        opt_preds = [0] * len(fold_list)
        for j, fold_n in enumerate(fold_list):
            for i in range(len(cfg_list) + cfg_list_n_plus):
                opt_preds[j] += pred_list_cv[j][i] * avg_weights[i]

        joblib.dump(
            opt_preds,
            f"../data/output/pred/{args.task}/{args.save_name}.txt",
            compress=3,
        )
    """


if __name__ == "__main__":
    main()
