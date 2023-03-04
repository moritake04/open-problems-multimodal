import argparse
import gc

import joblib
import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

import wandb
from regressor import (CatBoostRegressorInference, LightGBMRegressorInference,
                       MLPLargeRegressorInference, MLPRegressorInference,
                       OneDCNNRegressorInference, RidgeRegressorInference,
                       XGBoostRegressorInference)


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
    parser.add_argument("save_name", type=str, help="save name")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    args = parser.parse_args()
    return args


def wandb_start(save_name, fold_n, disabled="online"):
    wandb.init(
        project="open-problems-multimodal",
        name=f"{save_name}_{fold_n}",
        group=f"{save_name}_cv",
        job_type="blending",
        mode=disabled,
    )


def inference(cfg, input_size, output_size, test_X, valid_X):
    if cfg["model"] == "mlp":
        model = MLPRegressorInference(
            cfg,
            input_size,
            output_size,
            f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
        )

    if cfg["model"] == "largemlp":
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
                valid_preds += model.predict(valid_X)
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
    valid_preds = model.predict(valid_X)

    return test_preds, valid_preds


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
    torch.cuda.empty_cache()

    return test_preds, valid_preds, valid_indices


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
    torch.cuda.empty_cache()

    return test_preds, valid_preds, valid_indices


def all_train(cfg, train_X, train_y, test_X):
    print("[all_train] start")

    seed_everything(cfg["general"]["seed"], workers=True)

    # inference
    test_preds = inference(cfg, train_X.shape[1], train_y.shape[1], test_X)
    torch.cuda.empty_cache()

    return test_preds


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
            train_X_values_day2, test_X_values_day2 = (
                ss.transform(train_X_values_day2),
                ss.transform(test_X_values_day2),
            )
            # day3
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day3, test_X_values_day3], format="csr"
                )
            )
            train_X_values_day3, test_X_values_day3 = (
                ss.transform(train_X_values_day3),
                ss.transform(test_X_values_day3),
            )
            # day4
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day4, test_X_values_day4], format="csr"
                )
            )
            train_X_values_day4, test_X_values_day4 = (
                ss.transform(train_X_values_day4),
                ss.transform(test_X_values_day4),
            )
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
            train_X_values_day2, test_X_values_day2 = (
                ss.transform(train_X_values_day2),
                ss.transform(test_X_values_day2),
            )
            # day3
            ss = StandardScaler(with_mean=False)
            ss.fit(
                scipy.sparse.vstack(
                    [train_X_values_day3, test_X_values_day3], format="csr"
                )
            )
            train_X_values_day3, test_X_values_day3 = (
                ss.transform(train_X_values_day3),
                ss.transform(test_X_values_day3),
            )
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
            train_X_values_day7, test_X_values_day7 = (
                ss.transform(train_X_values_day7),
                ss.transform(test_X_values_day7),
            )
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
            test_preds, valid_preds, valid_indices = one_fold_gkf(
                fold_n,
                gkf,
                train_metadata["donor"],
                cfg,
                train_X_values,
                train_y_values,
                test_X_values,
            )
    elif cfg["general"]["cv"] == "normal":
        # k-fold cross-validation
        skf = KFold(n_splits=5, shuffle=True, random_state=cfg["general"]["seed"])
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            test_preds, valid_preds, valid_indices = one_fold(
                fold_n,
                skf,
                cfg,
                train_X_values,
                train_y_values,
                test_X_values,
            )
    else:
        # train all data
        cfg["fold_n"] = "all"
        test_pred = all_train(cfg, train_X_values, train_y_values, test_X_values)
        valid_preds, valid_indices = None, None

    return test_preds, valid_preds, valid_indices


def main():
    args = get_args()
    fold_list = range(3)  # 3 donor k-fold
    if args.fold is not None:
        fold_list = [args.fold]

    print(f"fold: {fold_list}")

    cfg_list = []
    if args.task == "cite":
        # old -> weights = [0.47708333, 0.52291667]

        # new -> weights = [0.3561849, 0.6438151], corr = [0.8917, 0.8971, 0.8930]

        # mlp_cnn_1028 -> weights = [0.59277344, 0.40722656], cv = [0.8919, 0.8974, 0.8932], cv_mean = 0.8941

        # mlp_cnn_corr100_1028 -> weights = [0.84521484, 0.15478516], cv = [0.8921, 0.8976, 0.8932] cv_mean = 0.8943

        # mlp_cnn_corr100_corr500_1028 -> [0.85120443, 0.14879557], cv = [0.8923, 0.8978, 0.8931], cv_mean = 0.8944

        # mlp_cnn_corr100_corr500_ambrosm_1028 -> [0.88717448, 0.11282552], cv = [0.8923, 0.8977, 0.8932], cv_mean = 0.8944↑

        # mlp_cnn_corr100_corr500_ambrosm_trimap_1028 -> [0.96318359, 0.03681641], cv = [0.8923, 0.8977, 0.8932], cv_mean = 0.8944↑

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_1028 -> [0.83339844, 0.16660156], cv = [0.8925, 0.8978, 0.8935], cv_mean = 0.8946

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_1028 -> [0.9101237, 0.0898763], cv = [0.8925, 0.8978, 0.8936], cv_mean = 0.8946↑

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_1029 -> [0.90266927, 0.09733073], cv = [0.8925, 0.8979, 0.8936], cv_mean = 0.8947

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_1029 -> [0.75696615, 0.24303385], cv = [0.8925, 0.8980, 0.8936], cv_mean = 0.8947↑

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_largemlpraw_1029 -> [0.95735677, 0.04264323], cv = [0.8926, 0.8980, 0.8937], cv_mean = 0.8948

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_largemlpraw_lgbm_1029 -> [0.84801432, 0.15198568], cv = [0.8927, 0.8983, 0.8937], cv_mean = 0.8949

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_largemlpraw_lgbm_ridgeambros_1030 -> [1.07851563 -0.07851563], cv = [0.8928, 0.8984, 0.8937], cv_mean = 0.8949↑

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_largemlpraw_lgbm_ridgeambros_lgbmcorr100_1030 -> [0.95250651, 0.04749349], cv = [0.8927, 0.8985, 0.8937], cv_mean = 0.8950

        # mlp_cnn_corr100_corr500_ambrosm_trimap_raw_ridge_largemlp_mse_largemlpraw_lgbm_ridgeambros_lgbmcorr100_1030_nosvddrop_1112 -> [0.95253906 0.04746094], cv = [0.8928, 0.8985, 0.8937], cv_meam = 0.8950

        # _1030_nosvddrop1dcnn_1112 -> [0.92815755 0.07184245], cv = [0.8928, 0.8984, 0.8937], cv_mean = 0.8950

        # _1030_nosvddrop1dcnn_dropout_1112 -> [1.09960938 -0.09960938], cv = [0.8928, 0.8985, 0.8938], cv_mean = 0.8950

        # _1030_nosvddrop1dcnn_dropout_xgboost_1112 -> [0.96738281 0.03261719], cv = [0.8928, 0.8985, 0.8938], cv_mean = 0.8950

        # _1030_nosvddrop1dcnn_dropout_xgboost_catboost_1112 -> [1.00986328 -0.00986328], cv = [0.8928, 0.8985, 0.8938], cv_mean = 0.8950

        # _1030_nosvddrop1dcnn_dropout_xgboost_catboost_dropoutraw_1115 -> [1.02750651, -0.02750651], cv = [0.8928, 0.8985, 0.8938], cv_mean = 0.8950

        # _1030_nosvddrop1dcnn_dropout_xgboost_catboost_dropoutraw2_1115 -> [0.97942708 0.02057292], cv = [0.8928, 0.8985, 0.8938], cv_mean = 0.8950

        base_preds_valid = joblib.load(
            "../data/output/pred/cite/valid/_1030_nosvddrop1dcnn_dropout_xgboost_catboost_dropoutraw_1115.txt"
        )
        base_preds_test = joblib.load(
            "../data/output/pred/cite/test/_1030_nosvddrop1dcnn_dropout_xgboost_catboost_dropoutraw_1115.txt"
        )
        use_base_pred = True
        weights = [0.97942708, 0.02057292]
        cfg_list.append(
            "../configs/cite/mlp_cite_largemlp_raw_dropout/mlp_cite_largemlp_raw_dropout"
        )
        seed_averaging_num_list = [5]

        """
        # base1
        cfg_list.append("../configs/cite/mlp_cite_e50_dayss/mlp_cite_e50_dayss")
        cfg_list.append("../configs/cite/mlp_cite_1dcnn_dayss/mlp_cite_1dcnn_dayss")

        # base2
        cfg_list.append(
            "../configs/cite/mlp_cite_1dcnn_dayss_corr_100/mlp_cite_1dcnn_dayss_corr_100"
        )

        # base3
        cfg_list.append(
            "../configs/cite/mlp_cite_1dcnn_dayss_corr_500/mlp_cite_1dcnn_dayss_corr_500"
        )

        cfg_list.append(
            "../configs/cite/mlp_cite_1dcnn_dayss_ambrosm/mlp_cite_1dcnn_dayss_ambrosm"
        )

        weights = [
            0.59277344,  # mlp
            0.40722656,  # 1dcnn
            0.84521484,  # mlp+1dcnn
            0.15478516,  # corr_100
            0.85120443,  # mlp+1dcnn+corr_100
            0.14879557,  # corr_500
            0.88717448,
            0.11282552,
        ]
        seed_averaging_num_list = [3, 5, 5, 5, 5]
        """

    else:
        # old -> weights = [0.10071615, 0.89928385]

        # new -> weights = [0.13372396, 0.86627604], corr = [0.6650078366776155, 0.6714081248805374, 0.6684539355783071]

        # mlp_cnn_1028 -> weights = [0.25283203 0.74716797], cv = [0.6654, 0.6718, 0.6689], cv_mean = 0.6687

        # mlp_cnn_trimap50_1028 -> weights = [0.65585938 0.34414062], cv = [0.6655, 0.6718, 0.6690], cv_mean = 0.6688

        # mlp_cnn_trimap50_corr10_1029 -> weights = [0.92903646, 0.07096354], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688↑

        # mlp_cnn_trimap50_corr10_raw_1101 -> weights = [0.7485026, 0.2514974], cv = [0.6656, 0.6720, 0.6692], cv_mean = 0.6689

        # mlp_cnn_trimap50_corr10_raw_dropout_1112 -> [0.92845052 0.07154948], cv = [0.6656, 0.6720, 0.6692], cv_mean = 0.6689

        # mlp_cnn_trimap50_corr10_raw_dropout_ridge_1114 -> [0.95592448 0.04407552], cv = [0.6656, 0.6720, 0.6692], cv_mean = 0.6689

        # for all
        # all_mlp_cnn_trimap50_corr10_dropout -> [0.85904948, 0.14095052], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688
        # [0.87467448 0.12532552]

        # mlp_cnn_trimap50_corr10_corr50_1030 -> weights = [0.91852214, 0.08147786], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688↑
        # mlp_cnn_trimap50_corr10_corr50_largemlp_1030 -> weights = [0.96595052, 0.03404948], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688↑ 超微増
        # mlp_cnn_trimap50_corr10_corr50_largemlp_mse_1030 -> weights = [0.6367513, 0.3632487], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688↑
        # mlp_cnn_trimap50_corr10_corr50_largemlp_mse_ridge_1031 -> weights = [0.81516927 0.18483073], cv = [0.6655, 0.6719, 0.6690], cv_mean = 0.6688↑
        # mlp_cnn_trimap50_corr10_corr50_largemlp_mse_ridge_ridgecorr10_1031 -> [0.95589193, 0.04410807], cv = [0.6655, 0.6719, 0.6691], cv_mean = 0.6688↑

        base_preds_valid = joblib.load(
            "../data/output/pred/multi/valid/mlp_cnn_trimap50_corr10_1029.txt"
        )
        base_preds_test = joblib.load(
            "../data/output/pred/multi/test/mlp_cnn_trimap50_corr10_1029.txt"
        )
        use_base_pred = True
        weights = [0.85904948, 0.14095052]
        cfg_list.append(
            "../configs/multi/ridge_multi_corr_10_dayss/ridge_multi_corr_10_dayss"
        )
        """
        raw_valid, raw_test = [0] * 3, [0] * 3
        raw_valid[0] = joblib.load(
            "../data/output/pred/multi/valid/raw_seedaveraging_0.txt"
        )
        raw_valid[1] = joblib.load(
            "../data/output/pred/multi/valid/raw_seedaveraging_1.txt"
        )
        raw_valid[2] = joblib.load(
            "../data/output/pred/multi/valid/raw_seedaveraging_2.txt"
        )
        raw_test[0] = joblib.load(
            "../data/output/pred/multi/test/raw_seedaveraging_0.txt"
        )
        raw_test[1] = joblib.load(
            "../data/output/pred/multi/test/raw_seedaveraging_1.txt"
        )
        raw_test[2] = joblib.load(
            "../data/output/pred/multi/test/raw_seedaveraging_2.txt"
        )
        """
        seed_averaging_num_list = [0]

    preds_list_valid_cv = [0] * len(fold_list)
    preds_list_test_cv = [0] * len(fold_list)
    corr = [0] * len(fold_list)
    for j, fold_n in enumerate(fold_list):
        preds_list_valid = []
        preds_list_test = []

        # 保存した予測値を使用するとき
        if use_base_pred:
            preds_list_valid.append(base_preds_valid[j])
            preds_list_test.append(base_preds_test[j])
            # preds_list_valid.append(raw_valid[j])
            # preds_list_test.append(raw_test[j])

        for i, c in enumerate(cfg_list):
            if seed_averaging_num_list[i] == 0:
                with open(c + ".yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                test_preds, valid_preds, valid_indices = get_preds(cfg, fold_n)
                preds_list_valid.append(valid_preds)
                preds_list_test.append(test_preds)
                del valid_preds, test_preds
                gc.collect()
            else:
                preds_list_for_averaging_valid = []
                preds_list_for_averaging_test = []
                with open(c + ".yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                test_preds, valid_preds, valid_indices = get_preds(cfg, fold_n)
                preds_list_for_averaging_valid.append(valid_preds)
                preds_list_for_averaging_test.append(test_preds)
                del valid_preds, test_preds
                gc.collect()
                for i in range(seed_averaging_num_list[i] - 1):
                    with open(c + f"_seed{i}.yaml", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    test_preds, valid_preds, valid_indices = get_preds(cfg, fold_n)
                    preds_list_for_averaging_valid.append(valid_preds)
                    preds_list_for_averaging_test.append(test_preds)
                    del valid_preds, test_preds
                    gc.collect()
                preds_list_valid.append(np.mean(preds_list_for_averaging_valid, axis=0))
                del preds_list_for_averaging_valid
                gc.collect()
                preds_list_test.append(np.mean(preds_list_for_averaging_test, axis=0))
                del preds_list_for_averaging_test
                gc.collect()

        valid_y = scipy.sparse.load_npz(
            f"../data/input/compressed/train_{args.task}_targets_values.sparse.npz"
        )[valid_indices]

        # del preds_list_valid[2], preds_list_test[2]

        """
        # blending_base0
        for i in range(2):
            preds_list_valid_cv[j] += preds_list_valid[i] * weights[i]
            preds_list_test_cv[j] += preds_list_test[i] * weights[i]

        # blending_base1
        preds_list_valid_cv[j] = (
            preds_list_valid_cv[j] * weights[2] + preds_list_valid[2] * weights[3]
        )
        preds_list_test_cv[j] = (
            preds_list_test_cv[j] * weights[2] + preds_list_test[2] * weights[3]
        )

        # blending_base2
        preds_list_valid_cv[j] = (
            preds_list_valid_cv[j] * weights[4] + preds_list_valid[3] * weights[5]
        )
        preds_list_test_cv[j] = (
            preds_list_test_cv[j] * weights[4] + preds_list_test[3] * weights[5]
        )

        # blending
        preds_list_valid_cv[j] = (
            preds_list_valid_cv[j] * weights[6] + preds_list_valid[4] * weights[7]
        )
        preds_list_test_cv[j] = (
            preds_list_test_cv[j] * weights[6] + preds_list_test[4] * weights[7]
        )
        """

        # blending
        for i in range(len(preds_list_valid)):
            preds_list_valid_cv[j] += preds_list_valid[i] * weights[i]
            preds_list_test_cv[j] += preds_list_test[i] * weights[i]

        wandb_start(args.save_name, fold_n)
        # print score
        corr[j] = correlation_score(valid_y.todense(), preds_list_valid_cv[j])
        print(f"fold_{j}:{corr[j]}")
        wandb.log({"corrscore": corr[j]})
        wandb.finish()

    print(f"mean_cv:{np.mean(corr, axis=0)}")

    # save
    joblib.dump(
        preds_list_valid_cv,
        f"../data/output/pred/{args.task}/valid/{args.save_name}.txt",
        compress=1,
    )
    joblib.dump(
        preds_list_test_cv,
        f"../data/output/pred/{args.task}/test/{args.save_name}.txt",
        compress=1,
    )

    # cv averaging
    final_test_preds = np.mean(preds_list_test_cv, axis=0)
    del preds_list_valid_cv, preds_list_test_cv, preds_list_valid, preds_list_test
    gc.collect()

    if args.task == "cite":
        final_test_preds = final_test_preds.flatten()
        submission = pd.DataFrame(data={"target": final_test_preds})
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

        submission.iloc[valid_multi_rows] = final_test_preds[
            eval_ids_cell_num[valid_multi_rows].to_numpy(),
            eval_ids_gene_num[valid_multi_rows].to_numpy(),
        ]

        submission.reset_index(drop=True, inplace=True)
        submission.index.name = "row_id"

    submission.to_csv(f"../data/output/{args.task}/blending/{args.save_name}.csv")
    print(submission[:5])
    print(submission[-5:])

    """
    pred_list = []
    for i, c in enumerate(cfg_list):
        if seed_averaging_num_list[i] == 0:
            with open(c + ".yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            test_preds = get_preds(cfg)
            pred_list.append(test_preds)
            del test_preds
            gc.collect()
        else:
            pred_list_for_averaging = []
            with open(c + ".yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            test_preds = get_preds(cfg)
            pred_list_for_averaging.append(test_preds)
            del test_preds
            gc.collect()
            for i in range(seed_averaging_num_list[i] - 1):
                with open(c + f"_seed{i}.yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                test_preds = get_preds(cfg)
                pred_list_for_averaging.append(test_preds)
                del test_preds
                gc.collect()
            pred_list.append(np.mean(pred_list_for_averaging, axis=0))
            del pred_list_for_averaging
            gc.collect()

    # blending
    final_test_preds = 0
    for i, p in enumerate(pred_list):
        final_test_preds += p * weights[i]

    # save
    if args.task == "cite":
        final_test_preds = final_test_preds.flatten()
        submission = pd.DataFrame(data={"target": final_test_preds})
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

        submission.iloc[valid_multi_rows] = final_test_preds[
            eval_ids_cell_num[valid_multi_rows].to_numpy(),
            eval_ids_gene_num[valid_multi_rows].to_numpy(),
        ]

        submission.reset_index(drop=True, inplace=True)
        submission.index.name = "row_id"

    submission.to_csv(f"../data/output/{args.task}/blending/{args.save_name}.csv")
    print(submission[:5])
    print(submission[5:])
    """


if __name__ == "__main__":
    main()
