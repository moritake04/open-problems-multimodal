import argparse
import gc

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from pytorch_lightning import seed_everything
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    parser.add_argument("save_name", type=str, help="save name")
    args = parser.parse_args()
    return args


def inference(cfg, input_size, output_size, test_X):
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
            for n in range(5):
                model = RidgeRegressorInference(
                    cfg,
                    f"../weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_{n}.ckpt",
                )
                test_preds += model.predict(test_X)
                return test_preds
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

    return test_preds


def all_train(cfg, train_X, train_y, test_X):
    print("[all_train] start")

    seed_everything(cfg["general"]["seed"], workers=True)

    # inference
    test_preds = inference(cfg, train_X.shape[1], train_y.shape[1], test_X)
    torch.cuda.empty_cache()

    return test_preds


def get_preds(cfg):
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

    # for train all data
    cfg["fold_n"] = "all"
    test_preds = all_train(cfg, train_X_values, train_y_values, test_X_values)

    return test_preds


def main():
    args = get_args()

    if args.task == "cite":

        cfg_list = [
            "../configs/cite/all/mlp_cite_e50_dayss/mlp_cite_e50_dayss",
            "../configs/cite/all/mlp_cite_1dcnn_dayss/mlp_cite_1dcnn_dayss",
            "../configs/cite/all/mlp_cite_1dcnn_dayss_corr_100/mlp_cite_1dcnn_dayss_corr_100",
            "../configs/cite/all/mlp_cite_1dcnn_dayss_corr_500/mlp_cite_1dcnn_dayss_corr_500",
            "../configs/cite/all/mlp_cite_1dcnn_dayss_ambrosm/mlp_cite_1dcnn_dayss_ambrosm",
            "../configs/cite/all/mlp_cite_1dcnn_dayss_trimap/mlp_cite_1dcnn_dayss_trimap",
            "../configs/cite/all/mlp_cite_1dcnn_raw/mlp_cite_1dcnn_raw",
            "../configs/cite/all/ridge_cite_dayss/ridge_cite_dayss",
            "../configs/cite/all/mlp_cite_largemlp_dayss/mlp_cite_largemlp_dayss",
            "../configs/cite/all/mlp_cite_largemlp_dayss_mse/mlp_cite_largemlp_dayss_mse",
            "../configs/cite/all/mlp_cite_largemlp_raw/mlp_cite_largemlp_raw",
            "../configs/cite/all/lightgbm_cite_dayss/lightgbm_cite_dayss",
            "../configs/cite/all/ridge_cite_ambrosm_dayss/ridge_cite_ambrosm_dayss",
            "../configs/cite/all/lightgbm_cite_corr_100_dayss/lightgbm_cite_corr_100_dayss",
            "../configs/cite/all/mlp_cite_largemlp_dayss_svdnodrop/mlp_cite_largemlp_dayss_svdnodrop",
            "../configs/cite/all/mlp_cite_1dcnn_dayss_svdnodrop/mlp_cite_1dcnn_dayss_svdnodrop",
            "../configs/cite/all/mlp_cite_largemlp_dayss_dropout/mlp_cite_largemlp_dayss_dropout",
            "../configs/cite/all/xgboost_cite_dayss/xgboost_cite_dayss",
            "../configs/cite/all/catboost_cite_dayss/catboost_cite_dayss",
            "../configs/cite/all/mlp_cite_1dcnn_raw_dropout/mlp_cite_1dcnn_raw_dropout",
            "../configs/cite/all/mlp_cite_largemlp_raw_dropout/mlp_cite_largemlp_raw_dropout",
        ]
        weights = [
            [0.59277344, 0.40722656],
            [0.84521484, 0.15478516],
            [0.85120443, 0.14879557],
            [0.88717448, 0.11282552],
            [0.96318359, 0.03681641],
            [0.83339844, 0.16660156],
            [0.9101237, 0.0898763],
            [0.90266927, 0.09733073],
            [0.75696615, 0.24303385],
            [0.95735677, 0.04264323],
            [0.84801432, 0.15198568],
            [1.07851563, -0.07851563],
            [0.95250651, 0.04749349],
            [0.95253906, 0.04746094],
            [0.92815755, 0.07184245],
            [1.09960938, -0.09960938],
            [0.96738281, 0.03261719],
            [1.00986328, -0.00986328],
            [1.02750651, -0.02750651],
            [0.97942708, 0.02057292],
        ]
        seed_averaging_num_list = [
            9,
            15,
            15,
            15,
            15,
            15,
            15,
            0,
            15,
            15,
            15,
            0,
            0,
            0,
            15,
            15,
            15,
            0,
            0,
            15,
            15,
        ]
    else:

        cfg_list = [
            "../configs/multi/all/mlp_multi_e50_dayss_umap10/mlp_multi_e50_dayss_umap10",
            "../configs/multi/all/mlp_multi_1dcnn_dayss_umap10/mlp_multi_1dcnn_dayss_umap10",
            "../configs/multi/all/mlp_multi_1dcnn_dayss_trimap50/mlp_multi_1dcnn_dayss_trimap50",
            "../configs/multi/all/mlp_multi_1dcnn_dayss_corr_10/mlp_multi_1dcnn_dayss_corr_10",
            "../configs/multi/all/mlp_multi_1dcnn_dayss_umap10_dropout/mlp_multi_1dcnn_dayss_umap10_dropout",
            # "../configs/multi/ridge_multi_corr_10_dayss/ridge_multi_corr_10_dayss",
        ]
        weights = [
            [0.25283203, 0.74716797],
            [0.65585938, 0.34414062],
            [0.92903646, 0.07096354],
            [0.85904948, 0.14095052],
            # [0.87467448, 0.12532552],
        ]
        seed_averaging_num_list = [9, 15, 15, 15, 15]

    final_test_preds = 0
    preds_list_test = []

    for i, c in enumerate(cfg_list):
        if seed_averaging_num_list[i] == 0:
            with open(c + ".yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            test_preds = get_preds(cfg)
            preds_list_test.append(test_preds)
            del test_preds
            gc.collect()
        else:
            with open(c + ".yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            test_preds = get_preds(cfg)
            for j in range(seed_averaging_num_list[i] - 1):
                with open(c + f"_seed{j}.yaml", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                test_preds += get_preds(cfg)
            preds_list_test.append(test_preds / seed_averaging_num_list[i])
            del test_preds
            gc.collect()

    # blending
    final_test_preds = (
        preds_list_test[0] * weights[0][0] + preds_list_test[1] * weights[0][1]
    )
    for i in range(2, len(cfg_list)):
        final_test_preds = (
            final_test_preds * weights[i - 1][0]
            + preds_list_test[i] * weights[i - 1][1]
        )

    del preds_list_test
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


if __name__ == "__main__":
    main()
