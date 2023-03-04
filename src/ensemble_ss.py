# python ensemble_ss.py ../data/output/***.csv ../data/output/***.csv sub_ensemble_

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sub1", type=str, help="path to submission1 (.csv)")
    parser.add_argument("sub2", type=str, help="path to submission2 (.csv)")
    parser.add_argument("save_name", type=str, help="save name")
    args = parser.parse_args()
    return args


def std(x):
    return (x - np.mean(x)) / np.std(x)


def gen_std_submission(path, cell_ids):
    """
    Standardize submission per cell_id
    """
    df = pd.read_csv(path)
    df["cell_id"] = cell_ids
    vals = []
    for idx, g in tqdm(
        df.groupby("cell_id", sort=False), desc=f"Standardizing {path}", miniters=1000
    ):
        vals.append(std(g.target).values)
    vals = np.concatenate(vals)
    return vals


def main():
    # read csv pass
    args = get_args()
    sub1_pass = args.sub1
    sub2_pass = args.sub2

    # read submission csv
    cell_ids = pd.read_parquet("../data/input/compressed/evaluation.parquet").cell_id

    # ensemble
    ensemble = (
        gen_std_submission(sub1_pass, cell_ids) * 0.8
        + gen_std_submission(sub2_pass, cell_ids) * 0.2
    )
    print(ensemble[:5])  # print head

    # save
    df_submit = pd.read_parquet("../data/input/compressed/sample_submission.parquet")
    df_submit["target"] = ensemble
    df_submit.to_csv(f"../data/output/{args.save_name}.csv", index=False)


if __name__ == "__main__":
    main()
