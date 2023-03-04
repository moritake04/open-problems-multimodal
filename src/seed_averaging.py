# python seed_averaging.py ../data/output/cite/*** ../data/output/multi/*** sub_mlp

import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cite", type=str, help="path to cite_submission")
    parser.add_argument("multi", type=str, help="path to multi_submission")
    parser.add_argument("save_name", type=str, help="save name")
    args = parser.parse_args()
    return args


def main():
    # read csv pass
    args = get_args()
    cite_pass = args.cite
    multi_pass = args.multi

    # read submission csv
    cite_submission = pd.read_csv(cite_pass + ".csv")
    cite_submission_0 = pd.read_csv(cite_pass + "_seed0.csv")
    cite_submission_1 = pd.read_csv(cite_pass + "_seed1.csv")
    cite_submission_2 = pd.read_csv(cite_pass + "_seed2.csv")
    cite_submission_3 = pd.read_csv(cite_pass + "_seed3.csv")
    multi_submission = pd.read_csv(multi_pass + ".csv")
    multi_submission_0 = pd.read_csv(multi_pass + "_seed0.csv")
    multi_submission_1 = pd.read_csv(multi_pass + "_seed1.csv")
    multi_submission_2 = pd.read_csv(multi_pass + "_seed2.csv")
    multi_submission_3 = pd.read_csv(multi_pass + "_seed3.csv")

    # averaging
    cite_submission["target"] = (
        cite_submission["target"]
        + cite_submission_0["target"]
        + cite_submission_1["target"]
        + cite_submission_2["target"]
        + cite_submission_3["target"]
    ) / 5.0
    multi_submission["target"] = (
        multi_submission["target"]
        + multi_submission_0["target"]
        + multi_submission_1["target"]
        + multi_submission_2["target"]
        + multi_submission_3["target"]
    ) / 5.0

    # concat and save
    target = pd.concat(
        [cite_submission[:6812820]["target"], multi_submission["target"].dropna()]
    ).reset_index(drop=True)
    target.index.name = "row_id"
    assert not target.isna().any()
    target.to_csv(f"../data/output/{args.save_name}.csv")


if __name__ == "__main__":
    main()
