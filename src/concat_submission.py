# python concat_submission.py ../data/output/cite/blending/all_1115_final.csv ../data/output/multi/blending/all_mlp_cnn_trimap50_corr10_dropout.csv sub_mlp

import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cite", type=str, help="path to cite_submission (.csv)")
    parser.add_argument("multi", type=str, help="path to multi_submission (.csv)")
    parser.add_argument("save_name", type=str, help="save name")
    args = parser.parse_args()
    return args


def main():
    # read csv pass
    args = get_args()
    cite_pass = args.cite
    multi_pass = args.multi

    # read submission csv
    cite_submission = pd.read_csv(cite_pass)
    multi_submission = pd.read_csv(multi_pass)

    # concat and save
    target = pd.concat(
        [cite_submission[:6812820]["target"], multi_submission[6812820:]["target"]]
    ).reset_index(drop=True)
    target.index.name = "row_id"
    assert not target.isna().any()
    print(target[:5])
    print(target[-5:])
    target.to_csv(f"../data/output/{args.save_name}.csv")


if __name__ == "__main__":
    main()
