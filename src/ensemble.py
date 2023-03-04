# python ensemble.py ../data/output/***.csv ../data/output/***.csv sub_ensemble_

import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sub1", type=str, help="path to submission1 (.csv)")
    parser.add_argument("sub2", type=str, help="path to submission2 (.csv)")
    parser.add_argument("save_name", type=str, help="save name")
    args = parser.parse_args()
    return args


def main():
    # read csv pass
    args = get_args()
    sub1_pass = args.sub1
    sub2_pass = args.sub2

    # read submission csv
    sub1 = pd.read_csv(sub1_pass)
    print("read1")
    sub2 = pd.read_csv(sub2_pass)
    print("read2")

    # ensemble and save
    sub1["target"] = sub1["target"] * 0.5 + sub2["target"] * 0.5
    # sub1["target"] = sub1["target"] + sub2["target"]
    print(sub1[:5])
    print(sub1[-5:])
    sub1.to_csv(f"../data/output/{args.save_name}.csv", index=False)


if __name__ == "__main__":
    main()
