"""Batch demo: predict every tweet in dataset/sample_test.csv and compare with the
manually assigned expected labels (dataset/sample_test_expected.csv).

    python predict_sample.py
"""
import pandas as pd

import config
from utils.predictor import predict_sentiment


def main():
    df = pd.read_csv(config.SAMPLE_EXPECTED_PATH)
    df["predicted"] = df["tweet"].apply(predict_sentiment)
    df["result"] = (df["expected"] == df["predicted"]).map({True: "Correct", False: "Incorrect"})
    df.index += 1
    print(df.to_string(index=True, index_names=False))
    correct = (df["result"] == "Correct").sum()
    print(f"\n{correct} of {len(df)} classified in agreement with the manual label "
          f"({correct / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
