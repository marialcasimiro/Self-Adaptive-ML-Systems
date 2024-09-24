#!/usr/bin/env python

import gc

import numpy as np
import pandas as pd

def normalize_d_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        train_df["D" + str(i)] = train_df[
            "D" + str(i)
        ] - train_df.TransactionDT / np.float32(24 * 60 * 60)
        test_df["D" + str(i)] = test_df[
            "D" + str(i)
        ] - test_df.TransactionDT / np.float32(24 * 60 * 60)


def label_encode_and_mem_reduce(train_df: pd.DataFrame, test_df: pd.DataFrame):
    for _, f in enumerate(train_df.columns):
        # FACTORIZE CATEGORICAL VARIABLES
        if (np.str(train_df[f].dtype) == "category") | (train_df[f].dtype == "object"):
            df_comb = pd.concat([train_df[f], test_df[f]], axis=0)
            df_comb, _ = df_comb.factorize(sort=False)
            if df_comb.max() > 32000:
                print(f, "needs int32")
            train_df[f] = df_comb[: len(train_df)].astype("int16")
            test_df[f] = df_comb[len(train_df) :].astype("int16")
        # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
        elif f not in ["TransactionAmt", "TransactionDT"]:
            mn = np.min((train_df[f].min(), test_df[f].min()))
            train_df[f] -= np.float32(mn)
            test_df[f] -= np.float32(mn)
            train_df[f].fillna(-1, inplace=True)
            test_df[f].fillna(-1, inplace=True)


def feature_selection(cols: list):
    """perform feature selection"""
    cols.remove("TransactionDT")
    for c in ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]:
        cols.remove(c)

    # FAILED TIME CONSISTENCY TEST
    for c in ["C3", "M5", "id_08", "id_33"]:
        cols.remove(c)
    for c in ["card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"]:
        cols.remove(c)
    for c in ["id_" + str(x) for x in range(22, 28)]:
        cols.remove(c)

    return cols


# Encoding Functions
# Below are 5 encoding functions:

# 1. `encode_FE` does frequency encoding where it combines train and test first and then encodes.
# 2. `encode_LE` is a label encoded for categorical features
# 3. `encode_AG` makes aggregated features such as aggregated mean and std
# 4. `encode_CB` combines two columns
# 5. `encode_AG2` makes aggregated features where it counts how many unique values of one feature is within a group.

# For more explanation about feature engineering, see the discussion [here][1]

# [1]: https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575#latest-641841


# 1) FREQUENCY ENCODE TOGETHER
def encode_fe(df, cols):
    """does frequency encoding where it combines train and test first and then encodes"""
    for col in cols:
        vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col + "_FE"
        df[nm] = df[col].map(vc)
        df[nm] = df[nm].astype("float32")
        print(nm, ", ", end="")


# 2) LABEL ENCODE
def encode_le(col, df, verbose=True):
    """`encode_LE` is a label encoded for categorical features"""
    df_comb, _ = df[col].factorize(sort=True)
    nm = col
    if df_comb.max() > 32000:
        df[nm] = df_comb.astype("int32")
    else:
        df[nm] = df_comb.astype("int16")

    del df_comb
    gc.collect()

    if verbose:
        print(nm, ", ", end="")


# 3) GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_ag(main_columns, uids, df, aggregations=None, fillna=True, usena=False):
    """AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS"""
    if aggregations is None:
        aggregations = ["mean"]
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + "_" + col + "_" + agg_type
                temp_df = pd.concat([df[[col, main_column]]])
                if usena:
                    # all instances of "-1" will be replaced by np.nan in column "main_column"
                    temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan

                temp_df = (
                    temp_df.groupby([col])[main_column]
                    .agg([agg_type])
                    .reset_index()
                    .rename(columns={agg_type: new_col_name})
                )

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                df[new_col_name] = df[col].map(temp_df).astype("float32")

                if fillna:
                    df[new_col_name].fillna(-1, inplace=True)

                print("'" + new_col_name + "'", ", ", end="")


# 4) COMBINE FEATURES
def encode_cb(col1, col2, df):
    """combines two columns"""
    nm = col1 + "_" + col2
    df[nm] = df[col1].astype(str) + "_" + df[col2].astype(str)
    encode_le(nm, df, verbose=False)
    print(nm, ", ", end="")


# 5) GROUP AGGREGATION NUNIQUE
def encode_ag2(main_columns, uids, df):
    """makes aggregated features where it counts how many unique values of one feature is within a group"""
    for main_column in main_columns:
        for col in uids:
            comb = df[[col] + [main_column]]
            mp = comb.groupby(col)[main_column].agg(["nunique"])["nunique"].to_dict()
            df[col + "_" + main_column + "_ct"] = df[col].map(mp).astype("float32")
            print(col + "_" + main_column + "_ct, ", end="")


def do_feature_engineering(df: pd.DataFrame):
    # TRANSACTION AMT CENTS
    df["cents"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype(
        "float32"
    )
    print("cents, ", end="")
    # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
    encode_fe(df, ["addr1", "card1", "card2", "card3", "P_emaildomain"])
    # COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
    encode_cb("card1", "addr1", df)
    encode_cb("card1_addr1", "P_emaildomain", df)
    # FREQUENCY ENOCDE
    encode_fe(df, ["card1_addr1", "card1_addr1_P_emaildomain"])
    # GROUP AGGREGATE
    encode_ag(
        ["TransactionAmt", "D9", "D11"],
        ["card1", "card1_addr1", "card1_addr1_P_emaildomain"],
        df,
        ["mean", "std"],
        usena=True,
    )
