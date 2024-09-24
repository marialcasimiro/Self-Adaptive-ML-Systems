#!/usr/bin/env python
import datetime
import gc
import math
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_fraud_count_per_time_interval(
    time_interval: int, dataset: pd.DataFrame, time_column: str, class_column: str
):
    """plot fraud count per specified interval
    expects
    - time_interval during which to count the instances of fraud (in seconds)
    - the data
    """
    num_intervals = math.ceil(dataset[time_column].max() / time_interval)
    print(f"Number of intervals: {num_intervals}")

    fraud_count = np.empty(num_intervals)

    for interval in range(num_intervals):
        data = dataset.loc[
            (dataset[time_column] >= (interval - 1) * time_interval)
            & (dataset[time_column] < interval * time_interval)
        ]
        fraud_count[interval] = data[class_column].sum()

    plt.figure(0)
    plt.plot(range(num_intervals), fraud_count)
    plt.xlabel("Time interval")
    plt.ylabel("Fraud Count")


def plot_transaction_count_per_time_interval(
    time_interval: int, dataset: pd.DataFrame, time_column: str, class_column: str
):
    """plot transaction count per specified interval
    expects:
    - time_interval during which to count the instances of fraud (in seconds)
    - the data
    """

    num_intervals = math.ceil(dataset[time_column].max() / time_interval)
    print(f"Number of intervals: {num_intervals}")

    transaction_count = np.empty(num_intervals)

    for interval in range(num_intervals):
        data = dataset.loc[
            (dataset[time_column] >= (interval - 1) * time_interval)
            & (dataset[time_column] < interval * time_interval)
        ]
        transaction_count[interval] = data[class_column].count()

    plt.figure(1)
    plt.plot(range(num_intervals), transaction_count)
    plt.xlabel("Time interval")
    plt.ylabel("Transaction Count")


def get_ieee_cis_columns():
    # COLUMNS WITH STRINGS
    str_type = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "id_12",
        "id_15",
        "id_16",
        "id_23",
        "id_27",
        "id_28",
        "id_29",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
        "id_35",
        "id_36",
        "id_37",
        "id_38",
        "DeviceType",
        "DeviceInfo",
    ]
    str_type += [
        "id-12",
        "id-15",
        "id-16",
        "id-23",
        "id-27",
        "id-28",
        "id-29",
        "id-30",
        "id-31",
        "id-33",
        "id-34",
        "id-35",
        "id-36",
        "id-37",
        "id-38",
    ]

    # FIRST 53 COLUMNS
    cols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
        "D11",
        "D12",
        "D13",
        "D14",
        "D15",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
    ]

    # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
    # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
    v = [1, 3, 4, 6, 8, 11]
    v += [13, 14, 17, 20, 23, 26, 27, 30]
    v += [36, 37, 40, 41, 44, 47, 48]
    v += [54, 56, 59, 62, 65, 67, 68, 70]
    v += [76, 78, 80, 82, 86, 88, 89, 91]

    # v += [96, 98, 99, 104] #relates to groups, no NAN
    v += [107, 108, 111, 115, 117, 120, 121, 123]  # maybe group, no NAN
    v += [124, 127, 129, 130, 136]  # relates to groups, no NAN

    # LOTS OF NAN BELOW
    v += [138, 139, 142, 147, 156, 162]  # b1
    v += [165, 160, 166]  # b1
    v += [178, 176, 173, 182]  # b2
    v += [187, 203, 205, 207, 215]  # b2
    v += [169, 171, 175, 180, 185, 188, 198, 210, 209]  # b2
    v += [218, 223, 224, 226, 228, 229, 235]  # b3
    v += [240, 258, 257, 253, 252, 260, 261]  # b3
    v += [264, 266, 267, 274, 277]  # b3
    v += [220, 221, 234, 238, 250, 271]  # b3

    v += [294, 284, 285, 286, 291, 297]  # relates to grous, no NAN
    v += [303, 305, 307, 309, 310, 320]  # relates to groups, no NAN
    v += [281, 283, 289, 296, 301, 314]  # relates to groups, no NAN
    # v += [332, 325, 335, 338] # b4 lots NAN

    cols += ["V" + str(x) for x in v]
    dtypes = {}
    for c in (
        cols
        + ["id_0" + str(x) for x in range(1, 10)]
        + ["id_" + str(x) for x in range(10, 34)]
        + ["id-0" + str(x) for x in range(1, 10)]
        + ["id-" + str(x) for x in range(10, 34)]
    ):
        dtypes[c] = "float32"
    for c in str_type:
        dtypes[c] = "category"

    return cols, dtypes


def load_ieee_cis_train_set(path: str, name: str):
    """load IEEE-CIS train set"""

    cols, dtypes = get_ieee_cis_columns()

    # LOAD TRAIN
    if "label_shift" in path:

        cols += ["id_0" + str(x) for x in range(1, 10)]
        cols += ["id_" + str(x) for x in range(10, 39)]
        cols += ["DeviceType", "DeviceInfo"]

        if "low" in name:
            dataset_name = "low_label_shift.csv"
        elif "mid" in name:
            dataset_name = "mid_label_shift.csv"
        elif "high" in name:
            dataset_name = "high_label_shift.csv"
        else:
            dataset_name = name + ".csv"

        x_train = pd.read_csv(
            path + dataset_name,
            index_col="TransactionID",
            dtype=dtypes,
            usecols=cols + ["isFraud"],
        )

        # TARGET
        y_train = x_train["isFraud"].copy()

        del x_train["isFraud"]
        gc.collect()
        return x_train, y_train

    x_train = pd.read_csv(
        path + "train_transaction.csv",
        index_col="TransactionID",
        dtype=dtypes,
        usecols=cols + ["isFraud"],
    )
    train_id = pd.read_csv(
        path + "train_identity.csv", index_col="TransactionID", dtype=dtypes
    )
    x_train = x_train.merge(train_id, how="left", left_index=True, right_index=True)

    # TARGET
    y_train = x_train["isFraud"].copy()

    del train_id, x_train["isFraud"]
    gc.collect()

    return x_train, y_train


def load_ieee_cis_test_set(path: str):

    """load IEEE-CIS test set"""
    print(f"\n[D] Loading test set from {path}")
    cols, dtypes = get_ieee_cis_columns()

    # LOAD TEST
    x_test = pd.read_csv(
        path + "test_transaction.csv",
        index_col="TransactionID",
        dtype=dtypes,
        usecols=cols,
    )
    train_id = pd.read_csv(
        path + "train_identity.csv", index_col="TransactionID", dtype=dtypes
    )
    test_id = pd.read_csv(
        path + "test_identity.csv", index_col="TransactionID", dtype=dtypes
    )
    fix = dict(zip(test_id.columns, train_id.columns))
    test_id.rename(columns=fix, inplace=True)
    x_test = x_test.merge(test_id, how="left", left_index=True, right_index=True)

    del test_id, train_id
    gc.collect()

    return x_test


def add_hour_day_month_ieee_cis(df: pd.DataFrame):

    """add timestamp, hour, day, and month columns to dataset"""

    # create target columns with day and month numbers
    START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")

    # add timestamp column
    df["timestamp"] = df["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )

    # add month column
    df["DT_M"] = df["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )
    df["DT_M"] = (df["DT_M"].dt.year - 2017) * 12 + df["DT_M"].dt.month

    # add day column
    df["DT_D"] = df["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )

    min_date = df["DT_D"].min()
    df["DT_D"] = df["DT_D"].apply(lambda x: (x - min_date))

    df["DT_D"] = df["DT_D"].dt.days

    # add hour column
    df["DT_H"] = df["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )

    df["DT_H"] = df["DT_H"].sub(df["DT_H"].iloc[0]).dt.total_seconds() // 3600


def split_train_val(
    train_val_df: pd.DataFrame,
    y_train_val: pd.Series,
    time_interval: int,
):

    val_size = int(0.3 * len(train_val_df))

    curr_time = train_val_df["timestamp"].max()
    curr_val_len = 0
    while curr_val_len < val_size:
        daily_transactions = train_val_df.loc[
            (train_val_df["timestamp"] <= curr_time)
            & (train_val_df["timestamp"] > curr_time - timedelta(hours=time_interval))
        ].copy()

        val = daily_transactions.loc[
            daily_transactions.index[int(0.7 * len(daily_transactions)) :]
        ].copy()
        y_val = y_train_val.loc[
            daily_transactions.index[int(0.7 * len(daily_transactions)) :]
        ].copy()

        if curr_time == train_val_df["timestamp"].max():
            val_data = val
            val_labels = y_val
        else:
            val_data = val_data.append(val)
            val_labels = val_labels.append(y_val)

        curr_time = curr_time - timedelta(hours=time_interval)
        curr_val_len = curr_val_len + len(val)

    train_data = train_val_df.loc[
        ~train_val_df.index.isin(val_data.index)
    ]
    train_labels = y_train_val.loc[
        ~y_train_val.index.isin(val_labels.index)
    ]

    assert (val_data.index == val_labels.index).all()
    assert (train_data.index == train_labels.index).all()

    train_data = train_data.drop(columns=["timestamp", "DT_H", "DT_D", "DT_M"])
    val_data = val_data.drop(columns=["timestamp", "DT_H", "DT_D", "DT_M"])

    return train_data, train_labels, val_data, val_labels


def split_ieee_cis(
    df: pd.DataFrame,
    labels: pd.Series,
    target_column: str,
    time_interval: int,
):
    """split dataset into training, validation, and test sets"""

    # split training dataset into training, validation, and testing
    dim = 590540

    idxTrain_val = df.index[: 2 * dim // 6]
    idxTest = df.index[2 * dim // 6 :]


    add_hour_day_month_ieee_cis(df)
    train_val_df = df.loc[idxTrain_val]
    y_train_val = labels.loc[idxTrain_val]


    train_data, train_labels, val_data, val_labels = split_train_val(
        train_val_df=train_val_df,
        y_train_val=y_train_val,
        time_interval=time_interval,
    )

    # test data
    test_data = df.loc[idxTest]
    test_data[target_column] = labels[idxTest]

    return train_data, train_labels, val_data, val_labels, test_data
