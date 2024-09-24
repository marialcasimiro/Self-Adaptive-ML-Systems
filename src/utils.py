#!/usr/bin/env python

import os
import subprocess as sub

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

import defs

def add_to_dict(dictionary: dict, key: str, data: np.ndarray):
    """add value to dictionary"""
    if key not in dictionary:
        dictionary[key] = []

    dictionary[key].append(data)


def get_stat(data: np.ndarray, stat: str):
    """get desired statistic from data"""

    if "avg" in stat:
        return np.mean(data)
    elif "stdv" in stat:
        return np.std(data)
    elif "50" in stat:
        return np.percentile(data, 50)
    elif "75" in stat:
        return np.percentile(data, 75)
    elif "90" in stat:
        return np.percentile(data, 90)
    else:  # "99" in stat:
        return np.percentile(data, 99)


def compute_feature_jsd(
    feature: str,
    stats_dict: dict,
    data: pd.DataFrame,
    index: int,  # encodes the end index of the target window
    size: int,  # encodes how many time intervals are in the target window
    val_feature: np.ndarray,  # feature in the validation set to use as ref window when there is not enough old data
):
    print(f"[D] Computing {feature} JSD")

    avg_transactions = 1245  # int(df_retrain["num_transactions"].mean() // 1)
    if "uncertainty_fraud" in feature:
        avg_transactions = 600
    print(f"Avg-transactions per time period: {avg_transactions}")
    print(f"index: {index}")
    print(f"size: {size}")

    ref_win = []
    target_win = []

    # there is no old data for the ref-window
    # so let's use the val_feature
    if index - size - 1 < 0 or len(data[feature]) == 0:
        ref_win.extend(val_feature[-3 * avg_transactions :])
    else:
        iterator = 1
        while len(ref_win) < 3 * avg_transactions and index - size - iterator >= 0:
            curr_feature = data[feature].to_numpy()[index - size - iterator]
            if len(ref_win) + len(curr_feature) <= 3 * avg_transactions:
                ref_win.extend(curr_feature)
                iterator += 1
            else:
                diff = 3 * avg_transactions - len(ref_win)
                ref_win.extend(curr_feature[-diff:])
        # there is not enough old data for the ref-window
        # so fill the difference with val_feature
        if len(ref_win) < 3 * avg_transactions:
            diff = 3 * avg_transactions - len(ref_win)
            ref_win.extend(val_feature[-diff:])

    iterator = 1
    while len(target_win) < avg_transactions and index - iterator >= 0:
        curr_feature = data[feature].to_numpy()[index - iterator]
        if len(target_win) + len(curr_feature) <= avg_transactions:
            target_win.extend(curr_feature)
            iterator += 1
        else:
            diff = avg_transactions - len(target_win)
            target_win.extend(curr_feature[:diff])
    # there is not enough old data for the target-window
    # so fill the difference with val_feature
    if len(target_win) < avg_transactions:
        diff = avg_transactions - len(target_win)
        target_win.extend(val_feature[-diff:])

    print(f"ref-win size: {len(ref_win)}")
    print(f"target-win size: {len(target_win)}")

    ref_win_hist, bins = np.histogram(ref_win)
    add_to_dict(
        stats_dict,
        f"{feature}-JSD",
        distance.jensenshannon(
            p=ref_win_hist, q=np.histogram(target_win, bins=bins)[0]
        ),
    )


def compute_stats(
    stats_dict: dict,
    df_no_retrain: pd.DataFrame,
    df_retrain: pd.DataFrame,
    index_low: int,
    index_high: int,
):

    stats = ["avg", "stdv", "50", "75", "90", "99"]

    print("\n[D] Computing stats for classification metrics")
    for metric in df_retrain.columns.values:
        # print(metric)
        if (
            "scores" in metric
            or "predictions" in metric
            or "retrain_occurred" in metric
            or "uncertainty" in metric
        ):
            continue

        if "real_labels" in metric:
            first = True
            for dataframe in [df_retrain, df_no_retrain]:
                conf_matrix_dict = {
                    "true_pos": [],
                    "true_neg": [],
                    "false_pos": [],
                    "false_neg": [],
                }
                for index in range(index_low, index_high):
                    # print(dataframe["real_labels"][index])
                    conf_matrix = confusion_matrix(  # tn, fp, fn, tp
                        dataframe["real_labels"][index],
                        dataframe["predictions"][index],
                    ).ravel()
                    conf_matrix_dict["true_neg"].append(conf_matrix[0])
                    conf_matrix_dict["false_pos"].append(conf_matrix[1])
                    conf_matrix_dict["false_neg"].append(conf_matrix[2])
                    conf_matrix_dict["true_pos"].append(conf_matrix[3])

                if first:
                    label = "retrain"
                    first = False
                else:
                    label = "no_retrain"
                for stat in stats:
                    add_to_dict(
                        stats_dict,
                        f"{stat}-TP-{label}",
                        get_stat(conf_matrix_dict["true_pos"], stat),
                    )
                    add_to_dict(
                        stats_dict,
                        f"{stat}-TN-{label}",
                        get_stat(conf_matrix_dict["true_neg"], stat),
                    )
                    add_to_dict(
                        stats_dict,
                        f"{stat}-FP-{label}",
                        get_stat(conf_matrix_dict["false_pos"], stat),
                    )
                    add_to_dict(
                        stats_dict,
                        f"{stat}-FN-{label}",
                        get_stat(conf_matrix_dict["false_neg"], stat),
                    )

                    if "avg" in stat:
                        add_to_dict(
                            stats_dict,
                            f"avg-TPR-{label}",
                            stats_dict[f"avg-TP-{label}"][-1]
                            / (
                                stats_dict[f"avg-TP-{label}"][-1]
                                + stats_dict[f"avg-FN-{label}"][-1]
                            ),
                        )

                        add_to_dict(
                            stats_dict,
                            f"avg-TNR-{label}",
                            stats_dict[f"avg-TN-{label}"][-1]
                            / (
                                stats_dict[f"avg-TN-{label}"][-1]
                                + stats_dict[f"avg-FP-{label}"][-1]
                            ),
                        )

                        add_to_dict(
                            stats_dict,
                            f"avg-FNR-{label}",
                            stats_dict[f"avg-FN-{label}"][-1]
                            / (
                                stats_dict[f"avg-TP-{label}"][-1]
                                + stats_dict[f"avg-FN-{label}"][-1]
                            ),
                        )

                        add_to_dict(
                            stats_dict,
                            f"avg-FPR-{label}",
                            stats_dict[f"avg-FP-{label}"][-1]
                            / (
                                stats_dict[f"avg-TN-{label}"][-1]
                                + stats_dict[f"avg-FP-{label}"][-1]
                            ),
                        )

        else:
            for stat in stats:
                add_to_dict(
                    stats_dict,
                    f"{stat}-{metric}-retrain",
                    get_stat(df_retrain[metric].to_numpy()[index_low:index_high], stat),
                )

                add_to_dict(
                    stats_dict,
                    f"{stat}-{metric}-no_retrain",
                    get_stat(
                        df_no_retrain[metric].to_numpy()[index_low:index_high], stat
                    ),
                )

    for rate in ["TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "FNR"]:
        add_to_dict(
            stats_dict,
            f"y-{rate}",
            stats_dict[f"avg-{rate}-retrain"][-1]
            - stats_dict[f"avg-{rate}-no_retrain"][-1],
        )


def compute_uncertainty_stats(
    stats_dict: dict,
    dataset: pd.DataFrame,
    curr_idx: int,
    prev_idx: int,
    val_features,
):
    # uncertainty stats have been computed
    # extract curr and prev uncertainties
    for idx, key in zip([curr_idx, prev_idx], ["curr", "prev"]):
        data = dataset
        uncertainty_key = "uncertainty"
        if idx == -1:
            data = val_features
            uncertainty_key = "val_uncertainty"
            idx = 0
        add_to_dict(
            stats_dict, f"{key}_uncertainty", np.mean(data[uncertainty_key][idx])
        )
        add_to_dict(
            stats_dict,
            f"{key}_uncertainty_fraud",
            np.mean(data[f"{uncertainty_key}_fraud"][idx]),
        )
        add_to_dict(
            stats_dict,
            f"{key}_uncertainty_legit",
            np.mean(data[f"{uncertainty_key}_legit"][idx]),
        )


def compute_performance_related_features(
    stats_dict: dict,
    tmp_dict: dict,
    dataset,
    time_interval,
):

    index_dict = {"index_low": 0, "index_high": 0}

    index_dict["index_low"] = tmp_dict["curr_retrain_hour"] // time_interval
    index_dict["index_high"] = index_dict["index_low"] + 1

    # the current confusion matrix is computed based on the previously trained model
    prev_idx = tmp_dict["prev_retrain_hour"] // time_interval - 1
    for idx, key in zip(
        [index_dict["index_low"] - 1, prev_idx, prev_idx + 1],
        ["curr", "prev", "next2prev"],
    ):
        if idx < 0:
            conf_matrix = np.full(4, -1)
        elif "next2prev" in key and stats_dict["retrain_delta"][-1] <= time_interval:
            conf_matrix = np.full(4, -1)
        else:
            conf_matrix = confusion_matrix(  # tn, fp, fn, tp
                dataset["real_labels"][idx],
                dataset["predictions"][idx],
            ).ravel()

        add_to_dict(stats_dict, f"{key}_tn", conf_matrix[0])
        add_to_dict(stats_dict, f"{key}_fp", conf_matrix[1])
        add_to_dict(stats_dict, f"{key}_fn", conf_matrix[2])
        add_to_dict(stats_dict, f"{key}_tp", conf_matrix[3])
        print(f"{idx}: {conf_matrix}")

    # save fraud-rate
    stats_dict["curr_fraud_rate"].append(
        (stats_dict["curr_tp"][-1] + stats_dict["curr_fn"][-1])
        / (
            stats_dict["curr_tp"][-1]
            + stats_dict["curr_tn"][-1]
            + stats_dict["curr_fp"][-1]
            + stats_dict["curr_fn"][-1]
        )
    )

    print(index_dict)
    return index_dict


def compute_uncertainty(
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
):
    uncertainty_dict = {
        "all": [],
        "fraud": [],
        "legit": [],
    }

    for idx, score in enumerate(scores):
        uncertainty = score - threshold
        uncertainty_dict["all"].append(np.abs(uncertainty))
        if predictions[idx] == 1:
            uncertainty_dict["fraud"].append(uncertainty)
        else:
            uncertainty_dict["legit"].append(uncertainty)

    return uncertainty_dict


def get_correlation(data1: np.ndarray, data2: np.ndarray):
    corr_dict = {}
    for corr_func in [pearsonr]:

        correlation, _ = corr_func(data1, data2)

        add_to_dict(
            corr_dict,
            f"{corr_func.__name__}",
            correlation,
        )

    return corr_dict


def save_results(path: str, file_name: str, results: pd.DataFrame):
    """save test/retrain results to file"""
    if not os.path.exists(path):
        os.makedirs(path)

    results.to_pickle(path + f"{file_name}")

    print(f"[D] Test results and metrics saved to {path}{file_name}")


def get_dataset_path(dataset_name: str):

    path = defs.BASE_DATASETS_PATH

    return path


def load_file(dataste_name: str, file_name: str):

    dir = get_dataset_path(dataste_name)
    try:
        file = pd.read_pickle(dir + f"tmp/{file_name}")
    except FileNotFoundError:
        print(f"[E] Could not find file {file_name} in dir {dir}tmp/.")
        print(f"[W] Looking up file in dir {dir}pre-generated/tmp/")
        file = pd.read_pickle(dir + f"pre-generated/tmp/{file_name}")
    return file

def get_file(dir: str, file_name: str):
    if file_name.endswith(".pkl"):
        open_file_func = pd.read_pickle
    elif file_name.endswith(".parquet"):
        open_file_func = pd.read_parquet
    else:
        open_file_func = pd.read_csv
    
    try:
        f = open_file_func(dir + file_name)
    except FileNotFoundError:
        print(f"[E] Could not find file {file_name} in dir {dir}")
        dir_tokens = dir.strip().split("/")
        new_dir = ""
        for i in range(len(dir_tokens)-2):
            new_dir = new_dir + dir_tokens[i] + "/"
        print(f"[W] Looking up file in dir {new_dir}pre-generated/{dir_tokens[-2]}/")
        f = open_file_func(new_dir + f"pre-generated/{dir_tokens[-2]}/" + file_name)
    return f

def load_data(path, time_intervals, retrain_periods):

    datasets_metrics = {}
    datasets_times = {}

    path = path + "tmp/"
    print(f"[D] Trying to load data from {path}")

    for time_interval in time_intervals:
        key = f"timeInterval_{time_interval}-retrainPeriod_0"
        datasets_metrics[key] = get_file(
            path, f"metrics-timeInterval_{time_interval}-noRetrain-seed_1.pkl"
        )
        datasets_times[key] = get_file(
            path, f"times-timeInterval_{time_interval}-noRetrain-seed_1.pkl"
        )

        retrain_mode = "single"

        for retrain_period in retrain_periods:
            key = f"timeInterval_{time_interval}-retrainPeriod_{retrain_period}"
            file_name = f"metrics-timeInterval_{time_interval}-retrainPeriodHours_{retrain_period}-retrainMode_{retrain_mode}-seed_1.pkl"
            print(f"[D] Loading file {file_name}")
            datasets_metrics[key] = get_file(path, file_name)

            file_name = f"times-timeInterval_{time_interval}-retrainPeriodHours_{retrain_period}-retrainMode_{retrain_mode}-seed_1.pkl"
            datasets_times[key] = get_file(path, file_name)

    return datasets_metrics, datasets_times


def exec_bash_cmd(cmd):

    return sub.run(cmd, shell=True, capture_output=True, check=True)


