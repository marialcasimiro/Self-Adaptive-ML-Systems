#!/usr/bin/env python

import os
import time

import defs
import numpy as np
import pandas as pd
import scipy
from predict_utils import estimate_confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error
from time_series import (
    build_time_series_and_predict,
    set_up_time_series_hyperparam_tuning,
)
from utils import add_to_dict, compute_feature_jsd, exec_bash_cmd, get_dataset_path
from utils_IEEE_CIS import add_hour_day_month_ieee_cis, load_ieee_cis_train_set

# PRISM does not support floats (only ints and bools)
# so deltas and confusion matrix rates have to be rounded
PRISM_ROUNDING_FACTOR = 10000


def exec_prism(args, tactics):

    results = {}
    for tactic in tactics:
        cmd = f"{defs.PRISM_PATH}{defs.PRISM_EXEC_FILE} {args} {tactic} > {defs.PRISM_PATH}{tactic}_output_log.txt"
        print(f"[D]\t{cmd}")
        proc_res = exec_bash_cmd(cmd)
        results[tactic] = proc_res.returncode

    print(results)
    return results


def get_prism_utils(_dir, tactics):
    utils_dict = {}
    for f in os.listdir(_dir):
        if f.endswith(".txt") and "adv" in f:
            with open(_dir + f, "r", encoding="utf-8") as ff:
                lines = ff.readlines()
                for line in lines:
                    if "Result" in line:
                        continue
                    sysU = line

            for tactic in tactics:
                if tactic in f:
                    utils_dict[tactic] = float(sysU)

    return utils_dict


def rename_columns(col):
    if "delta" not in col and "avg" not in col:
        if "TPR" in col:
            return "curr-TPR"
        if "TNR" in col:
            return "curr-TNR"
        if "fraud_rate" in col:
            return "curr_fraud_rate"

    return col


class Prism:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    def __init__(
        self,
        time_interval: int,
        model_type: str,
        dataset_name: str,
        seed: int,
        train_split_percent: float = 0.7,  # 70% for train and val; 30% for test
        val_split_percent: float = 0.2,  # 80% for train; 20% for val
        sat_value: float = 0.9,
        adaptation_tactics: list = ["retrain", "nop"],
        nop_models: str = None,  # whether to predict what happens when NOP is executed
        retrain_models: str = "random_forest",
        retrain_cost: int = 8,  # cost of the retrain tactic
        retrain_latency: int = 1,  # how long the retrain tactic takes to execute (in hours)
        fpr_threshold: int = 1,  # SLA agreed maximum FPR (in %)
        recall_threshold: int = 60,  # SLA agreed minimum RECALL (in %)
        fpr_sla_cost: int = 10,  # cost of violating the FPR SLA
        recall_sla_cost: int = 10,  # cost of violating the RECALL SLA
        replace_cost: int = 5,  # cost of replacing the ML model for a rule-based model
        replace_tpr_avg: int = 70,  # tpr offered by the rule-based model
        replace_tpr_std: int = 5,  # tpr offered by the rule-based model
        replace_tnr_avg: int = 97,  # tnr offered by the rule-based model
        replace_tnr_std: int = 2,  # tnr offered by the rule-based model
        baseline: str = "aip",
    ):
        self.time_interval = time_interval
        self.baseline = baseline
        self.model_type = model_type
        self.seed = seed
        self.models_dict = {
            "retrain_models": retrain_models,
            "nop_models": nop_models,
            "sat_value": sat_value,
            "total_data_max": 0,  # maximum allowed value for feature total_data
            "old_data_max": 0,  # maximum allowed value for feature old_data
            "ratio_new_old_data_max": 0,  # maximum allowed value for feature ratio_new_old_data
        }

        self.experiment_constants = {
            "adaptation_tactics": adaptation_tactics,
            "retrain_cost": retrain_cost,
            "retrain_latency": retrain_latency,
            "fpr_threshold": int(fpr_threshold * (PRISM_ROUNDING_FACTOR / 100)),
            "recall_threshold": int(recall_threshold * (PRISM_ROUNDING_FACTOR / 100)),
            "fpr_sla_cost": fpr_sla_cost,
            "recall_sla_cost": recall_sla_cost,
            "replace_cost": replace_cost,
            "replace_tpr_avg": replace_tpr_avg,
            "replace_tpr_std": replace_tpr_std,
            "replace_tnr_avg": replace_tnr_avg,
            "replace_tnr_std": replace_tnr_std,
        }

        dataset_path = get_dataset_path(dataset_name)

        got_file = False
        for f in os.listdir(dataset_path + "new/"):
            if (
                f"timeInterval_{time_interval}" in f
                and "delay_metrics-taas.pkl" in f
                and f.endswith(".pkl")
            ):
                benefits_dataset_file = dataset_path + "new/" + f
                got_file = True
                break

        if not got_file:
            for f in os.listdir(dataset_path + "pre-generated/new/"):
                if f"timeInterval_{time_interval}" in f and f.endswith(".pkl"):
                    benefits_dataset_file = dataset_path + "pre-generated/new/" + f
                    break

        self.aid = pd.read_pickle(benefits_dataset_file)
        print(f"[D] Loaded AID file: {benefits_dataset_file}")

        splits, self.test_start_time = self.__prepare_splits(
            self.aid,
            train_split_percent,
            val_split_percent,
        )

        # whether to estimate deltas via:
        # - estimated current TPR and TNR
        # - actual (real) current TPR and TNR
        self.estimated_deltas = True

        # initialize AIPs
        models = self.__init_benefits_models(train_split=splits[0], val_split=splits[1])
        self.models_dict["retrain_tpr_model"] = models[0]
        self.models_dict["retrain_tnr_model"] = models[1]
        if "random_forest" in nop_models:
            self.models_dict["nop_tpr_model"] = models[2]
            self.models_dict["nop_tnr_model"] = models[3]

        # set up time series models to estimate future number
        # of transactions that will be received
        self.original_dataset, y_train = load_ieee_cis_train_set(
            dataset_path + "original/", dataset_name
        )
        self.original_dataset["isFraud"] = y_train

        add_hour_day_month_ieee_cis(self.original_dataset)
        self.__set_up_arma_target()

    def get_test_start_time(self):
        return self.test_start_time

    def get_model_type(self):
        return self.model_type

    def __set_up_arma_target(self):

        txs_df = self.original_dataset[["timestamp", "isFraud"]].copy()
        txs_df["num_txs"] = txs_df["isFraud"]
        txs_df = txs_df.groupby(pd.Grouper(key="timestamp", freq="1H")).agg(
            {"num_txs": "count", "isFraud": "sum"}
        )
        txs_df["fraud_rate"] = txs_df["isFraud"] / txs_df["num_txs"] * 100

        self.original_dataset = txs_df

    def __prepare_splits(
        self,
        dataset,
        train_split_percent: float = 0.7,  # 70% for train and val; 30% for test
        val_split_percent: float = 0.2,  # 80% for train; 20% for val
    ):
        # transform confusion matrix features into rates
        for prefix in ["curr", "prev", "next2prev"]:
            dataset[f"{prefix}-TPR"] = dataset[f"{prefix}_tp"] / (
                dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
            )
            dataset[f"{prefix}-TNR"] = dataset[f"{prefix}_tn"] / (
                dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
            )
            dataset[f"{prefix}-FNR"] = dataset[f"{prefix}_fn"] / (
                dataset[f"{prefix}_tp"] + dataset[f"{prefix}_fn"]
            )
            dataset[f"{prefix}-FPR"] = dataset[f"{prefix}_fp"] / (
                dataset[f"{prefix}_tn"] + dataset[f"{prefix}_fp"]
            )

        # add delta column
        dataset["delta-TPR"] = dataset["avg-TPR-retrain"] - dataset["curr-TPR"]
        dataset["delta-TNR"] = dataset["avg-TNR-retrain"] - dataset["curr-TNR"]
        dataset["delta-TPR-nop"] = dataset["avg-TPR-no_retrain"] - dataset["curr-TPR"]
        dataset["delta-TNR-nop"] = dataset["avg-TNR-no_retrain"] - dataset["curr-TNR"]

        # split dataset into train, val, test
        duration = dataset["period_end_time"].max() - dataset["period_end_time"].min()

        # use train_split_percent % for training
        train_test_split_time = (
            int(duration.days * train_split_percent) * 24
        )  # in hours
        # ensure it is a multiple of the time_interval
        if train_test_split_time % self.time_interval != 0:
            train_test_split_time = train_test_split_time - (
                train_test_split_time % self.time_interval
            )
        test_start_time = dataset["prev_retrain_hour"].min() + train_test_split_time

        train_val_split = dataset.loc[dataset["curr_retrain_hour"] < test_start_time]

        # split train into train (1-val_split_percent) and val (val_split_percent)
        val_split_time = train_val_split["curr_retrain_hour"].min() + int(
            train_test_split_time * (1 - val_split_percent)
        )

        # TRAIN SPLIT
        train_split = train_val_split.loc[
            train_val_split["curr_retrain_hour"] < val_split_time
        ]

        # filter train split to consider retrain-deltas up to 40 time intervals
        train_split = train_split.loc[train_split["retrain_delta"] <= 400]

        # VAL SPLIT
        val_split = train_val_split.loc[
            (train_val_split["prev_retrain_hour"] >= val_split_time)
            # & (train_val_split["curr_retrain_hour"] >= val_split_time)
        ].copy()

        # saturate features in val and test
        sat_value = self.models_dict["sat_value"]
        max_val = train_split["total_data"].max()
        print(max_val)
        self.models_dict["total_data_max"] = max_val
        val_split["total_data"] = val_split["total_data"].apply(
            lambda x: x if x < sat_value * max_val else sat_value * max_val
        )
        max_val = train_split["amount_old_data"].max()
        print(max_val)
        self.models_dict["old_data_max"] = max_val
        val_split["amount_old_data"] = val_split["amount_old_data"].apply(
            lambda x: x if x < sat_value * max_val else sat_value * max_val
        )
        max_val = train_split["ratio_new_old_data"].max()
        print(max_val)
        self.models_dict["ratio_new_old_data_max"] = max_val
        val_split["ratio_new_old_data"] = val_split["ratio_new_old_data"].apply(
            lambda x: x if x < sat_value * max_val else sat_value * max_val
        )

        # remove NaN rows of delayed metrics
        if "aip" not in self.baseline:
            delay = self.baseline.split("_")[-1]
            cols = []
            for feature in ["-TPR", "-TNR", "_fraud_rate"]:
                cols += [
                    f"curr{feature}-{delay}",
                    f"delayed{feature}-{delay}",
                    f"predict{feature}-{delay}-ATC-big-val",
                    f"predict{feature}-{delay}-CBATC-big-val",
                ]

            train_split = train_split.dropna(subset=cols)
            val_split = val_split.dropna(subset=cols)

        print("[D] SET SIZES:")
        print("[D] \ttrain:" + str(len(train_split)))
        print("[D] \tval:" + str(len(val_split)))

        return [train_split, val_split], test_start_time

    def select_baseline_features(self):
        # specify model features
        baseline_model_features = [
            "amount_new_data",
            # "amount_old_data",
            # "total_data",
            "ratio_new_old_data",
            "retrain_delta",
            "scores-JSD",
            # "curr_fraud_rate",
            # "curr-TPR",
            # "curr-TNR",
        ]

        if "aip" in self.baseline:
            features = baseline_model_features + [
                "curr_fraud_rate",
                "curr-TPR",
                "curr-TNR",
            ]
        else:
            delay = self.baseline.split("_")[-1]
            print(f"[D] Considering feature delay of {delay}")
            if "delayed" in self.baseline:
                features = baseline_model_features + [
                    f"delayed_fraud_rate-{delay}",
                    f"delayed-TPR-{delay}",
                    f"delayed-TNR-{delay}",
                ]
            if "mixed" in self.baseline:
                features = baseline_model_features + [
                    f"predict_fraud_rate-{delay}-ATC-big-val",
                    f"predict-TPR-{delay}-CBATC-big-val",
                    f"predict-TNR-{delay}-CBATC-big-val",
                ]

            if "atc" in self.baseline:
                features = baseline_model_features + [
                    f"predict_fraud_rate-{delay}-ATC-big-val",
                    f"predict-TPR-{delay}-ATC-big-val",
                    f"predict-TNR-{delay}-ATC-big-val",
                ]

            if "cbatc" in self.baseline:
                features = baseline_model_features + [
                    f"predict_fraud_rate-{delay}-CBATC-big-val",
                    f"predict-TPR-{delay}-CBATC-big-val",
                    f"predict-TNR-{delay}-CBATC-big-val",
                ]

            if "atc_small" in self.baseline:
                features = baseline_model_features + [
                    f"predict_fraud_rate-{delay}-without-classes",
                    f"predict-TPR-{delay}-without-classes",
                    f"predict-TNR-{delay}-without-classes",
                ]

            if "cbatc_small" in self.baseline:
                features = baseline_model_features + [
                    f"predict_fraud_rate-{delay}-with-classes",
                    f"predict-TPR-{delay}-with-classes",
                    f"predict-TNR-{delay}-with-classes",
                ]

        return features

    def get_target_cols(self):
        if "delta" in self.model_type:
            target_TPR = "delta-TPR"
            target_TNR = "delta-TNR"
            target_TPR_nop = "delta-TPR-nop"
            target_TNR_nop = "delta-TNR-nop"
        else:
            target_TPR = "avg-TPR-retrain"
            target_TNR = "avg-TNR-retrain"
            target_TPR_nop = "avg-TPR-no_retrain"
            target_TNR_nop = "avg-TNR-no_retrain"

        if "aip" not in self.baseline:
            delay = self.baseline.split("_")[-1]
            # deltas are computed based on estimated current TPR and TNR
            if self.estimated_deltas:
                if "delayed" in self.baseline:
                    target_cols = "delayed"
                elif "cbatc" in self.baseline:
                    target_cols = "CBATC-big-val"
                else:
                    target_cols = "ATC-big-val"
                target_TPR = f"delta-est-TPR-{delay}-{target_cols}"
                target_TNR = f"delta-est-TNR-{delay}-{target_cols}"
                target_TPR_nop = f"delta-est-TPR-nop-{delay}-{target_cols}"
                target_TNR_nop = f"delta-est-TNR-nop-{delay}-{target_cols}"
            else:  # deltas are computed based on the actual current TPR and TNR
                target_TPR = target_TPR + "-" + str(delay)
                target_TNR = target_TNR + "-" + str(delay)
                target_TPR_nop = target_TPR_nop + "-" + str(delay)
                target_TNR_nop = target_TNR_nop + "-" + str(delay)

        return target_TPR, target_TNR, target_TPR_nop, target_TNR_nop

    def __init_benefits_models(self, train_split, val_split):

        features = self.select_baseline_features()
        target_TPR, target_TNR, target_TPR_nop, target_TNR_nop = self.get_target_cols()
        print(
            f"[D] {self.baseline}: using features\n\t{features}\n\tand target cols:\n\t{target_TPR},{target_TNR},{target_TPR_nop},{target_TNR_nop}"
        )

        # select only necessary columns according to the current baseline
        cols = features + [target_TNR, target_TPR, target_TNR_nop, target_TPR_nop]
        train_split = train_split.drop(
            columns=[col for col in train_split if col not in cols]
        )
        val_split = val_split.drop(
            columns=[col for col in val_split if col not in cols]
        )

        # rename features so that feature names are the same regardless of the baseline
        benefits_model_features = []
        for feature in features:
            benefits_model_features.append(rename_columns(feature))

        train_split = train_split.rename(columns=rename_columns)
        val_split = val_split.rename(columns=rename_columns)

        print(f"[D] Features after renaming:\n\t{benefits_model_features}")

        print(f"[D] Training TPR and TNR {self.model_type} models")
        # train models to predict delta-TP and delta-TN
        if "random_forest" in self.models_dict["retrain_models"]:
            retrain_TPR_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.seed,
            )
            retrain_TNR_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.seed,
            )

        retrain_TPR_model.fit(
            train_split[benefits_model_features],
            train_split[target_TPR],
        )

        mae = mean_absolute_error(
            val_split[target_TPR],
            retrain_TPR_model.predict(val_split[benefits_model_features]),
        )
        print(
            f"[D] TPR RB model validation MAE is {mae} and feature importances:\n\t{retrain_TPR_model.feature_importances_}"
        )

        retrain_TNR_model.fit(
            train_split[benefits_model_features],
            train_split[target_TNR],
        )

        mae = mean_absolute_error(
            val_split[target_TNR],
            retrain_TNR_model.predict(val_split[benefits_model_features]),
        )
        print(
            f"[D] TNR RB model validation MAE is {mae} and feature importances:\n\t{retrain_TNR_model.feature_importances_}"
        )

        if "random_forest" in self.models_dict["nop_models"]:
            print(f"[D] Training NOP TPR and TNR {self.model_type} models")
            nop_TPR_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.seed,
            )
            nop_TNR_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.seed,
            )

            nop_TPR_model.fit(
                train_split[benefits_model_features],
                train_split[target_TPR_nop],
            )
            nop_TNR_model.fit(
                train_split[benefits_model_features],
                train_split[target_TNR_nop],
            )

            return (
                retrain_TPR_model,
                retrain_TNR_model,
                nop_TPR_model,
                nop_TNR_model,
            )

        return retrain_TPR_model, retrain_TNR_model

    def __compute_features_for_benefits_prediction(
        self,
        results_dict: dict,
        samples_dict: dict,
    ):

        features_dict = {}

        features = self.select_baseline_features()
        sat_value = self.models_dict["sat_value"]

        features_dict["amount_new_data"] = samples_dict["curr_new_samples"]

        amount_old_data = (
            results_dict["amount_old_data"][-1] + results_dict["amount_new_data"][-1]
        )

        features_dict["ratio_new_old_data"] = (
            features_dict["amount_new_data"] / amount_old_data
        )

        # Saturate features if needed
        if amount_old_data > sat_value * self.models_dict["old_data_max"]:
            amount_old_data = sat_value * self.models_dict["old_data_max"]
            print("[D] Saturating amount_old_data")
        if "amount_old_data" in features:
            features_dict["amount_old_data"] = amount_old_data

        if "total_data" in features:
            features_dict["total_data"] = (
                features_dict["amount_new_data"] + features_dict["amount_old_data"]
            )

            if (
                features_dict["total_data"]
                > sat_value * self.models_dict["total_data_max"]
            ):
                features_dict["total_data"] = (
                    sat_value * self.models_dict["total_data_max"]
                )
                print("[D] Saturating total_data")

        if (
            features_dict["ratio_new_old_data"]
            > sat_value * self.models_dict["ratio_new_old_data_max"]
        ):
            features_dict["ratio_new_old_data"] = (
                sat_value * self.models_dict["ratio_new_old_data_max"]
            )
            print("[D] Saturating ratio_new_old_data")

        features_dict["retrain_delta"] = int(
            (samples_dict["_time"] - results_dict["retrain_hour"][-1])
        )
        print("retrain-delta: " + str(features_dict["retrain_delta"]))

        scores_JSD_dict = {}
        compute_feature_jsd(
            "scores",
            scores_JSD_dict,
            pd.DataFrame(
                {
                    "scores": results_dict["scores"],
                    "num_transactions": results_dict["num_transactions"],
                }
            ),
            index=len(results_dict["scores"]),
            size=int(features_dict["retrain_delta"] // samples_dict["time_interval"]),
            val_feature=samples_dict["val_scores"],
        )

        features_dict["scores-JSD"] = scores_JSD_dict["scores-JSD"][0]

        features_dict["curr_fraud_rate"] = (
            results_dict["num_fraud_transactions"][-1]
            / results_dict["num_transactions"][-1]
        )

        return features_dict

    def get_features_for_benefits_prediction(
        self,
        prism_dict: dict,
        results_dict: dict,
        samples_dict: dict,
    ):

        features_dict = self.__compute_features_for_benefits_prediction(
            results_dict, samples_dict
        )
        features_dict["curr-TPR"] = prism_dict["curr_tpr"]  # curr_tp
        features_dict["curr-TNR"] = prism_dict["curr_tnr"]  # curr_tn

        for col in ["benefits_model_target_tpr", "benefits_model_target_tnr"]:
            if results_dict[col][-1] != -1:
                if "abs" in self.model_type:
                    target = prism_dict["curr_" + col.split("_")[3]]
                else:
                    target = (
                        prism_dict["curr_" + col.split("_")[3]] - results_dict[col][-1]
                    )
                results_dict[col][-1] = target
            add_to_dict(results_dict, col, prism_dict["curr_" + col.split("_")[3]])

        return features_dict

    def __predict_benefits(self, features, prism_dict, results_dict):

        # predict new TPR and new TNR WITH retrain
        models = ["retrain_tpr_model", "retrain_tnr_model"]
        if "random_forest" in self.models_dict["nop_models"]:
            models.extend(["nop_tpr_model", "nop_tnr_model"])
        elif "time_series" in self.models_dict["nop_models"]:
            for target in ["tpr", "tnr"]:
                (
                    prism_dict[f"new_{target.upper()}_noRetrain"],
                    prism_dict[f"new_{target.upper()}_noRetrain_std"],
                    prism_dict[f"new_{target.upper()}_noRetrain_5"],
                    prism_dict[f"new_{target.upper()}_noRetrain_50"],
                    prism_dict[f"new_{target.upper()}_noRetrain_95"],
                    _,
                ) = self.__time_series_predict(
                    target_col=target,
                    future_time_intervals=1,
                    end_time=results_dict["curr_timestamp"][-1],
                    start_time=results_dict["retrain_timestamp"][-1],
                    preds=results_dict["predictions"],
                    labels=results_dict["real_labels"],
                )

                if "delta" in self.model_type:
                    prism_dict[f"new_{target.upper()}_noRetrain"] = prism_dict[
                        f"new_{target.upper()}_noRetrain"
                    ] - round(prism_dict[f"curr_{target}"] * PRISM_ROUNDING_FACTOR)
                    prism_dict[f"new_{target.upper()}_noRetrain_5"] = prism_dict[
                        f"new_{target.upper()}_noRetrain_5"
                    ] - round(prism_dict[f"curr_{target}"] * PRISM_ROUNDING_FACTOR)
                    prism_dict[f"new_{target.upper()}_noRetrain_50"] = prism_dict[
                        f"new_{target.upper()}_noRetrain_50"
                    ] - round(prism_dict[f"curr_{target}"] * PRISM_ROUNDING_FACTOR)
                    prism_dict[f"new_{target.upper()}_noRetrain_95"] = prism_dict[
                        f"new_{target.upper()}_noRetrain_95"
                    ] - round(prism_dict[f"curr_{target}"] * PRISM_ROUNDING_FACTOR)

        # random-forest prediction
        for model in models:
            # use trained AIPs
            retrain_preds = []
            for estimator in self.models_dict[model].estimators_:
                retrain_preds.append(estimator.predict(features.values))
            pred = np.mean(retrain_preds)
            assert round(pred, 5) == round(
                self.models_dict[model].predict(features)[0], 5
            )
            pred_std = np.std(retrain_preds)

            if "abs" in self.model_type:
                pred = min(pred, 1.0)
                pred = max(pred, 0.0)

            print(f"[D] {model} pred={pred}   pred_std = {pred_std}")
            tokens = model.split("_")
            model_type = "retrain"
            if "nop" in tokens[0]:
                model_type = "noRetrain"
            prism_dict[f"new_{tokens[1].upper()}_{model_type}"] = round(
                pred * PRISM_ROUNDING_FACTOR
            )
            prism_dict[f"new_{tokens[1].upper()}_{model_type}_std"] = round(
                pred_std * PRISM_ROUNDING_FACTOR
            )

            prism_dict[f"new_{tokens[1].upper()}_{model_type}_5"] = round(
                pred * PRISM_ROUNDING_FACTOR
            )
            prism_dict[f"new_{tokens[1].upper()}_{model_type}_50"] = round(
                pred * PRISM_ROUNDING_FACTOR
            )
            prism_dict[f"new_{tokens[1].upper()}_{model_type}_95"] = round(
                pred * PRISM_ROUNDING_FACTOR
            )

            print(
                f"[D] {model}: predicted mean = {prism_dict[f'new_{tokens[1].upper()}_{model_type}']}    stdev = {prism_dict[f'new_{tokens[1].upper()}_{model_type}_std']}"
            )
            if pred_std > 0:
                prism_dict[f"new_{tokens[1].upper()}_{model_type}_5"] = int(
                    scipy.stats.norm.ppf(0.05, loc=pred, scale=pred_std)
                    * PRISM_ROUNDING_FACTOR
                )
                prism_dict[f"new_{tokens[1].upper()}_{model_type}_50"] = int(
                    scipy.stats.norm.ppf(0.5, loc=pred, scale=pred_std)
                    * PRISM_ROUNDING_FACTOR
                )
                prism_dict[f"new_{tokens[1].upper()}_{model_type}_95"] = int(
                    scipy.stats.norm.ppf(0.95, loc=pred, scale=pred_std)
                    * PRISM_ROUNDING_FACTOR
                )
            print(
                f"[D]\tperc_5 = {prism_dict[f'new_{tokens[1].upper()}_{model_type}_5']}\tperc_50 = {prism_dict[f'new_{tokens[1].upper()}_{model_type}_50']}\tperc_95 = {prism_dict[f'new_{tokens[1].upper()}_{model_type}_95']}"
            )

    def get_benefits_prediction(self, features, prism_dict, results_dict):

        prism_dict["new_TPR_noRetrain"] = 0
        prism_dict["new_TNR_noRetrain"] = 0
        prism_dict["new_TPR_noRetrain_std"] = 0
        prism_dict["new_TNR_noRetrain_std"] = 0

        prism_dict["new_TPR_noRetrain_5"] = 0
        prism_dict["new_TPR_noRetrain_50"] = 0
        prism_dict["new_TPR_noRetrain_95"] = 0

        prism_dict["new_TNR_noRetrain_5"] = 0
        prism_dict["new_TNR_noRetrain_50"] = 0
        prism_dict["new_TNR_noRetrain_95"] = 0

        print(f"PRISM DICT:\n\t{prism_dict}")

        # actually get model predictions
        self.__predict_benefits(features, prism_dict, results_dict)

        for key, value in prism_dict.items():
            if "new" in key and ("TPR" in key or "TNR" in key):
                prism_dict[key] = int(value)

        for key_out, key_in in zip(
            ["benefits_model_predictions", "nop_model_predictions"],
            ["retrain", "noRetrain"],
        ):
            add_to_dict(
                results_dict,
                key_out,
                {
                    "TPR": prism_dict[f"new_TPR_{key_in}"],
                    "TPR_std": prism_dict[f"new_TPR_{key_in}_std"],
                    "TNR": prism_dict[f"new_TNR_{key_in}"],
                    "TNR_std": prism_dict[f"new_TNR_{key_in}_std"],
                },
            )

    def __set_up_time_series_data(
        self, target_col, end_time, start_time=None, preds=None, labels=None
    ):

        if start_time is not None:

            if preds is not None and labels is not None:
                # compute how many time intervals have passed since the last retrain
                tdelta = end_time - start_time
                target_dict = {target_col: []}
                num_intervals = (
                    (tdelta.seconds / 3600) + tdelta.days * 24
                ) / self.time_interval
                print(
                    f"num intervals = {num_intervals}    days passed: {tdelta.days}    hours passed: {tdelta.seconds/3600}"
                )
                num_intervals = int(num_intervals)
                while num_intervals > 0:
                    conf_mat = confusion_matrix(  # tn, fp, fn, tp
                        labels[-num_intervals],
                        preds[-num_intervals],
                    ).ravel()

                    if "tpr" in target_col:
                        target_dict[target_col].append(
                            conf_mat[3] / (conf_mat[3] + conf_mat[2]) * 100
                        )
                    elif "tnr" in target_col:
                        target_dict[target_col].append(
                            conf_mat[0] / (conf_mat[0] + conf_mat[1]) * 100
                        )

                    num_intervals -= 1

                data = pd.DataFrame(target_dict)

            else:
                data = self.original_dataset.loc[
                    start_time.strftime("%Y-%m-%d %H") : end_time.strftime(
                        "%Y-%m-%d %H"
                    )
                ].copy()

        return data

    def __time_series_predict(
        self,
        target_col,
        future_time_intervals,
        end_time,
        start_time=None,
        preds=None,
        labels=None,
    ):

        print(
            f"[D] Time-series predicting {target_col}: start-time={start_time}    end-time={end_time}"
        )

        if start_time is None and preds is None and labels is None:
            data = self.original_dataset.loc[: end_time.strftime("%Y-%m-%d %H")].copy()
        else:
            data = self.__set_up_time_series_data(
                target_col, end_time, start_time, preds, labels
            )

        order = (1, 0, 1)
        trend = None
        if "tpr" in target_col or "tnr" in target_col:

            if len(data) >= 3:
                _, best = set_up_time_series_hyperparam_tuning(
                    target_col,
                    data.loc[data.index[-1]][target_col],
                    data.loc[data.index[:-1]],
                    future_time_intervals,
                )
                order = (best["order_p"], best["order_d"], best["order_q"])
                trend = best["trend"]

        return build_time_series_and_predict(
            target_col,
            data,
            future_time_intervals,
            order=order,
            trend=trend,
        )

    def compute_prism_inputs(
        self,
        results_dict: dict,
        samples_dict: dict,
    ):
        prism_dict = {}

        prism_dict["curr_tpr"] = 0
        prism_dict["curr_tnr"] = 0
        prism_dict["curr_fnr"] = 0
        prism_dict["curr_fpr"] = 0

        if len(results_dict["real_labels"]) > 0:
            (
                real_curr_tn,
                real_curr_fp,
                real_curr_fn,
                real_curr_tp,
            ) = confusion_matrix(  # tn, fp, fn, tp
                results_dict["real_labels"][-1],
                results_dict["predictions"][-1],
            ).ravel()
            prism_dict["real_curr_tpr"] = (real_curr_tp) / (real_curr_tp + real_curr_fn)
            prism_dict["real_curr_tnr"] = (real_curr_tn) / (real_curr_tn + real_curr_fp)
            prism_dict["real_curr_fnr"] = (real_curr_fn) / (real_curr_tp + real_curr_fn)
            prism_dict["real_curr_fpr"] = (real_curr_fp) / (real_curr_tn + real_curr_fp)
            prism_dict["real_curr_fraud_rate"] = (
                results_dict["num_fraud_transactions"][-1]
                / results_dict["num_transactions"][-1]
            )

            if "aip" not in self.baseline:

                if "mixed" in self.baseline:
                    (
                        curr_tpr,
                        curr_tnr,
                        curr_fnr,
                        curr_fpr,
                        _,
                    ) = estimate_confusion_matrix(
                        features=self.select_baseline_features(),
                        retrain_delta=int(
                            (samples_dict["_time"] - results_dict["retrain_hour"][-1])
                        ),
                        amount_new_data=samples_dict["curr_new_samples"],
                        aid=self.aid,
                    )
                    _, _, _, _, fraud_rate = estimate_confusion_matrix(
                        features=self.select_baseline_features(),
                        retrain_delta=int(
                            (samples_dict["_time"] - results_dict["retrain_hour"][-1])
                        ),
                        amount_new_data=samples_dict["curr_new_samples"],
                        aid=self.aid,
                    )
                else:
                    (
                        curr_tpr,
                        curr_tnr,
                        curr_fnr,
                        curr_fpr,
                        fraud_rate,
                    ) = estimate_confusion_matrix(
                        features=self.select_baseline_features(),
                        retrain_delta=int(
                            (samples_dict["_time"] - results_dict["retrain_hour"][-1])
                        ),
                        amount_new_data=samples_dict["curr_new_samples"],
                        aid=self.aid,
                    )
            else:  # AIP ==> no delay
                curr_tpr = prism_dict["real_curr_tpr"]
                curr_tnr = prism_dict["real_curr_tnr"]
                curr_fnr = prism_dict["real_curr_fnr"]
                curr_fpr = prism_dict["real_curr_fpr"]
                fraud_rate = prism_dict["real_curr_fraud_rate"]

            prism_dict["curr_tpr"] = curr_tpr
            prism_dict["curr_tnr"] = curr_tnr
            prism_dict["curr_fnr"] = curr_fnr
            prism_dict["curr_fpr"] = curr_fpr

        # compute remaining input features for AIP models
        features = self.get_features_for_benefits_prediction(
            prism_dict, results_dict, samples_dict
        )
        print(
            f"[D] {self.baseline}:\n\treal_tpr={round(prism_dict['real_curr_tpr'], 5)}\treal_tnr={round(prism_dict['real_curr_tnr'], 5)}\treal_fnr={round(prism_dict['real_curr_fnr'], 5)}\treal_fpr={round(prism_dict['real_curr_fpr'], 5)}\treal_fraud_rate={round(prism_dict['real_curr_fraud_rate'], 5)}"
        )
        # update value of curr_fraud_rate for
        # baselines = [atc, cbatc, delayed, mixed]
        if "aip" not in self.baseline:
            features["curr_fraud_rate"] = fraud_rate
            print(
                f"\test_tpr={round(curr_tpr, 5)}\t est_tnr={round(curr_tnr, 5)}\t est_fnr={round(curr_fnr, 5)}\t est_fpr={round(curr_fpr, 5)}\t est_fraud_rate={round(fraud_rate, 5)}"
            )

        prism_dict["est_curr_tpr"] = prism_dict["curr_tpr"]
        prism_dict["est_curr_tnr"] = prism_dict["curr_tnr"]
        prism_dict["est_curr_fnr"] = prism_dict["curr_fnr"]
        prism_dict["est_curr_fpr"] = prism_dict["curr_fpr"]
        prism_dict["est_curr_fraud_rate"] = features["curr_fraud_rate"]

        # predict benefits of retrain in terms of delta (increase/decrease) in each cell of the confusion matrix
        print(f"[D] Features for benefits prediction:\n\t{features}")
        add_to_dict(results_dict, "benefits_model_features", features)
        self.get_benefits_prediction(
            pd.DataFrame(features, index=[0]), prism_dict, results_dict
        )

        # round confusion matrix values for PRISM (it only supports ints)
        for col in ["curr_tpr", "curr_tnr", "curr_fnr", "curr_fpr"]:
            prism_dict[col] = round(prism_dict[col] * PRISM_ROUNDING_FACTOR)

        # compute remaining inputs required for PRISM
        prism_dict["horizon"] = 1
        arma_start = time.time()
        (
            prism_dict["expected_num_transactions"],
            _,
            _,
            _,
            _,
            _,
        ) = self.__time_series_predict(
            "num_txs", self.time_interval, samples_dict["curr_timestamp"]
        )
        results_dict["time_series_time_overhead"].append(time.time() - arma_start)
        results_dict["time_series_preds"].append(
            prism_dict["expected_num_transactions"]
        )
        txs_during_retrain = 0
        if self.experiment_constants["retrain_latency"] > 0:
            (txs_during_retrain, _, _, _, _, _,) = self.__time_series_predict(
                "num_txs",
                self.experiment_constants["retrain_latency"],
                samples_dict["curr_timestamp"],
            )

        prism_dict["percent_txs"] = round(
            (txs_during_retrain / prism_dict["expected_num_transactions"]) * 100
        )

        prism_dict["amount_new_data"] = samples_dict["curr_new_samples"]

        prism_dict["retrain_cost"] = self.experiment_constants["retrain_cost"]
        prism_dict["retrain_latency"] = round(
            (self.experiment_constants["retrain_latency"] / self.time_interval) * 100
        )
        prism_dict["fpr_threshold"] = self.experiment_constants["fpr_threshold"]
        prism_dict["recall_threshold"] = self.experiment_constants["recall_threshold"]
        prism_dict["fpr_sla_cost"] = self.experiment_constants["fpr_sla_cost"]
        prism_dict["recall_sla_cost"] = self.experiment_constants["recall_sla_cost"]

        for model_type in ["tpr", "tnr"]:
            avg = self.experiment_constants[f"replace_{model_type}_avg"]
            std = self.experiment_constants[f"replace_{model_type}_std"]
            prism_dict[f"replace_{model_type}_5"] = int(
                scipy.stats.norm.ppf(0.05, loc=avg, scale=std)
            )
            prism_dict[f"replace_{model_type}_50"] = int(
                scipy.stats.norm.ppf(0.5, loc=avg, scale=std)
            )
            prism_dict[f"replace_{model_type}_95"] = int(
                scipy.stats.norm.ppf(0.95, loc=avg, scale=std)
            )

        prism_dict["no_retrain_models"] = (
            "true"
            if (
                "random_forest" in self.models_dict["nop_models"]
                or "time_series" in self.models_dict["nop_models"]
            )
            else "false"
        )

        return prism_dict

    def ask_prism(
        self,
        results_dict: dict,
        samples_dict: dict,
    ):

        print("[D] Computing PRISM inputs")
        prism_dict = self.compute_prism_inputs(
            results_dict,
            samples_dict,
        )

        for key, value in prism_dict.items():
            print(f"{key}: {value}")

        # ask prism whether to adapt
        print("[D] Asking PRISM whether to retrain")

        args = (
            f"HORIZON={prism_dict['horizon']},"
            f"AVG_TRANSACTIONS={prism_dict['expected_num_transactions']},"
            f"PERCENT_TXS={prism_dict['percent_txs']},"
            f"INIT_TNR={prism_dict['curr_tnr']},"
            f"INIT_TPR={prism_dict['curr_tpr']},"
            f"INIT_FNR={prism_dict['curr_fnr']},"
            f"INIT_FPR={prism_dict['curr_fpr']},"
            f"FPR_SLA_COST={prism_dict['fpr_sla_cost']},"
            f"RECALL_SLA_COST={prism_dict['recall_sla_cost']},"
            f"FPR_THRESHOLD={prism_dict['fpr_threshold']},"
            f"RECALL_THRESHOLD={prism_dict['recall_threshold']},"
            f"CURR_NEW_DATA={prism_dict['amount_new_data']},"
            f"new_TPR_retrain={prism_dict['new_TPR_retrain']},"
            f"new_TPR_retrain_std={prism_dict['new_TPR_retrain_std']},"
            f"new_TPR_retrain_5={prism_dict['new_TPR_retrain_5']},"
            f"new_TPR_retrain_50={prism_dict['new_TPR_retrain_50']},"
            f"new_TPR_retrain_95={prism_dict['new_TPR_retrain_95']},"
            f"new_TNR_retrain={prism_dict['new_TNR_retrain']},"
            f"new_TNR_retrain_std={prism_dict['new_TNR_retrain_std']},"
            f"new_TNR_retrain_5={prism_dict['new_TNR_retrain_5']},"
            f"new_TNR_retrain_50={prism_dict['new_TNR_retrain_50']},"
            f"new_TNR_retrain_95={prism_dict['new_TNR_retrain_95']},"
            f"new_TPR_noRetrain={prism_dict['new_TPR_noRetrain']},"
            f"new_TPR_noRetrain_std={prism_dict['new_TPR_noRetrain_std']},"
            f"new_TPR_noRetrain_5={prism_dict['new_TPR_noRetrain_5']},"
            f"new_TPR_noRetrain_50={prism_dict['new_TPR_noRetrain_50']},"
            f"new_TPR_noRetrain_95={prism_dict['new_TPR_noRetrain_95']},"
            f"new_TNR_noRetrain={prism_dict['new_TNR_noRetrain']},"
            f"new_TNR_noRetrain_std={prism_dict['new_TNR_noRetrain_std']},"
            f"new_TNR_noRetrain_5={prism_dict['new_TNR_noRetrain_5']},"
            f"new_TNR_noRetrain_50={prism_dict['new_TNR_noRetrain_50']},"
            f"new_TNR_noRetrain_95={prism_dict['new_TNR_noRetrain_95']},"
            f"RETRAIN_COST={prism_dict['retrain_cost']},"
            f"RETRAIN_LATENCY={prism_dict['retrain_latency']},"
            f"REPLACE_COST={self.experiment_constants['replace_cost']},"
            f"rb_model_TPR_5={prism_dict['replace_tpr_5']},"
            f"rb_model_TPR_50={prism_dict['replace_tpr_50']},"
            f"rb_model_TPR_95={prism_dict['replace_tpr_95']},"
            f"rb_model_TNR_5={prism_dict['replace_tnr_5']},"
            f"rb_model_TNR_50={prism_dict['replace_tnr_50']},"
            f"rb_model_TNR_95={prism_dict['replace_tnr_95']},"
            f"NO_RETRAIN_MODELS={prism_dict['no_retrain_models']},"
            f"BENEFITS_MODEL_TYPE={0 if self.model_type =='delta' else 1}"
        )

        prism_start = time.time()
        results = exec_prism(args, self.experiment_constants["adaptation_tactics"])
        results_dict["prism_time_overhead"].append(time.time() - prism_start)

        if all(value == 0 for value in results.values()):  # terminated successfully
            # check whether prism said to retrain
            utils_dict = get_prism_utils(
                defs.PRISM_PATH, self.experiment_constants["adaptation_tactics"]
            )
            print("[D] PRISM finished running")

        selected_tactic = min(utils_dict, key=utils_dict.get)

        prism_decision_dict = {}
        for tactic in self.experiment_constants["adaptation_tactics"]:
            print(f"\n\t{tactic} util\t{utils_dict[tactic]}")
            add_to_dict(results_dict, f"{tactic}_util", utils_dict[tactic])
            if selected_tactic == tactic:
                prism_decision_dict[tactic] = True
            else:
                prism_decision_dict[tactic] = False

        return prism_decision_dict, prism_dict
