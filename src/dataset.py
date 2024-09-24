#!/usr/bin/env python

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import base

import defs
from data_preprocess_IEEE_CIS import (
    do_feature_engineering,
    feature_selection,
    label_encode_and_mem_reduce,
    normalize_d_columns,
)
from eval_metrics import (
    compute_and_plot_roc_curve,
    compute_metrics,
    get_predictions,
    get_threshold_at_fpr,
)
from prism import Prism
from utils import (
    add_to_dict,
    compute_feature_jsd,
    compute_performance_related_features,
    compute_stats,
    compute_uncertainty,
    compute_uncertainty_stats,
    get_correlation,
    get_dataset_path,
    load_data,
    save_results,
)
from utils_IEEE_CIS import (
    add_hour_day_month_ieee_cis,
    load_ieee_cis_test_set,
    load_ieee_cis_train_set,
    split_ieee_cis,
    split_train_val,
)


class Dataset:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-lines

    def __init__(
        self,
        time_interval: int,
        name="ieee-cis",
        target_fpr=0.01,
    ):
        """initialize dataset object"""

        self.target_fpr = target_fpr
        self.threshold = -1
        self.dataset = name
        self.seed = 1
        self.time_interval = time_interval

        if "ieee-cis" in name:
            self.path = get_dataset_path(name)
            self.time_column = "DT_H"
            self.target_column = "labels"
            self.init_train_time = 0.0

            print(f"\n[D] Loading dataset from {self.path}original/")
            x_train, y_train = load_ieee_cis_train_set(self.path + "original/", name)
            x_test = load_ieee_cis_test_set(get_dataset_path("ieee-cis") + "original/")

            print("\n[D] Performing feature engineering.")

            normalize_d_columns(
                x_train, x_test
            )  # normalize columns according to kaggle's solution

            label_encode_and_mem_reduce(
                x_train, x_test
            )  # label encode and memory reduce

            # create train, validation, and test splits
            train_data, train_labels, val_data, val_labels, test_data = split_ieee_cis(
                x_train,
                y_train,
                self.target_column,
                self.time_interval,
            )

            # feature engineering must be performed independently for
            # train, val and test to ensure that the train and val are
            # the same for all datasets (original and with shifts)
            do_feature_engineering(train_data)
            do_feature_engineering(val_data)
            do_feature_engineering(test_data)

            # feature selection must be performed after feature engineering
            self.features = feature_selection(list(train_data.columns))
            print(f"\n[D] Using {len(self.features)} features")

            add_hour_day_month_ieee_cis(train_data)
            add_hour_day_month_ieee_cis(val_data)
            add_hour_day_month_ieee_cis(test_data)

            self.x_train = train_data
            self.y_train = train_labels
            self.x_val = val_data
            self.y_val = val_labels
            self.test_data = test_data

            self.start_time = self.test_data[self.time_column].min()
            self.end_time = self.test_data[self.time_column].max()

            print("\n\n[D] Dataset ready for training and testing.\n")

    def __set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)

    def __update_threshold(self, new_threshold: float):
        self.threshold = new_threshold

    def __set_init_train_time(self, train_time: float):
        self.init_train_time = train_time

    def recompute_threshold(
        self,
        model: base.ClassifierMixin,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ):

        threshold_start = time.time()
        val_scores = model.predict_proba(x_val[self.features].copy())[:, 1]

        fpr, _, thresholds = compute_and_plot_roc_curve(y_val.copy(), val_scores)

        # compute predictions for a specific FPR (e.g. 10%)
        threshold = get_threshold_at_fpr(self.target_fpr, fpr, thresholds)
        threshold_time = time.time() - threshold_start
        print(f"[D] Threshold @ target FPR {self.target_fpr}   ---   {threshold}")

        return threshold, threshold_time

    def __train_model(
        self,
        model: base.ClassifierMixin,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        train_start = time.time()
        model.fit(
            x_train.copy(),
            y_train.copy(),
            eval_set=[(x_val.copy(), y_val.copy())],
            verbose=100,
            early_stopping_rounds=100,
        )
        train_time = time.time() - train_start

        # compute FPR, TPR, and threshold based on validation set
        print("\n[D] Computing ROC-AUC curve based on validation set.")
        threshold, threshold_time = self.recompute_threshold(
            model=model, x_val=x_val.copy(), y_val=y_val.copy()
        )

        return model, train_time, threshold, threshold_time

    def get_initial_model(self, seed=1):
        """Train initial model"""

        self.__set_seed(seed)

        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=1,
            colsample_bytree=1,
            missing=-1,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=self.seed,
            tree_method="hist",
        )

        print("\n[D] Training initial model.")

        clf, train_time, threshold, _ = self.__train_model(
            model=clf,
            x_train=self.x_train[self.features].copy(),
            y_train=self.y_train.copy(),
            x_val=self.x_val[self.features].copy(),
            y_val=self.y_val.copy(),
        )

        self.__set_init_train_time(train_time)

        self.__update_threshold(threshold)

        return clf

    def __retrain(
        self,
        last_retrain,
        curr_timestamp,
        clf: base.ClassifierMixin,
        time_interval: int,
    ):
        """retrain model"""

        # create new training set with old training set
        new_train_val_set = self.x_train.copy()
        y_new_train_val_set = self.y_train.copy()

        # append initial validation set
        new_train_val_set = new_train_val_set.append(self.x_val)
        y_new_train_val_set = y_new_train_val_set.append(self.y_val)

        # append test samples already used for training
        print(f"[D] last retrain was at {last_retrain}")

        if (
            pd.Timestamp(last_retrain)
            > self.test_data.loc[self.test_data[self.time_column] == self.start_time][
                "timestamp"
            ].min()
        ):
            test_samples = self.test_data.loc[
                self.test_data["timestamp"] < last_retrain
            ].copy()
            new_train_val_set = new_train_val_set.append(
                test_samples[list(self.x_train.columns)]
            )
            y_new_train_val_set = y_new_train_val_set.append(
                test_samples[self.target_column]
            )

        # get new samples
        new_samples = self.test_data.loc[
            (self.test_data["timestamp"] >= pd.Timestamp(last_retrain))
            & (self.test_data["timestamp"] < pd.Timestamp(curr_timestamp))
        ].copy()

        # append new samples
        new_train_val_set = new_train_val_set.append(
            new_samples[list(self.x_train.columns)]
        )
        y_new_train_val_set = y_new_train_val_set.append(
            new_samples[self.target_column]
        )

        new_x_train, new_y_train, new_x_val, new_y_val = split_train_val(
            train_val_df=new_train_val_set,
            y_train_val=y_new_train_val_set,
            time_interval=time_interval,
        )

        return self.__train_model(
            model=clf,
            x_train=new_x_train[self.features].copy(),
            y_train=new_y_train.copy(),
            x_val=new_x_val[self.features].copy(),
            y_val=new_y_val.copy(),
        )

    def retrain(
        self,
        results_dict: dict,
        tmp_dict: dict,
        model: base.ClassifierMixin,
        curr_timestamp,
    ):
        last_retrain = results_dict["retrain_timestamp"][-1]
        results_dict["retrain_timestamp"].append(curr_timestamp)
        print("[D] Retraining at " + str(list(results_dict["retrain_timestamp"])[-1]))
        results_dict["retrain_hour"].append(tmp_dict["_time"])

        (model, retrain_time, threshold, recompute_threshold_time) = self.__retrain(
            last_retrain=last_retrain,
            curr_timestamp=curr_timestamp,
            clf=model,
            time_interval=tmp_dict["time_interval"],
        )

        results_dict["amount_old_data"].append(
            results_dict["amount_old_data"][-1] + results_dict["amount_new_data"][-1]
        )
        results_dict["amount_new_data"].append(tmp_dict["curr_new_samples"])
        results_dict["amount_new_fraud"].append(tmp_dict["curr_new_fraud_samples"])
        results_dict["retrain_cost"].append(int(np.mean(results_dict["retrain_times"])))
        results_dict["retrain_times"].append(retrain_time)
        results_dict["threshold_times"].append(recompute_threshold_time)
        results_dict["thresholds"].append(threshold)
        results_dict["feature_imp"].append(
            sorted(zip(model.feature_importances_, self.features))
        )

        tmp_dict["retrain_period_counter"] = 0
        tmp_dict["curr_new_samples"] = 0
        tmp_dict["curr_new_fraud_samples"] = 0

        return model, threshold

    def __update_scores_and_preds(
        self,
        results_dict: dict,
        model: base.ClassifierMixin,
        threshold: float,
        retrain_time: int,
        time_interval: int,
    ):

        _time = self.start_time
        counter = 0

        while _time < retrain_time:
            batch = self.test_data.loc[
                (self.test_data[self.time_column] >= _time)
                & (self.test_data[self.time_column] < _time + time_interval)
            ].copy()

            scores = model.predict_proba(batch[self.features])[:, 1]
            predictions = get_predictions(threshold, scores)

            results_dict["scores"][counter] = scores
            results_dict["predictions"][counter] = predictions

            counter += 1
            _time += time_interval

    def store_results(
        self,
        results_dict: dict,
        tmp_dict: dict,
        batch: pd.DataFrame,
    ):
        results_dict["real_labels"].append(batch[self.target_column].values)

        uncertainties = compute_uncertainty(
            scores=results_dict["scores"][-1],
            predictions=results_dict["predictions"][-1],
            threshold=tmp_dict["threshold"],
        )

        results_dict["uncertainty"].append(uncertainties["all"])
        results_dict["uncertainty_fraud"].append(uncertainties["fraud"])
        results_dict["uncertainty_legit"].append(uncertainties["legit"])

        results_dict["num_transactions"].append(batch[self.time_column].count())
        results_dict["num_fraud_transactions"].append(batch[self.target_column].sum())

        if batch.loc[
            (batch["predictions"] == 1) & (batch[self.target_column] == 1)
        ].empty:
            results_dict["avg_money_tp"].append(0)
            results_dict["count_tp"].append(0)
        else:
            results_dict["count_tp"].append(
                batch.loc[
                    (batch["predictions"] == 1) & (batch[self.target_column] == 1)
                ]["TransactionAmt"].count()
            )
            results_dict["avg_money_tp"].append(
                batch.loc[
                    (batch["predictions"] == 1) & (batch[self.target_column] == 1)
                ]["TransactionAmt"].sum()
                / results_dict["count_tp"][-1]
            )

        if batch.loc[(batch["predictions"] == 1) & (batch[self.target_column] == 0)][
            "TransactionAmt"
        ].empty:
            results_dict["avg_money_fp"].append(0)
            results_dict["count_fp"].append(0)
        else:
            results_dict["count_fp"].append(
                batch.loc[
                    (batch["predictions"] == 1) & (batch[self.target_column] == 0)
                ]["TransactionAmt"].count()
            )
            results_dict["avg_money_fp"].append(
                batch.loc[
                    (batch["predictions"] == 1) & (batch[self.target_column] == 0)
                ]["TransactionAmt"].sum()
                / results_dict["count_fp"][-1]
            )

        if batch.loc[(batch["predictions"] == 0) & (batch[self.target_column] == 1)][
            "TransactionAmt"
        ].empty:
            results_dict["avg_money_fn"].append(0)
            results_dict["count_fn"].append(0)
        else:
            results_dict["count_fn"].append(
                batch.loc[
                    (batch["predictions"] == 0) & (batch[self.target_column] == 1)
                ]["TransactionAmt"].count()
            )
            results_dict["avg_money_fn"].append(
                batch.loc[
                    (batch["predictions"] == 0) & (batch[self.target_column] == 1)
                ]["TransactionAmt"].sum()
                / results_dict["count_fn"][-1]
            )

        if batch.loc[(batch["predictions"] == 0) & (batch[self.target_column] == 0)][
            "TransactionAmt"
        ].empty:
            results_dict["avg_money_tn"].append(0)
            results_dict["count_tn"].append(0)
        else:
            results_dict["count_tn"].append(
                batch.loc[
                    (batch["predictions"] == 0) & (batch[self.target_column] == 0)
                ]["TransactionAmt"].count()
            )
            results_dict["avg_money_tn"].append(
                batch.loc[
                    (batch["predictions"] == 0) & (batch[self.target_column] == 0)
                ]["TransactionAmt"].sum()
                / results_dict["count_tn"][-1]
            )

        # save new samples
        tmp_dict["new_samples"].extend(batch[self.features].values.tolist())
        tmp_dict["y_new_samples"].extend(batch[self.target_column].values.tolist())
        tmp_dict["curr_new_samples"] += len(batch[self.target_column])
        tmp_dict["curr_new_fraud_samples"] += batch[self.target_column].sum()

    def __evaluate_model_and_store_results(
        self,
        model: base.ClassifierMixin,
        batch: pd.DataFrame,
        tmp_dict: dict,
        results_dict: dict,
    ):

        # get scores
        scores = model.predict_proba(batch[self.features])[:, 1]

        # compute predictions based on threshold (set to achieve a given FPR)
        results_dict["predictions"].append(
            get_predictions(tmp_dict["threshold"], scores)
        )
        batch["predictions"] = results_dict["predictions"][-1]

        # save scores
        results_dict["scores"].append(scores)

        self.store_results(results_dict, tmp_dict, batch)

    def test(
        self,
        time_interval: int,
        model: base.ClassifierMixin,
        test_prism: bool = False,
        do_retrain: bool = False,
        retrain_period: int = None,
        retrain_mode: str = "multi",
    ):
        """test model"""

        results_dict = {
            "scores": [],
            "predictions": [],
            "real_labels": [],
            "uncertainty": [],
            "uncertainty_fraud": [],
            "uncertainty_legit": [],
            "num_transactions": [],
            "num_fraud_transactions": [],
            "total_times": [],
            "nop_util": [
                -1,
                -1,
            ],  # these must be initialized with 2 values
            "retrain_util": [-1, -1],  # because PRISM is not called for
            "benefits_model_features": [-1, -1],  # the 1st 2 updates of the other lists
            "benefits_model_predictions": [-1, -1],
            "benefits_model_target_tpr": [-1, -1],
            "benefits_model_target_tnr": [-1, -1],
            "nop_model_predictions": [-1, -1],
            "prism_time_overhead": [-1, -1],
            "time_series_time_overhead": [-1, -1],
            "time_series_preds": [-1, -1],
            "retrain_occurred": [],
            "avg_money_tp": [],  # amount of TP transactions: money saved due to fraud that was caught
            "avg_money_fn": [],  # amount of FN transactions: money lost due to fraud not caught
            "avg_money_tn": [],
            "avg_money_fp": [],
            "count_tp": [],
            "count_tn": [],
            "count_fp": [],
            "count_fn": [],
            "amount_old_data": [len(self.y_train)],
            "amount_new_data": [0],
            "amount_new_fraud": [0],
            "retrain_cost": [int(self.init_train_time)],
            "retrain_times": [self.init_train_time],
            "retrain_timestamp": [
                self.test_data.loc[self.test_data[self.time_column] == self.start_time][
                    "timestamp"
                ].min()
            ],
            "retrain_hour": [self.start_time],
            "threshold_times": [0],
            "thresholds": [self.threshold],
            "feature_imp": [sorted(zip(model.feature_importances_, self.features))],
        }

        tmp_dict = {
            "retrain": False,
            "new_samples": [],
            "y_new_samples": [],
            "curr_new_samples": 0,
            "curr_new_fraud_samples": 0,
            "len_train": len(self.y_train),
            "_time": self.start_time,
            "retrain_period_counter": 0,
            "time_interval": time_interval,
            "start_time": 0,  # time instant of when the simulation starts
            "end_time": 0,  # time instant of when the simulation ends
            "val_scores": model.predict_proba(self.x_val[self.features].copy())[:, 1],
            "threshold": self.threshold,
            "curr_timestamp": self.test_data.loc[
                self.test_data[self.time_column] == self.start_time
            ]["timestamp"].min(),
        }
        # val_scores will be used to compute scores-JSD when there
        # is not enough data in the reference window
        # same for val_uncertainty
        tmp_dict["val_uncertainties"] = compute_uncertainty(
            scores=tmp_dict["val_scores"],
            predictions=get_predictions(tmp_dict["threshold"], tmp_dict["val_scores"]),
            threshold=tmp_dict["threshold"],
        )

        # to test prism I need to train the model to predict benefits
        # to do this, I'll use part of the test set
        if test_prism:
            prism = Prism(
                time_interval=time_interval,
                model_type=defs.RB_MODEL_TYPE,
                dataset_name=self.dataset,
                seed=self.seed,
                nop_models=defs.NOP_MODELS,
                retrain_models=defs.RETRAIN_MODELS,
            )

            tmp_dict["test_start_time"] = prism.get_test_start_time()

            tmp_dict["_time"] = self.test_data.loc[
                self.test_data[self.time_column]
                >= self.start_time + tmp_dict["test_start_time"]
            ][self.time_column].min()

            batch = self.test_data.loc[
                self.test_data[self.time_column] < tmp_dict["_time"]
            ].copy()

            self.__evaluate_model_and_store_results(
                model=model,
                batch=batch,
                tmp_dict=tmp_dict,
                results_dict=results_dict,
            )

            model, tmp_dict["threshold"] = self.retrain(
                results_dict, tmp_dict, model, tmp_dict["test_start_time"]
            )

            results_dict["retrain_occurred"].append(True)
            results_dict["total_times"].append(0)

        # START EVALUATION
        while tmp_dict["_time"] < self.end_time:
            tmp_dict["start_time"] = time.time()
            batch = self.test_data.loc[
                (self.test_data[self.time_column] >= tmp_dict["_time"])
                & (self.test_data[self.time_column] < tmp_dict["_time"] + time_interval)
            ].copy()

            tmp_dict["curr_timestamp"] = batch.loc[
                batch[self.time_column] == tmp_dict["_time"]
            ]["timestamp"].min()

            if do_retrain and len(tmp_dict["new_samples"]) > 0:
                if (tmp_dict["retrain_period_counter"] == retrain_period) or (
                    retrain_period is None and np.random.uniform(0, 1) > 0.5
                ):
                    tmp_dict["retrain"] = True
                    if (
                        "single" in retrain_mode
                        and len(results_dict["retrain_times"]) == 2
                    ):
                        tmp_dict["retrain"] = False
                        tmp_dict["retrain_period_counter"] = 0

            if test_prism and tmp_dict["curr_new_samples"] > 0:
                tmp_dict["retrain"] = prism.ask_prism(
                    results_dict,
                    tmp_dict,
                )
                print("[D] Going to retrain: " + str(tmp_dict["retrain"]).upper())

            results_dict["retrain_occurred"].append(tmp_dict["retrain"])
            if tmp_dict["retrain"]:
                model, tmp_dict["threshold"] = self.retrain(
                    results_dict,
                    tmp_dict,
                    model,
                    tmp_dict["curr_timestamp"],
                )
                tmp_dict["retrain"] = False

                model.save_model(
                    f"{self.path}tmp/models/timeInterval_{time_interval}-retrainPeriodHours_{retrain_period}-retrainMode_{retrain_mode}-seed_{self.seed}.txt"
                )

                if "single" in retrain_mode:
                    self.__update_scores_and_preds(
                        results_dict,
                        model,
                        tmp_dict["threshold"],
                        tmp_dict["_time"],
                        time_interval,
                    )

            self.__evaluate_model_and_store_results(
                model=model,
                batch=batch,
                tmp_dict=tmp_dict,
                results_dict=results_dict,
            )

            # advance time
            tmp_dict["_time"] = tmp_dict["_time"] + time_interval

            # advance retrain_period_counter
            tmp_dict["retrain_period_counter"] += time_interval

            tmp_dict["end_time"] = time.time()
            add_to_dict(
                results_dict,
                "total_times",
                tmp_dict["end_time"] - tmp_dict["start_time"],
            )

        # compute metrics and store results
        metrics_df = compute_metrics(
            results_dict["real_labels"],
            results_dict["predictions"],
            results_dict["scores"],
            self.target_fpr,
        )[0]

        for key, value in results_dict.items():
            print(f"{key}: {len(value)}")
            if len(value) == len(results_dict["real_labels"]):
                metrics_df[key] = results_dict[key]

        if do_retrain or test_prism:
            return (
                pd.DataFrame(
                    {
                        "retrain_times": results_dict["retrain_times"],
                        "recompute_threshold_times": results_dict["threshold_times"],
                        "retrain_timestamp": results_dict["retrain_timestamp"],
                        "retrain_hour": results_dict["retrain_hour"],
                        "retrain_cost": results_dict["retrain_cost"],
                        "amount_old_data": results_dict["amount_old_data"],
                        "amount_new_data": results_dict["amount_new_data"],
                        "amount_new_fraud": results_dict["amount_new_fraud"],
                        "thresholds": results_dict["thresholds"],
                        "feature_imp": results_dict["feature_imp"],
                    }
                ),
                self.path,
                metrics_df,
            )
        else:
            return (
                pd.DataFrame(
                    {
                        "thresholds": self.threshold,
                        "val_scores": [tmp_dict["val_scores"]],
                        "val_uncertainty": [tmp_dict["val_uncertainties"]["all"]],
                        "val_uncertainty_fraud": [
                            tmp_dict["val_uncertainties"]["fraud"]
                        ],
                        "val_uncertainty_legit": [
                            tmp_dict["val_uncertainties"]["legit"]
                        ],
                        "amount_old_data": results_dict["amount_old_data"],
                        "feature_imp": results_dict["feature_imp"],
                        "retrain_timestamp": results_dict["retrain_timestamp"],
                    },
                    index=[0],
                ),
                self.path,
                metrics_df,
            )

    def __compute_corrcoef(
        self,
        stats_dict: dict,
        new_data: pd.DataFrame,
        new_samples: list,
        y_new_samples: list,
        sampling_method: str = "rand",
    ):
        old_data = (
            self.x_train[self.features]
            .copy()
            .append(pd.DataFrame(new_samples, columns=self.features))
        )
        old_data[self.target_column] = self.y_train.copy().append(
            pd.Series(y_new_samples)
        )

        stats_dict["old_fraud_rate"].append(
            old_data[self.target_column].sum() / old_data[self.target_column].count()
        )

        if len(old_data.index) >= len(new_data.index):
            old_data_strat = old_data.sample(
                n=len(new_data.index), random_state=self.seed
            )
            new_data_strat = new_data.copy()
        else:
            old_data_strat = old_data
            new_data_strat = new_data.sample(
                n=len(old_data.index), random_state=self.seed
            )

        for idx, col1 in enumerate(old_data.columns.values):
            for col2 in old_data.columns.values[idx:]:

                # compute correlation between the same features in the old VS new data
                if col1 == col2:
                    corr_dict = get_correlation(
                        old_data_strat[col1].to_numpy(), new_data_strat[col2].to_numpy()
                    )

                    for key, corr in corr_dict.items():
                        add_to_dict(
                            stats_dict,
                            f"{key}-{col1}-old_vs_new",
                            corr[0],
                        )

    def __compute_time_related_features(
        self,
        stats_dict: dict,
        dataset_names,
        datasets_times,
    ):
        # dataset_names[0] : has the name of the current retrain period
        # dataset_names[1] : has the name of the previous retrain period

        test_data = self.test_data.copy()

        tmp_dict = {}
        # TIME
        # retrain_period is the time difference between two consecutive retrains
        tmp_dict["curr_retrain_hour"] = int(
            dataset_names[0].split("-")[1].split("_")[1]
        )
        tmp_dict["prev_retrain_hour"] = int(
            dataset_names[1].split("-")[1].split("_")[1]
        )
        tmp_dict["retrain_period"] = (
            tmp_dict["curr_retrain_hour"] - tmp_dict["prev_retrain_hour"]
        )

        # the previous retrain was the initial one, at "time 0"
        if tmp_dict["prev_retrain_hour"] == 0:
            tmp_dict["prev_timestamp"] = test_data["timestamp"].min()
        else:
            tmp_dict["prev_timestamp"] = datasets_times[dataset_names[1]][
                "retrain_timestamp"
            ][1]

        stats_dict["period_start_time"].append(tmp_dict["prev_timestamp"])

        stats_dict["retrain_delta"].append(tmp_dict["retrain_period"])

        stats_dict["period_end_time"].append(
            datasets_times[dataset_names[0]]["retrain_timestamp"][1]
        )

        stats_dict["curr_retrain_hour"].append(tmp_dict["curr_retrain_hour"])
        stats_dict["prev_retrain_hour"].append(tmp_dict["prev_retrain_hour"])

        return tmp_dict

    def __compute_data_related_features(
        self,
        stats_dict: dict,
        tmp_dict: dict,
        dataset_names,
        datasets_times,
        sampling_method,
    ):
        test_data = self.test_data.copy()

        # DATA RELATED METRICS
        print("\n[D] Computing correlation metrics")
        batch = test_data.loc[
            (test_data["timestamp"] >= tmp_dict["prev_timestamp"])
            & (
                test_data["timestamp"]
                < datasets_times[dataset_names[0]]["retrain_timestamp"][1]
            )
        ]
        new_data = batch[self.features].copy()
        new_data[self.target_column] = batch[self.target_column].copy()

        old_data = test_data.loc[(test_data["timestamp"] < tmp_dict["prev_timestamp"])]

        # compute correlation coefficient between old and new data
        self.__compute_corrcoef(
            stats_dict,
            new_data,
            old_data[self.features].copy(),
            old_data[self.target_column].copy(),
            sampling_method,
        )

        # save amount of old and new data
        stats_dict["amount_old_data"].append(
            len(old_data[self.target_column]) + self.y_train.count()
        )
        stats_dict["amount_new_data"].append(batch[self.target_column].count())

        stats_dict["total_data"].append(
            stats_dict["amount_new_data"][-1] + stats_dict["amount_old_data"][-1]
        )
        stats_dict["ratio_new_old_data"].append(
            stats_dict["amount_new_data"][-1] / stats_dict["amount_old_data"][-1]
        )

    def __compute_features(
        self,
        stats_dict,
        dataset_names,
        dataset,
        datasets_times,
        prev_dataset,
        sampling_method,
        time_interval,
        val_features,
    ):

        # dataset_names[0] : has the name of the current retrain period
        # dataset_names[1] : has the name of the previous retrain period

        tmp_dict = self.__compute_time_related_features(
            stats_dict, dataset_names, datasets_times
        )

        index_dict = compute_performance_related_features(
            stats_dict, tmp_dict, prev_dataset, time_interval
        )

        for feature, val_feature in zip(
            ["scores", "uncertainty", "uncertainty_fraud", "uncertainty_legit"],
            [
                val_features["val_scores"][0],
                val_features["val_uncertainty"][0],
                val_features["val_uncertainty_fraud"][0],
                val_features["val_uncertainty_legit"][0],
            ],
        ):
            compute_feature_jsd(
                feature,
                stats_dict,
                prev_dataset,
                index_dict["index_low"],
                tmp_dict["retrain_period"] // time_interval,
                val_feature,
            )

        compute_stats(
            stats_dict,
            prev_dataset,
            dataset,
            index_dict["index_low"],
            index_dict["index_high"],
        )

        compute_uncertainty_stats(
            stats_dict=stats_dict,
            dataset=prev_dataset,
            curr_idx=index_dict["index_low"] - 1,
            prev_idx=tmp_dict["prev_retrain_hour"] // time_interval - 1,
            val_features=val_features,
        )

        self.__compute_data_related_features(
            stats_dict,
            tmp_dict,
            dataset_names,
            datasets_times,
            sampling_method,
        )

    def estimate_benefits(
        self,
        time_interval,
        datasets_metrics,
        datasets_times,
        stats_dict,
        sampling_method,
    ):

        # compare retrain and no retrain until last retrain timestamp
        for dataset_name, dataset in datasets_metrics.items():
            df_times = datasets_times[dataset_name]

            # this corresponds to the no retrain case
            if int(dataset_name.split("-")[1].split("_")[1]) == 0:
                continue

            # this corresponds to the last model retrain
            if (df_times["retrain_timestamp"][1]) >= self.test_data["timestamp"].max():
                continue

            for prev_dataset_name, prev_dataset in datasets_metrics.items():
                # we only want to analyze models that were trained before the current timestamp
                if int(prev_dataset_name.split("-")[1].split("_")[1]) >= int(
                    dataset_name.split("-")[1].split("_")[1]
                ):
                    continue

                print("\n ----------------------------------------------------")
                print(f"[D] Current retrain: {dataset_name}")
                print(f"[D] Previous retrain: {prev_dataset_name}")

                for tcol in df_times.columns.values:
                    if "times" in tcol or "thresholds" in tcol:
                        add_to_dict(
                            stats_dict,
                            tcol,
                            df_times.iloc[1][tcol],
                        )

                self.__compute_features(
                    stats_dict,
                    [dataset_name, prev_dataset_name],
                    dataset,
                    datasets_times,
                    prev_dataset,
                    sampling_method,
                    time_interval,
                    datasets_times[f"timeInterval_{time_interval}-retrainPeriod_0"],
                )

    def generate_new_dataset(
        self,
        time_interval: int,
        retrain_periods,
        sampling_method: str = "rand",
        use_pgf: bool = False,
    ):
        """
        generate dataset by computing avg metrics
        for each test period, as well as metrics
        that compare new and old data
        """

        stats_dict = {
            "amount_old_data": [],
            "amount_new_data": [],
            "total_data": [],
            "ratio_new_old_data": [],
            "curr_fraud_rate": [],
            "old_fraud_rate": [],
            "retrain_delta": [],  # time since last retrain, in hours
            "period_start_time": [],
            "period_end_time": [],
            "curr_retrain_hour": [],
            "prev_retrain_hour": [],
        }

        # load all data files
        print("[D] Loading retrain files")
        path = self.path
        if use_pgf:
            path = path + "pre-generated/"
        datasets_metrics, datasets_times = load_data(
            path, [time_interval], retrain_periods
        )

        self.estimate_benefits(
            time_interval, datasets_metrics, datasets_times, stats_dict, sampling_method
        )

        for key, value in stats_dict.items():
            if len(value) < len(stats_dict["amount_old_data"]):
                print(f"{key}: {len(value)}")

        results_df = pd.DataFrame.from_dict(stats_dict)

        save_results(
            self.path + "new/",
            f"timeInterval_{time_interval}-{sampling_method}_sample.pkl",
            results_df,
        )
