#!/usr/bin/env python

import defs
import numpy as np
import pandas as pd
from predict_utils import (
    compute_fraud_rate,
    compute_TNR,
    compute_TPR,
    predict_confusion_matrix,
)


def main():

    print(
        "[D] --------------------------- GENERATE DELAY RESULTS ---------------------------"
    )

    DATASET_PATH = defs.BASE_DATASETS_PATH + "new/timeInterval_10-rand_sample.pkl"
    DATASET_SAVE_PATH = (
        defs.BASE_DATASETS_PATH + "new/timeInterval_10-rand_sample-new_deltas.pkl"
    )
    METRICS_PATH = defs.BASE_DATASETS_PATH + "pre-generated/tmp/"
    TIME_INTERVAL = 10
    DELAY_PERIODS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 17, 34]

    print("\n[D] Load Results")

    dataset = pd.read_pickle(DATASET_PATH)
    metrics = [
        pd.read_pickle(METRICS_PATH + "metrics-timeInterval_10-noRetrain-seed_1.pkl")
    ]
    metrics += [
        pd.read_pickle(
            METRICS_PATH
            + f"metrics-timeInterval_10-retrainPeriodHours_{i}-retrainMode_single-seed_1.pkl"
        )
        for i in range(10, defs.MAX_TIME, TIME_INTERVAL)
    ]

    print("\n[D] Add Confusion Matrix Rates")
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

    print("\n[D] Add Deltas")

    # no delay
    dataset["delta-TPR-0"] = dataset["avg-TPR-retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR-0"] = dataset["avg-TNR-retrain"] - dataset["curr-TNR"]
    dataset["delta-TPR-nop-0"] = dataset["avg-TPR-no_retrain"] - dataset["curr-TPR"]
    dataset["delta-TNR-nop-0"] = dataset["avg-TNR-no_retrain"] - dataset["curr-TNR"]

    # add deltas
    for delay in DELAY_PERIODS:
        for index, row in dataset.iterrows():
            # last retrained model without delay a.k.a. current model without delay
            prev_hour = row["prev_retrain_hour"]
            curr_hour = row["curr_retrain_hour"]  # current time instant
            # last retrained model WITH delay a.k.a. current model WITH delay
            delay_prev_hour = prev_hour - delay * TIME_INTERVAL
            # model we obtain if we retrain now WITH delay
            delay_curr_hour = curr_hour - delay * TIME_INTERVAL

            if delay_prev_hour < 0 or (curr_hour + TIME_INTERVAL) > defs.MAX_TIME:
                continue

            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]
            desired_metrics_after_training = metrics[delay_curr_hour // TIME_INTERVAL]

            # current metrics of the system accounting for delay
            res_before_training = desired_metrics_before_training.loc[
                curr_hour // TIME_INTERVAL - 1
            ]
            # if we DO NOT retrain, these are the metrics our system will have
            res_after_no_training = desired_metrics_before_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]
            # if we retrain, these are the metrics our system will have
            res_after_training = desired_metrics_after_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]

            dataset.at[index, f"delta-TPR-{delay}"] = compute_TPR(
                res_after_training["count_tp"], res_after_training["count_fn"]
            ) - compute_TPR(
                res_before_training["count_tp"], res_before_training["count_fn"]
            )
            dataset.at[index, f"delta-TNR-{delay}"] = compute_TNR(
                res_after_training["count_tn"], res_after_training["count_fp"]
            ) - compute_TNR(
                res_before_training["count_tn"], res_before_training["count_fp"]
            )
            dataset.at[index, f"delta-TPR-nop-{delay}"] = compute_TPR(
                res_after_no_training["count_tp"], res_after_no_training["count_fn"]
            ) - compute_TPR(
                res_before_training["count_tp"], res_before_training["count_fn"]
            )
            dataset.at[index, f"delta-TNR-nop-{delay}"] = compute_TNR(
                res_after_no_training["count_tn"], res_after_no_training["count_fp"]
            ) - compute_TNR(
                res_before_training["count_tn"], res_before_training["count_fp"]
            )

        print(f"\n\t[L] Delay {delay}")

    print("\n[D] Add Delay Metrics")

    # add delay metrics
    for delay in DELAY_PERIODS:
        for index, row in dataset.iterrows():
            prev_hour, curr_hour = row["prev_retrain_hour"], row["curr_retrain_hour"]
            delay_prev_hour, delay_curr_hour = (
                prev_hour - delay * TIME_INTERVAL,
                curr_hour - delay * TIME_INTERVAL,
            )

            if delay_prev_hour < 0:
                continue

            # metrics of the last retrained model WITH delay
            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]

            delay_curr_before_training = desired_metrics_before_training.loc[
                delay_curr_hour // TIME_INTERVAL - 1
            ]
            curr_before_training = desired_metrics_before_training.loc[
                curr_hour // TIME_INTERVAL - 1
            ]

            delayed_TPR = compute_TPR(
                delay_curr_before_training["count_tp"],
                delay_curr_before_training["count_fn"],
            )
            delayed_TNR = compute_TNR(
                delay_curr_before_training["count_tn"],
                delay_curr_before_training["count_fp"],
            )
            delayed_fraud_rate = compute_fraud_rate(
                delay_curr_before_training["count_tp"],
                delay_curr_before_training["count_tn"],
                delay_curr_before_training["count_fp"],
                delay_curr_before_training["count_fn"],
            )

            dataset.at[index, f"delayed-TPR-{delay}"] = delayed_TPR
            dataset.at[index, f"delayed-TNR-{delay}"] = delayed_TNR
            dataset.at[index, f"delayed_fraud_rate-{delay}"] = delayed_fraud_rate

            dataset.at[index, f"curr-TPR-{delay}"] = compute_TPR(
                curr_before_training["count_tp"], curr_before_training["count_fn"]
            )
            dataset.at[index, f"curr-TNR-{delay}"] = compute_TNR(
                curr_before_training["count_tn"], curr_before_training["count_fp"]
            )
            dataset.at[index, f"curr_fraud_rate-{delay}"] = compute_fraud_rate(
                curr_before_training["count_tp"],
                curr_before_training["count_tn"],
                curr_before_training["count_fp"],
                curr_before_training["count_fn"],
            )

            # ESTIMATED DELTAS BASED ON DELAYED CURR TPR/TNR
            # if we DO NOT retrain, these are the metrics our system will have
            res_after_no_training = desired_metrics_before_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]

            # if retrain happens, these are the metrics
            desired_metrics_after_training = metrics[delay_curr_hour // TIME_INTERVAL]
            # if we retrain, these are the metrics our system will have
            res_after_training = desired_metrics_after_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]

            # add estimated deltas based on delayed curr TPR/TNR
            dataset.at[index, f"delta-est-TPR-{delay}-delayed"] = (
                compute_TPR(
                    res_after_training["count_tp"], res_after_training["count_fn"]
                )
                - delayed_TPR
            )
            dataset.at[index, f"delta-est-TNR-{delay}-delayed"] = (
                compute_TNR(
                    res_after_training["count_tn"], res_after_training["count_fp"]
                )
                - delayed_TNR
            )
            dataset.at[index, f"delta-est-TPR-nop-{delay}-delayed"] = (
                compute_TPR(
                    res_after_no_training["count_tp"], res_after_no_training["count_fn"]
                )
                - delayed_TPR
            )
            dataset.at[index, f"delta-est-TNR-nop-{delay}-delayed"] = (
                compute_TNR(
                    res_after_no_training["count_tn"], res_after_no_training["count_fp"]
                )
                - delayed_TNR
            )

        print(f"\n\t[L] Delay {delay}")

    print("\n[D] Add Delay Predictions")

    # add predictions
    for delay in DELAY_PERIODS:
        for index, row in dataset.iterrows():
            prev_hour, curr_hour = row["prev_retrain_hour"], row["curr_retrain_hour"]
            delay_prev_hour, delay_curr_hour = (
                prev_hour - delay * TIME_INTERVAL,
                curr_hour - delay * TIME_INTERVAL,
            )

            if delay_prev_hour < 0:
                continue

            # metrics of the last retrained model WITH delay
            desired_metrics_before_training = metrics[delay_prev_hour // TIME_INTERVAL]

            test = desired_metrics_before_training.loc[curr_hour // TIME_INTERVAL - 1]

            test_scores = test["scores"]
            test_predictions = test["predictions"]
            test_scores = np.stack((1 - test_scores, test_scores)).T

            # Only 1 chunk of data ==> small validation set
            # validation = desired_metrics_before_training.loc[
            #     delay_curr_hour // TIME_INTERVAL - 1
            # ]

            # val_labels = validation["real_labels"]
            # val_scores = validation["scores"]
            # val_predictions = validation["predictions"]
            # val_scores = np.stack((1 - val_scores, val_scores)).T

            # TPR, TNR, fraud_rate = predict_confusion_matrix(
            #     val_scores,
            #     val_predictions,
            #     val_labels,
            #     test_scores,
            #     test_predictions,
            #     classes=True,
            # )

            # dataset.at[index, f"predict-TPR-{delay}-CBATC-small-val"] = TPR
            # dataset.at[index, f"predict-TNR-{delay}-CBATC-small-val"] = TNR
            # dataset.at[
            #     index, f"predict_fraud_rate-{delay}-CBATC-small-val"
            # ] = fraud_rate

            # TPR, TNR, fraud_rate = predict_confusion_matrix(
            #     val_scores,
            #     val_predictions,
            #     val_labels,
            #     test_scores,
            #     test_predictions,
            #     classes=False,
            # )

            # dataset.at[index, f"predict-TPR-{delay}-ATC-small-val"] = TPR
            # dataset.at[index, f"predict-TNR-{delay}-ATC-small-val"] = TNR
            # dataset.at[index, f"predict_fraud_rate-{delay}-ATC-small-val"] = fraud_rate

            # All chunks of data between delay_prev and delay_curr ==> big validation set

            validation = desired_metrics_before_training.loc[
                [
                    t // TIME_INTERVAL
                    for t in range(
                        delay_prev_hour,
                        delay_curr_hour,
                        TIME_INTERVAL,
                    )
                ]
            ]

            all_val_labels = validation["real_labels"].to_numpy()
            all_val_scores = validation["scores"].to_numpy()
            all_val_predictions = validation["predictions"].to_numpy()

            val_labels = np.concatenate(all_val_labels)
            val_scores = np.concatenate(all_val_scores)
            val_predictions = np.concatenate(all_val_predictions)
            val_scores = np.stack((1 - val_scores, val_scores)).T

            # CB-ATC BIG VAL
            cbatc_TPR, cbatc_TNR, fraud_rate = predict_confusion_matrix(
                val_scores,
                val_predictions,
                val_labels,
                test_scores,
                test_predictions,
                classes=True,
            )
            dataset.at[index, f"predict-TPR-{delay}-CBATC-big-val"] = cbatc_TPR
            dataset.at[index, f"predict-TNR-{delay}-CBATC-big-val"] = cbatc_TNR
            dataset.at[index, f"predict_fraud_rate-{delay}-CBATC-big-val"] = fraud_rate

            # if retrain happens, these are the metrics
            desired_metrics_after_training = metrics[delay_curr_hour // TIME_INTERVAL]

            # if we DO NOT retrain, these are the metrics our system will have
            res_after_no_training = desired_metrics_before_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]
            # if we retrain, these are the metrics our system will have
            res_after_training = desired_metrics_after_training.loc[
                (curr_hour // TIME_INTERVAL)
            ]

            # future TPR and TNR if we retrain and if we do not retrain
            future_TPR_retrain = compute_TPR(
                res_after_training["count_tp"], res_after_training["count_fn"]
            )
            future_TNR_retrain = compute_TNR(
                res_after_training["count_tn"], res_after_training["count_fp"]
            )
            future_TPR_nop = compute_TPR(
                res_after_no_training["count_tp"], res_after_no_training["count_fn"]
            )
            future_TNR_nop = compute_TNR(
                res_after_no_training["count_tn"], res_after_no_training["count_fp"]
            )

            # compute deltas based on ATC and CB-ATC current estimated values
            dataset.at[index, f"delta-est-TPR-{delay}-CBATC-big-val"] = (
                future_TPR_retrain - cbatc_TPR
            )
            dataset.at[index, f"delta-est-TNR-{delay}-CBATC-big-val"] = (
                future_TNR_retrain - cbatc_TNR
            )
            dataset.at[index, f"delta-est-TPR-nop-{delay}-CBATC-big-val"] = (
                future_TPR_nop - cbatc_TPR
            )
            dataset.at[index, f"delta-est-TNR-nop-{delay}-CBATC-big-val"] = (
                future_TNR_nop - cbatc_TNR
            )

            # ATC BIG VAL
            atc_TPR, atc_TNR, fraud_rate = predict_confusion_matrix(
                val_scores,
                val_predictions,
                val_labels,
                test_scores,
                test_predictions,
                classes=False,
            )

            dataset.at[index, f"predict-TPR-{delay}-ATC-big-val"] = atc_TPR
            dataset.at[index, f"predict-TNR-{delay}-ATC-big-val"] = atc_TNR
            dataset.at[index, f"predict_fraud_rate-{delay}-ATC-big-val"] = fraud_rate

            # compute deltas based on ATC and CB-ATC current estimated values
            dataset.at[index, f"delta-est-TPR-{delay}-ATC-big-val"] = (
                future_TPR_retrain - atc_TPR
            )
            dataset.at[index, f"delta-est-TNR-{delay}-ATC-big-val"] = (
                future_TNR_retrain - atc_TNR
            )
            dataset.at[index, f"delta-est-TPR-nop-{delay}-ATC-big-val"] = (
                future_TPR_nop - atc_TPR
            )
            dataset.at[index, f"delta-est-TNR-nop-{delay}-ATC-big-val"] = (
                future_TNR_nop - atc_TNR
            )

        print(f"\n\t[L] Delay {delay}")

        dataset.to_pickle(DATASET_SAVE_PATH)

        print("\n\t[D] Save data checkpoint")


if __name__ == "__main__":
    main()
