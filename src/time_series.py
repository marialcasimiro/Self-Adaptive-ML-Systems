#!/usr/bin/env python

import itertools
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_time_series_and_predict(
    target, data, future_time_intervals, order=(1, 0, 1), trend=None
):

    if len(data) < 2:
        print("[W] Data for time-series prediction has only 1 value.")
        print("[W] \tAssuming mean as the existing value and 0 stdev.")
        print(data[target])
        mean = data[target].to_numpy()[0]
        stdev = 0
        success = True
    else:
        success = False
        try:
            armaModel = SARIMAX(data[target], order=order, trend=trend).fit()
            preds = armaModel.get_forecast(future_time_intervals)
            mean = np.sum(preds.predicted_mean.to_numpy())
            stdev = np.mean(preds.se_mean.to_numpy())

            if not np.isnan(mean) and not np.isnan(stdev):
                success = True
            else:
                mean = data[target].to_numpy()[0]
                stdev = 0
        except Exception as e:  # pylint: disable=broad-except
            print(f"[E] Time-series params did not work: {e}")
            print("[E] \tAssuming mean as the existing value and 0 stdev.")
            mean = data[target].to_numpy()[0]
            stdev = 0

    print(f"[D] Expected {target} -- mean={mean}   stdev={stdev}")
    # conf_int = preds.conf_int(alpha=0.05)
    # perc_5 = round(conf_int[0, 0])
    # perc_95 = round(conf_int[0, 1])
    perc_5 = mean
    perc_50 = mean
    perc_95 = mean
    if stdev > 0:
        perc_5 = scipy.stats.norm.ppf(0.05, loc=mean, scale=stdev)
        perc_50 = scipy.stats.norm.ppf(0.5, loc=mean, scale=stdev)
        perc_95 = scipy.stats.norm.ppf(0.95, loc=mean, scale=stdev)

    print(
        f"[D] Expected {target} -- perc_5={perc_5}   perc_50={perc_50}   perc_95={perc_95}"
    )

    return (
        round(mean),
        round(stdev),
        round(perc_5),
        round(perc_50),
        round(perc_95),
        success,
    )


def do_hyperparam_tuning(
    hpts, target_col, real_target_val, data, future_time_intervals
):
    results = {
        "p": [],
        "d": [],
        "q": [],
        "trend": [],
        "preds": [],
        "rmse": [],
    }

    start_time = time.time()
    for p, d, q, trend in hpts:
        results["p"].append(p)
        results["d"].append(d)
        results["q"].append(q)
        results["trend"].append(trend)
        print("-" * 50)
        print(
            f"[D] Testing order=({p}, {d}, {q}) and trend={trend}  ---  time-series with {len(data)} points"
        )
        pred, _, _, _, _, success = build_time_series_and_predict(
            target_col, data, future_time_intervals, order=(p, d, q), trend=trend
        )
        results["preds"].append(pred)
        if success:
            results["rmse"].append(
                np.sqrt(mean_squared_error([real_target_val], [pred]))
            )
        else:
            results["rmse"].append(np.Infinity)

        print(
            f"[D] Predicted value={pred}    target value={real_target_val}    rmse={results['rmse'][-1]}"
        )
    end_time = time.time() - start_time

    return results, end_time


def set_up_time_series_hyperparam_tuning(
    target_col, real_target_val, data, future_time_intervals
):

    hyperparams = {
        "order_p": [0, 1, 2],
        "order_d": [0, 1, 2],
        "order_q": [0, 1, 2],
        "trend": ["n", "c", "t", "ct"],
    }

    prod = itertools.product(
        hyperparams["order_p"],
        hyperparams["order_d"],
        hyperparams["order_q"],
        hyperparams["trend"],
    )

    print(
        "[GREP-INDEX];target;len-data;order_p;order_d;order_q;trend;pred;real-val;rmse;tuning-time"
    )

    results, hpt_duration = do_hyperparam_tuning(
        prod, target_col, real_target_val, data, future_time_intervals
    )

    res = pd.DataFrame(results)
    best_hyperparams = {
        "order_p": res.loc[res["rmse"] == res["rmse"].min()]["p"].to_numpy()[0],
        "order_d": res.loc[res["rmse"] == res["rmse"].min()]["d"].to_numpy()[0],
        "order_q": res.loc[res["rmse"] == res["rmse"].min()]["q"].to_numpy()[0],
        "trend": res.loc[res["rmse"] == res["rmse"].min()]["trend"].to_numpy()[0],
        "pred": res.loc[res["rmse"] == res["rmse"].min()]["preds"].to_numpy()[0],
        "rmse": res.loc[res["rmse"] == res["rmse"].min()]["rmse"].to_numpy()[0],
        "hpt-duration": hpt_duration,
    }

    print(
        f"[D] Time-series hyper-parameter tuning complete! Time-series was fit with {len(data)} points. Best hyper-parameters found:"
    )
    print(
        f"[D]\torder=({best_hyperparams['order_p']},{best_hyperparams['order_d']},{best_hyperparams['order_q']})    trend={best_hyperparams['trend']}    with rmse={best_hyperparams['rmse']}"
    )
    print(
        f"[GREP];{target_col};{len(data)};{best_hyperparams['order_p']};{best_hyperparams['order_d']};{best_hyperparams['order_q']};{best_hyperparams['trend']};{best_hyperparams['pred']};{real_target_val};{best_hyperparams['rmse']};{best_hyperparams['hpt-duration']}"
    )
    return res, best_hyperparams
