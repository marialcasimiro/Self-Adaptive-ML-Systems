#!/usr/bin/env python

import itertools
from copy import deepcopy

import defs
from dataset import Dataset
from utils import save_results


def main():

    print(
        "[D] --------------------------- GENERATE RETRAIN FILES ---------------------------"
    )
    for seed in range(1, defs.NUM_RUNS + 1):
        print(
            f"[D] --------------------------- Run {seed} / {defs.NUM_RUNS} ---------------------------"
        )

        for time_interval in defs.TIME_INTERVALS:
            ieee_cis = Dataset(time_interval=time_interval, name=defs.DATASET_NAME)
            clf = ieee_cis.get_initial_model(seed)

            print(
                f"\n[D] Starting testing with time interval of {time_interval} hours"
            )
            times_df, base_save_path, metrics_df = ieee_cis.test(
                time_interval=time_interval,
                model=deepcopy(clf),
            )

            save_results(
                base_save_path + "tmp/",
                f"metrics-timeInterval_{time_interval}-noRetrain-seed_{seed}.pkl",
                metrics_df,
            )

            save_results(
                base_save_path + "tmp/",
                f"times-timeInterval_{time_interval}-noRetrain-seed_{seed}.pkl",
                times_df,
            )

            retrain_periods = list(
                range(time_interval, defs.MAX_TIME, time_interval)
            )
            print(f"[D] Testing retrain periods: {retrain_periods}")
            prod = itertools.product(retrain_periods, defs.RETRAIN_MODES)
            for (retrain_period, retrain_mode) in prod:
                print(
                    f"\n[D] Starting testing with time interval of {time_interval} and retrain period of {retrain_period} hours"
                )
                times_df, base_save_path, metrics_df = ieee_cis.test(
                    time_interval=time_interval,
                    model=deepcopy(clf),
                    test_prism=False,
                    do_retrain=True,
                    retrain_period=retrain_period,
                    retrain_mode=retrain_mode,
                )
                save_results(
                    base_save_path + "tmp/",
                    f"metrics-timeInterval_{time_interval}-retrainPeriodHours_{retrain_period}-retrainMode_{retrain_mode}-seed_{seed}.pkl",
                    metrics_df,
                )

                save_results(
                    base_save_path + "tmp/",
                    f"times-timeInterval_{time_interval}-retrainPeriodHours_{retrain_period}-retrainMode_{retrain_mode}-seed_{seed}.pkl",
                    times_df,
                )


if __name__ == "__main__":
    main()
