#!/usr/bin/env python

import argparse
import defs
from dataset import Dataset


def main(use_pgf: False):

    print(
        "[D] --------------------------- GENERATING RETRAIN DATASET ---------------------------"
    )
    for time_interval in defs.TIME_INTERVALS:
        ieee_cis = Dataset(time_interval=time_interval, name=defs.DATASET_NAME)
        retrain_periods = list(range(time_interval, defs.MAX_TIME, time_interval))
        ieee_cis.generate_new_dataset(
            time_interval, retrain_periods, sampling_method="rand", use_pgf=use_pgf
        )


if __name__ == "__main__":

    use_pre_generated_files = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pgf", help="use pre-generated files", action="store_true")
    args = parser.parse_args()

    if args.use_pgf:
        print("[D] Using pre-generated files")
        use_pre_generated_files = True

    main(use_pre_generated_files)
