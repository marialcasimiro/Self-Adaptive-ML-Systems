#!/usr/bin/env python

"""
#####################################
## EXPERIMENTAL SETTINGS/VARIABLES ##
#####################################
"""

# SYSTEM SLAs
FPR_T = [1]  # in %
RECALL_T = [70]  # in %
SLA_COSTS = [10]


# TACTICS AVAILABLE TO ADAPT THE SYSTEM
ADAPTATION_TACTICS = ["nop", "retrain"]#, "replace"]

# RETRAIN TACTIC SETTINGS
RETRAIN_COSTS = [8]
RETRAIN_LATENCIES = [0] # in hours

# COMPONENT REPLACEMENT TACTIC SETTINGS
REPLACE_COSTS = [5]
REPLACE_TPRS_AVGS = [70]   # RECALL == TPR
REPLACE_TPRS_STDS = [5]
REPLACE_TNRS_AVGS = [97]   # FPR = 1 - TNR
REPLACE_TNRS_STDS = [2]

BASELINES = [
    # "optimum",
    # "no_retrain",
    # "periodic",
    # "reactive",
    # "random",
    # "delta_aip_retrain", # this is AIP
    "delta_cbatc_6",
    "delta_atc_6",
    "delta_delayed_6",
]