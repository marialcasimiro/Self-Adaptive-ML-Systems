import os
import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_dataset_path, get_file

import defs

SIZE = 20

plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE)
plt.rc('font', size=SIZE)
plt.rc('axes', labelsize="medium")
plt.rcParams["figure.figsize"] = (15,11)


palette ={
    'no_retrain': 'black',
    'reactive': 'red',
    'optimum': 'green',
    'AIP': 'magenta',
    'random': 'blue',
    'periodic': 'orange',
    'delayed': 'grey',
    'atc': 'gold',
    'cbatc': 'brown',
    'atc_s': 'yellow',
    'cbatc_s': 'cyan',
}


def compute_total_cost(targets, results, retrain_cost, SLA_cost):
    for target in targets:
        dataset_name = target
        dataset = results[target]
        #print(dataset_name)
        dataset['total_cost'] = dataset['total_sla_violations'] * SLA_cost
        if "no_retrain" in dataset_name:
            continue
        if "periodic" in dataset_name:
            dataset['total_cost'] = dataset['total_cost'].add(retrain_cost)
        else:
            costs = dataset['total_cost'].to_numpy()
            for idx, retrain_occurred in enumerate(results[dataset_name]['retrain_occurred']):
                if retrain_occurred:
                    costs[idx] = costs[idx] + retrain_cost
            print(f"baseline: {target} --- Total costs AFTER retrain\n\t{costs[-98:]}")
            dataset['total_cost'] = costs
        

def get_all_data(
    results,
    baselines, 
    time_interval,
    fpr_thresholds, 
    recall_thresholds, 
    retrain_costs, 
    retrain_latencies, 
    nop_models, 
    sat_value, 
    sla_cost,
    cols,
):
    
    res_dict = {
        "retrain_cost": [],
        "retrain_latency": [],
        "thresholds": [],
        "fpr_t": [],
        "recall_t": [],
        "baseline": [],
        "num_retrains": [],
        "sat_values": [],
        "nop_models": [],
        "sla_cost": [],
    }
    for col in cols:
        res_dict[col] = []
        
    prod = itertools.product(fpr_thresholds, recall_thresholds, retrain_costs, retrain_latencies, baselines)
    for (fpr_t, recall_t, retrain_cost, retrain_latency, baseline) in prod:
    
        if "delta" in baseline or "abs" in baseline:
            ask_prism = True
            models = nop_models
        else:
            ask_prism = False
            models = [False]
            
        for model in models:
            dataset_name = f"timeInterval_{time_interval}-baseline_{baseline}-fprT_{fpr_t}-recallT_{recall_t}-retrainCost_{retrain_cost}-retrainLatency_{retrain_latency}-slaCost_{sla_cost}-satValue_{sat_value}-nopModels_{model}-askPrism_{ask_prism}-seed_1"

            for key, value in results.items():
                if dataset_name not in key:
                    continue

                compute_total_cost(
                    targets=[dataset_name],
                    results=results, 
                    retrain_cost=retrain_cost, 
                    SLA_cost=sla_cost,
                )

                target_retrain = value['retrain_occurred'].to_numpy()[-98:]
                idx = [i for i, x in enumerate(target_retrain) if x]

                for col in cols:
                    res_dict[col].append(np.cumsum(value[col].to_numpy()[-98:])[-1])
                res_dict["retrain_cost"].append(retrain_cost)
                res_dict["retrain_latency"].append(retrain_latency)
                res_dict["thresholds"].append(f"{fpr_t}-{recall_t}")
                res_dict["fpr_t"].append(fpr_t)
                res_dict["recall_t"].append(recall_t)
                res_dict["sat_values"].append(sat_value)
                res_dict["nop_models"].append(model)
                if "delta" in baseline:
                    res_dict["baseline"].append("AIP")
                else:
                    res_dict["baseline"].append(baseline)
                res_dict["sla_cost"].append(sla_cost)
                res_dict["num_retrains"].append(len(idx))

    res_df = pd.DataFrame(res_dict)
    
    return res_df


def get_prism_time_overhead(results):
    overhead_dict = {
        "num_retrains": [],
        "total_overhead_avg": [],
        "total_overhead_std": [],
        "prism_overhead_avg": [],
        "prism_overhead_std": [],
    }
    
    for key, df in results.items():
        if "overall_time_overhead" not in df.columns:
            continue
        
        tokens = key.split("-")
        
        for token in tokens:
            sub_tokens = token.split("_", maxsplit=1)
            k = sub_tokens[0] # key
            v = sub_tokens[1] # value
            if k not in overhead_dict:
                overhead_dict[k] = []
                
            overhead_dict[k].append(v)
        
        time_overhead = df.loc[df["overall_time_overhead"] != -1]["overall_time_overhead"]
        overhead_dict["total_overhead_avg"].append(time_overhead.mean())
        overhead_dict["total_overhead_std"].append(time_overhead.std())
        
        prism_overhead = df.loc[df["prism_time_overhead"] != -1]["prism_time_overhead"]
        overhead_dict["prism_overhead_avg"].append(prism_overhead.mean())
        overhead_dict["prism_overhead_std"].append(prism_overhead.std())
        
        overhead_dict["num_retrains"].append(df["retrain_occurred"].sum())
        
    return pd.DataFrame(overhead_dict)


def plot_context_res(cols, res_df, sat_value, fpr_t, x_axis, save_path):
    
    ORDER = ["no_retrain", "periodic", "reactive", "random", "AIP", "optimum"]
    
    for col in cols:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=x_axis, 
            y=col, 
            hue="baseline",
            palette=palette,
            data=res_df, 
            ci=None,
            hue_order=ORDER,
        )
        
        ax.legend(
            loc='lower center', 
            ncol=3,
        )
        
        if "recall" in x_axis:
            fig_name="fig3a"
        elif "cost" in x_axis:
            fig_name="fig3b"
        else:
            fig_name="fig3c"

        plt.savefig(save_path + fig_name + ".png")

def get_label_and_color(target, num_retrains, time_interval):
    if 'noRetrain' in target or 'no_retrain' in target:
        label = "No retrain (0)"
        color = palette["no_retrain"]
    elif 'reactive' in target:
        label = f'Reactive ({num_retrains})'
        color = palette["reactive"]
    elif 'optimum' in target:
        label = f'Optimum ({num_retrains})'
        color = palette["optimum"]
    elif "delayed" in target:
        label = f'Delayed ({num_retrains})'
        color = palette["delayed"]
    elif "cbatc_small" in target:
        label = f'CB-ATC_s ({num_retrains})'
        color = palette["cbatc_s"]
    elif "cbatc" in target:
        label = f'CB-ATC ({num_retrains})'
        color = palette["cbatc"]
    elif "atc_small" in target:
        label = f'ATC_s ({num_retrains})'
        color = palette["atc_s"]
    elif "atc" in target:
        label = f'ATC ({num_retrains})'
        color = palette["atc"]
    elif "delta" in target:
        label = f'AIP ({num_retrains})'
        color = palette["AIP"]
    elif 'random' in target:
        tokens = target.split("-")[1].split("_")
        if len(tokens) <= 2:
            label = f'Random ({num_retrains})'
        else:
            label = f'Random-{tokens[2]} ({num_retrains})'
        color = palette["random"]
    else:
        label = f'Retrain-{time_interval} ({num_retrains})'
        color = palette["periodic"]
                
    return label, color

def plot_prism_res(retrain_cost, sla_cost, results, targets, save_path, title = None):
    
    compute_total_cost(
        targets=targets,
        results=results, 
        retrain_cost=retrain_cost, 
        SLA_cost=sla_cost, 
    )
    
    time_interval = int(targets[0].split("-")[0].split("_")[1])
    min_len = float("inf")
    
    for target in targets:
        if len(results[target]['total_sla_violations'].to_numpy()) < min_len:
            min_len = len(results[target]['total_sla_violations'].to_numpy())
    
    min_len = 98
    x = range(min_len)

    offset = len(results[targets[0]]['total_sla_violations'].to_numpy()) - min_len

    fraud_txs = []
    fraud_txs_shift = []
    idx = []
    for col in ['total_cost', 'total_sla_violations']:
        if col in results[targets[-1]].columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            counter = 1
            for target in targets:
                y = results[target][col].to_numpy()[-min_len:]
                if 'no_retrain' not in target:
                    if 'retrain_occurred' in results[target].columns:
                        idx = [i for i, x in enumerate(results[target]['retrain_occurred']) if x]

                    idx = [i for i in idx if i >= offset]
                    idx = [i - offset for i in idx]
                    
                    
                label, color = get_label_and_color(target, len(idx), time_interval)
                
                linestyle = 'solid'
                marker = "s"
                
                if 'no_retrain' not in target:
                    ax.scatter(idx, np.cumsum(y)[idx], color=color, marker=marker)
                
                ax.plot(x, np.cumsum(y), label=label, color=color, linestyle=linestyle)
                
                counter += 1
            
            
            ax.set(xlabel='Time interval', ylabel=col)
            ax.legend(
                loc='best', 
                ncol=1,
            )
            ax.grid()
            if title is not None:
                ax.set_title(title)

            if 'total_cost' in col:
                fig_name = "fig2a"
            else:
                fig_name = "fig2b"
            plt.savefig(save_path + fig_name + ".png")





def main(use_pre_generated_files: False):

    # load data
    dataset_path = get_dataset_path(defs.DATASET_NAME)
    path = dataset_path
    if use_pre_generated_files:
        path = dataset_path + "pre-generated/"
    path = path + "results/files/"

    results = {}

    print(f"[D] Loading result files from {path}")
    if len(os.listdir(path)) == 0:
        print(f"[E] Directory {path} is empty. Defaulting to pre-generated files")
        path = dataset_path + "pre-generated/results/files/"

    for f in os.listdir(path):
        if 'baseline' in f:
            key = f[8:-4]
            results[key] = get_file(path, f)


    # get data for plotting
    print(f"[D] Parsing results data")
    res_df = get_all_data(
        results,
        baselines=[
            "optimum", "no_retrain", "periodic", "reactive", "random", "delta_aip_retrain",
            "delayed", "atc", "cbatc", "atc_small", "cbatc_small",
        ],
        time_interval=10,
        fpr_thresholds=[1], 
        recall_thresholds=[50, 60, 70, 80, 90], 
        retrain_costs=[1, 5, 8, 10, 15], 
        retrain_latencies=[0, 1, 5], 
        nop_models=["random_forest"], 
        sat_value=0.9, 
        sla_cost=10,
        cols=["total_cost", "total_sla_violations"],
    )
    print("-"*100)

    # get PRISM time overheads
    print("[D] Computing PRISM time overheads")
    time_overheads_df = get_prism_time_overhead(results)
    time_overheads_df.loc[
        (time_overheads_df['recallT'] == '70')
        & (time_overheads_df['retrainCost'] == '8')
        & (time_overheads_df['retrainLatency'] == '0')
        & (time_overheads_df['baseline'] == 'delta_retrain')
    ].to_csv(dataset_path + "results/figures/prism_time_overheads.csv", index=False)
    print(f"[D] PRISM time overheads saved to\n{dataset_path}results/prism_time_overheads.csv")

    # generate figure 2
    print("-"*100)
    print("[D] Generating Figures 2a and 2b")
    TIME_INTERVAL = 10
    RETRAIN_COST = 8
    RETRAIN_LATENCY = 0
    FPR_T = 1
    RECALL_T = 70
    SLA_COST = 10
    SAT_VALUE = 0.9
    NOP_MODELS = False
    ASK_PRISM = False
    DELAY = 2

    settings = f"fprT_{FPR_T}-recallT_{RECALL_T}-retrainCost_{RETRAIN_COST}-retrainLatency_{RETRAIN_LATENCY}-slaCost_{SLA_COST}-satValue_{SAT_VALUE}"

    targets = [
        f'timeInterval_{TIME_INTERVAL}-baseline_no_retrain-{settings}-nopModels_False-askPrism_False-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_periodic-{settings}-nopModels_False-askPrism_False-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_reactive-{settings}-nopModels_False-askPrism_False-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_random-{settings}-nopModels_False-askPrism_False-seed_1',    
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_aip_retrain-{settings}-nopModels_random_forest-askPrism_True-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_optimum-{settings}-nopModels_False-askPrism_False-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_delayed_{DELAY}-{settings}-nopModels_random_forest-askPrism_True-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_atc_{DELAY}-{settings}-nopModels_random_forest-askPrism_True-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_cbatc_{DELAY}-{settings}-nopModels_random_forest-askPrism_True-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_atc_small_{DELAY}-{settings}-nopModels_random_forest-askPrism_True-seed_1',
        f'timeInterval_{TIME_INTERVAL}-baseline_delta_cbatc_small_{DELAY}-{settings}-nopModels_random_forest-askPrism_True-seed_1',
    ]

    plot_prism_res(
        retrain_cost=RETRAIN_COST, 
        sla_cost=SLA_COST, 
        results=results, 
        targets=targets,
        save_path=dataset_path + "results/figures/",
    )
    print(f"[D] Figures 2a and 2b saved to {dataset_path}results/figures/")



    # generate figure 3.a)
    print("-"*100)
    print("[D] Generating Figure 3a")
    X_AXIS = "recall_t"
    RETRAIN_COST = 8
    RETRAIN_LATENCY = 0

    plot_context_res(
        cols=["total_cost"],
        res_df=res_df.loc[
            (res_df['retrain_cost'] == RETRAIN_COST)
            & (res_df['retrain_latency'] == RETRAIN_LATENCY)
        ],
        sat_value=0.9,
        fpr_t=1,
        x_axis=X_AXIS,
        save_path=dataset_path + "results/figures/",
    )
    print(f"[D] Figure 3a saved to {dataset_path}results/figures/")

    # generate figure 3.b)
    print("-"*100)
    print("[D] Generating Figure 3b")
    X_AXIS = "retrain_cost"
    RECALL_T = 70
    RETRAIN_LATENCY = 0

    plot_context_res(
        cols=["total_cost"],
        res_df=res_df.loc[
            (res_df['recall_t'] == RECALL_T)
            & (res_df['retrain_latency'] == RETRAIN_LATENCY)
        ],
        sat_value=0.9,
        fpr_t=1,
        x_axis=X_AXIS,
        save_path=dataset_path + "results/figures/",
    )
    print(f"[D] Figure 3b saved to {dataset_path}results/figures/")

    # generate figure 3.c)
    print("-"*100)
    print("[D] Generating Figure 3c")
    X_AXIS = "retrain_latency"
    RECALL_T = 70
    RETRAIN_COST = 8

    plot_context_res(
        cols=["total_cost"],
        res_df=res_df.loc[
            (res_df['recall_t'] == RECALL_T)
            & (res_df['retrain_cost'] == RETRAIN_COST)
        ],
        sat_value=0.9,
        fpr_t=1,
        x_axis=X_AXIS,
        save_path=dataset_path + "results/figures/",
    )
    print(f"[D] Figure 3c saved to {dataset_path}results/figures/")

if __name__ == "__main__":

    use_pre_generated_files = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pgf", help="use pre-generated files", action="store_true")
    args = parser.parse_args()

    if args.use_pgf:
        print("[D] Using pre-generated files")
        use_pre_generated_files = True

    main(use_pre_generated_files)
