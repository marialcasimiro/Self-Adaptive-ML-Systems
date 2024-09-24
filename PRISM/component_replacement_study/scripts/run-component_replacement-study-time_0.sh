#!/usr/bin/env bash

# the values in this simulation are made up

PRISM="/Users/marialcasimiro/Downloads/prism-4.6-osx64/bin/prism"
BASE_DIR="/Users/marialcasimiro/IST/PhD/4_ano/ACSOS22-ML-Adaptation-Framework/SA4ML/PRISM/"
MODEL="${BASE_DIR}model/system_model-component_replacement_study.prism"
PROPERTIES="${BASE_DIR}model/properties.props"

RESULTS_DIR="${BASE_DIR}component_replacement_study/results/"

# PRISM OPTIONS
CUDD_MEM=16g
JAVA_MEM=12g
JAVA_STACK=1g
# available engines:
# - hybrid (default)
# - sparse
# - MTBDD
# - explicit
ENGINE=explicit

APPROXIMATE_MODEL_CHECKING=false
# available approximate model checking methods
# - ci:    confidence inteval
# - aci:   asymptotic confidence interval
SIMULATION_METHOD=ci

# TACTICS
NOP=0
ALL=1
RETRAIN=2
REPLACE=3

tactics=$ALL

SNAPSHOT_TIME=0

HORIZON=1

CURR_NEW_DATA=313
RETRAIN_LATENCY=0
AVG_TRANSACTIONS=764
PERCENT_TXS=0

INIT_TNR=98
INIT_TPR=75
INIT_FNR=25
INIT_FPR=2

FPR_SLA_COST=10
RECALL_SLA_COST=10
FPR_THRESHOLD=1
RECALL_THRESHOLD=70
#RETRAIN_COST=8

new_TPR_retrain=0
# new_TNR_retrain=0
new_TPR_noRetrain=1
new_TNR_noRetrain=0

new_TPR_retrain_std=0
new_TNR_retrain_std=0
new_TPR_noRetrain_std=0
new_TNR_noRetrain_std=0

new_TPR_retrain_5=0
new_TPR_retrain_50=0
new_TPR_retrain_95=0

# new_TNR_retrain_5=0
# new_TNR_retrain_50=0
# new_TNR_retrain_95=0

new_TPR_noRetrain_5=1
new_TPR_noRetrain_50=1
new_TPR_noRetrain_95=1

new_TNR_noRetrain_5=0
new_TNR_noRetrain_50=0
new_TNR_noRetrain_95=0

BENEFITS_MODEL_TYPE=0
NO_RETRAIN_MODELS=true

REPLACE_COST=15
rbModelTPR=75
# rbModelTNR=99

settings="BENEFITS_MODEL_TYPE=${BENEFITS_MODEL_TYPE},NO_RETRAIN_MODELS=${NO_RETRAIN_MODELS},HORIZON=${HORIZON},RETRAIN_LATENCY=${RETRAIN_LATENCY},AVG_TRANSACTIONS=${AVG_TRANSACTIONS},PERCENT_TXS=${PERCENT_TXS},CURR_NEW_DATA=${CURR_NEW_DATA}"

threshold_cost_settings="FPR_SLA_COST=${FPR_SLA_COST},RECALL_SLA_COST=${RECALL_SLA_COST},FPR_THRESHOLD=${FPR_THRESHOLD},RECALL_THRESHOLD=${RECALL_THRESHOLD}"

curr_model_perf_values="INIT_TPR=${INIT_TPR},INIT_TNR=${INIT_TNR},INIT_FPR=${INIT_FPR},INIT_FNR=${INIT_FNR}"

new_TPR_retrain_values="new_TPR_retrain=${new_TPR_retrain},new_TPR_retrain_std=${new_TPR_retrain_std},new_TPR_retrain_5=${new_TPR_retrain_5},new_TPR_retrain_50=${new_TPR_retrain_50},new_TPR_retrain_95=${new_TPR_retrain_95}"
new_TPR_noRetrain_values="new_TPR_noRetrain=${new_TPR_noRetrain},new_TPR_noRetrain_std=${new_TPR_noRetrain_std},new_TPR_noRetrain_5=${new_TPR_noRetrain_5},new_TPR_noRetrain_50=${new_TPR_noRetrain_50},new_TPR_noRetrain_95=${new_TPR_noRetrain_95}"
new_TNR_noRetrain_values="new_TNR_noRetrain=${new_TNR_noRetrain},new_TNR_noRetrain_std=${new_TNR_noRetrain_std},new_TNR_noRetrain_5=${new_TNR_noRetrain_5},new_TNR_noRetrain_50=${new_TNR_noRetrain_50},new_TNR_noRetrain_95=${new_TNR_noRetrain_95}"

# Experiment variables
tactics=(0 2 3)
retrainCosts=(8) #1 8 16)

for retrainCost in "${retrainCosts[@]}"
do  
    for (( new_TNR_retrain=-8; new_TNR_retrain<=2; new_TNR_retrain=new_TNR_retrain+1 ))
    do  
        for (( rbModelTNR=90; rbModelTNR<=100; rbModelTNR=rbModelTNR+1 ))
        do
            for tactic in "${tactics[@]}"
            do
                if [[ "$tactic" == "$NOP" ]]; then
                    TACTIC=nop
                elif [ "$tactic" == "$RETRAIN" ]; then
                    TACTIC=retrain
                else
                    TACTIC=replace
                fi
                printf "###################################################################################################################\n"
                printf "new_TNR_retrain: ${new_TNR_retrain}   rb_model_TNR: ${rbModelTNR}   rb_model_TPR: ${rbModelTPR}   replace cost: ${REPLACE_COST}   retrain cost: ${retrainCost}   tactic: ${TACTIC}\n"
                file_name="${RESULTS_DIR}${TACTIC}/adv-snapshotTime_${SNAPSHOT_TIME}-rbModelTPR_${rbModelTPR}-rbModelTNR_${rbModelTNR}-replaceCost_${REPLACE_COST}-retrainCost_${retrainCost}-newTNRretrain_${new_TNR_retrain}-tactic_${TACTIC}"
                printf "command:\n\t${PRISM} ${MODEL} ${PROPERTIES} -prop 2 -const RETRAIN_COST=${retrainCost},rb_model_TPR=${rbModelTPR},rb_model_TNR=${rbModelTNR},$settings,${threshold_cost_settings},${curr_model_perf_values},${new_TPR_retrain_values},${new_TNR_retrain_values},${new_TPR_noRetrain_values},${new_TNR_noRetrain_values} -exportresults ${file_name}.txt -exportadvmdp ${file_name}.tra -${ENGINE}"
                new_TNR_retrain_5=$new_TNR_retrain
                new_TNR_retrain_50=$new_TNR_retrain
                new_TNR_retrain_95=$new_TNR_retrain
                new_TNR_retrain_values="new_TNR_retrain=${new_TNR_retrain},new_TNR_retrain_std=${new_TNR_retrain_std},new_TNR_retrain_5=${new_TNR_retrain_5},new_TNR_retrain_50=${new_TNR_retrain_50},new_TNR_retrain_95=${new_TNR_retrain_95}"
                $PRISM $MODEL $PROPERTIES -prop 2 -const TACTICS=${tactic},RETRAIN_COST=$retrainCost,REPLACE_COST=$REPLACE_COST,rb_model_TPR=$rbModelTPR,rb_model_TNR=$rbModelTNR,$settings,$threshold_cost_settings,$curr_model_perf_values,$new_TPR_retrain_values,$new_TNR_retrain_values,$new_TPR_noRetrain_values,$new_TNR_noRetrain_values -exportresults $file_name.txt -exportadvmdp $file_name.tra -$ENGINE
            done
        done
    done
done